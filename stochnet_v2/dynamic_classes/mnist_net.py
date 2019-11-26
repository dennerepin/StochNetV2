import numpy as np
import tensorflow as tf
import json
import logging
import os
import pickle
import shutil
from functools import partial

import stochnet_v2.static_classes.nn_bodies as nn_bodies
from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.utils.errors import NotRestoredVariables
import stochnet_v2.dynamic_classes.nn_body_search as nn_body_search
import stochnet_v2.dynamic_classes.nn_body as nn_body
from stochnet_v2.dynamic_classes.nn_body_search import get_genotypes
from stochnet_v2.utils.registry import CONSTRAINTS_REGISTRY
from stochnet_v2.utils.registry import REGULARIZERS_REGISTRY


LOGGER = logging.getLogger('dynamic_classes.mnist_model')


def _body_search(x, **kwargs):
    shape = x.shape.as_list()
    x = tf.reshape(x, shape=(shape[0] or -1, np.prod(shape[1:]),))
    return nn_body_search.body(x, **kwargs)


def _body_trained(x, **kwargs):
    shape = x.shape.as_list()
    x = tf.reshape(x, shape=(shape[0] or -1, np.prod(shape[1:]),))
    return nn_body.body(x, **kwargs)


class MnistNet:

    def __init__(
            self,
            nb_features,
            project_folder,
            timestep,
            dataset_id,
            model_id,
            body_config_path=None,
            ckpt_path=None,
            mode='normal',
    ):
        self.nb_features = nb_features
        self.timestep = timestep

        self.project_explorer = ProjectFileExplorer(project_folder)
        self.dataset_explorer = self.project_explorer.get_dataset_file_explorer(self.timestep, dataset_id)
        self.model_explorer = self.project_explorer.get_model_file_explorer(self.timestep, model_id)

        self._input_placeholder = None
        self._input_placeholder_name = None
        self._pred_placeholder = None
        self._pred_placeholder_name = None

        self._pred_tensor = None
        self._pred_tensor_name = None

        self.restored = False

        self.graph = tf.compat.v1.Graph()

        with self.graph.as_default():

            self.session = tf.compat.v1.Session()

            if mode == 'normal':
                self._init_normal(body_config_path, ckpt_path)

            elif mode == 'inference':
                self._load_model_from_frozen_graph()

            elif mode == 'inference_ckpt':
                self._load_model_from_checkpoint(ckpt_path)

            else:
                raise ValueError(
                    "Unknown keyword for `mode` parameter. Use 'normal', 'inference' or 'inference_ckpt'"
                )

            LOGGER.info(f"Model created in {mode} mode.")

    def _init_normal(
            self,
            body_config_path,
            ckpt_path,
    ):
        if body_config_path is None:
            body_config = self._read_config(self.model_explorer.body_config_fp)
        else:
            body_config = self._read_config(
                body_config_path,
                os.path.basename(self.model_explorer.body_config_fp)
            )

        body_fn = self._get_body_fn(body_config)
        self._build_main_graph(body_fn)
        self._save_graph_keys()
        if ckpt_path:
            self.restore_from_checkpoint(ckpt_path)

    def _read_config(self, file_path, model_dir_save_name=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, 'r') as f:
            config = json.load(f)

        if model_dir_save_name:
            save_path = os.path.join(self.model_explorer.model_folder, model_dir_save_name)
            with open(save_path, 'w') as f:
                json.dump(config, f, indent='\t')

        return config

    def _get_body_fn(self, body_config):

        self.n_cells = body_config['n_cells']
        self.expansion_multiplier = body_config['expansion_multiplier']
        self.cell_size = body_config['cell_size']
        self.n_states_reduce = body_config['n_states_reduce']

        kernel_constraint = body_config.pop("kernel_constraint", "none")
        body_config["kernel_constraint"] = CONSTRAINTS_REGISTRY[kernel_constraint]

        kernel_regularizer = body_config.pop("kernel_regularizer", "none")
        body_config["kernel_regularizer"] = REGULARIZERS_REGISTRY[kernel_regularizer]

        bias_constraint = body_config.pop("bias_constraint", "none")
        body_config["bias_constraint"] = CONSTRAINTS_REGISTRY[bias_constraint]

        bias_regularizer = body_config.pop("bias_regularizer", "none")
        body_config["bias_regularizer"] = REGULARIZERS_REGISTRY[bias_regularizer]

        activity_regularizer = body_config.pop("activity_regularizer", "none")
        body_config["activity_regularizer"] = REGULARIZERS_REGISTRY[activity_regularizer]

        return partial(_body_search, **body_config)

    def _build_main_graph(self, body_fn):
        self.input_placeholder = tf.compat.v1.placeholder(
            tf.float32, (None, self.nb_features), name="input"
        )
        self.pred_placeholder = tf.compat.v1.placeholder(
            tf.float32, (None,), name="pred_placeholder"
        )
        body_output = body_fn(self.input_placeholder)
        self.pred_tensor = tf.compat.v1.layers.Dense(10, activation='softmax', name='pred_tensor')(body_output)

        self.loss = self._loss_function(self.pred_placeholder, self.pred_tensor)

    @staticmethod
    def _loss_function(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=False,
            axis=-1
        )

    def restore_from_checkpoint(self, ckpt_path):
        with self.graph.as_default():
            variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            saver = tf.compat.v1.train.Saver(var_list=variables)
            saver.restore(self.session, ckpt_path)
        self.restored = True

    def save(self):
        if not self.restored:
            raise NotRestoredVariables()
        self._save_frozen_graph()
        self._save_graph_keys()

    def _save_frozen_graph(self):
        frozen_graph_def = self._freeze_graph()
        tf.compat.v1.train.write_graph(
            frozen_graph_def,
            logdir=self.model_explorer.model_folder,
            name=os.path.basename(self.model_explorer.frozen_graph_fp),
            as_text=False,
        )
        LOGGER.info(f"Model's frozen graph saved in {self.model_explorer.model_folder}")

    def _freeze_graph(self):
        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=self.session,
            input_graph_def=self.graph.as_graph_def(),
            output_node_names=self.dest_nodes,
        )
        return frozen_graph_def

    def _save_graph_keys(self):
        graph_keys_dict = {
            'input_placeholder': self._input_placeholder_name,
            'pred_tensor': self._pred_tensor_name,
            'pred_placeholder':  self._pred_placeholder_name,
        }

        with open(self.model_explorer.graph_keys_fp, 'w') as f:
            json.dump(graph_keys_dict, f, indent='\t')
        LOGGER.info(f"Model's graph keys saved at {self.model_explorer.graph_keys_fp}")

    def _load_model_from_frozen_graph(self):
        graph_path = self.model_explorer.frozen_graph_fp
        if not os.path.exists(graph_path):
            raise FileNotFoundError(
                f"Could not find model's frozen graph file: {graph_path}. Did you save the model?"
            )
        graph_def = tf.compat.v1.GraphDef()
        with open(graph_path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        self._load_graph_keys()
        self.restored = True

    def _load_model_from_checkpoint(self, ckpt_path):
        if ckpt_path is None:
            raise ValueError("Should provide `ckpt_path` to build model")
        meta_ckpt_path = ckpt_path + '.meta'
        if not os.path.exists(meta_ckpt_path):
            raise FileNotFoundError(
                f"Could not find model's checkpoint: {meta_ckpt_path}."
            )
        saver = tf.compat.v1.train.import_meta_graph(meta_ckpt_path)
        saver.restore(self.session, ckpt_path)
        self._load_graph_keys()
        self.restored = True

    def _load_graph_keys(self):
        graph_keys_path = self.model_explorer.graph_keys_fp
        if not os.path.exists(graph_keys_path):
            raise FileNotFoundError(
                f"Could not find model's graph keys file: {graph_keys_path}. Did you save the model?"
            )
        with open(graph_keys_path, 'r') as f:
            graph_keys = json.load(f)

        self.input_placeholder = self.graph.get_tensor_by_name(graph_keys['input_placeholder'])
        self.pred_tensor = self.graph.get_tensor_by_name(graph_keys['pred_tensor'])
        self.pred_placeholder = self.graph.get_tensor_by_name(graph_keys['pred_placeholder'])

    @property
    def input_placeholder(self):
        return self._input_placeholder

    @input_placeholder.setter
    def input_placeholder(self, tensor):
        self._input_placeholder = tensor
        self._input_placeholder_name = tensor.name

    @property
    def pred_tensor(self):
        return self._pred_tensor

    @pred_tensor.setter
    def pred_tensor(self, tensor):
        self._pred_tensor = tensor
        self._pred_tensor_name = tensor.name

    @property
    def pred_placeholder(self):
        return self._pred_placeholder

    @pred_placeholder.setter
    def pred_placeholder(self, tensor):
        self._pred_placeholder = tensor
        self._pred_placeholder_name = tensor.name

    @property
    def dest_nodes(self):
        return [t.split(':')[0] for t in [self._pred_tensor_name, self._pred_placeholder_name]]

    def _copy_dataset_scaler(self):
        print('Copying dataset scaler to model dir...')
        scaler_fp = os.path.join(self.model_explorer.model_folder, 'scaler.pickle')
        shutil.copy2(
            self.dataset_explorer.scaler_fp,
            scaler_fp,
        )

    def predict(self, input_values):

        if not self.restored:
            raise NotRestoredVariables()

        prediction_values = self.session.run(
            self._pred_tensor_name,
            feed_dict={
                self._input_placeholder_name: input_values
            }
        )
        return np.argmax(prediction_values, axis=-1)

    def save_genotypes(self, n_cells=None, cell_size=None, n_states_reduce=None):
        n_cells = n_cells or self.n_cells
        cell_size = cell_size or self.cell_size
        n_states_reduce = n_states_reduce or self.n_states_reduce
        genotypes = get_genotypes(self.session, n_cells=n_cells, cell_size=cell_size, n_states_reduce=n_states_reduce)
        with open(os.path.join(self.model_explorer.model_folder, 'genotypes.pickle'), 'wb') as f:
            pickle.dump(genotypes, f)

    def _load_genotypes(self):
        with open(os.path.join(self.model_explorer.model_folder, 'genotypes.pickle'), 'rb') as f:
            genotypes = pickle.load(f)
        return genotypes

    def recreate_from_genome(self, ckpt_path):
        self.session.close()
        del self.graph
        del self.session
        self._sample_tensor = None

        body_config_path = os.path.join(self.model_explorer.model_folder, 'body_config.json')
        body_config = self._read_config(body_config_path)

        expansion_multiplier = body_config['expansion_multiplier']
        genotypes = self._load_genotypes()

        body_fn = partial(
            _body_trained,
            genotypes=genotypes,
            expansion_multiplier=expansion_multiplier
        )

        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            self.session = tf.compat.v1.Session()
            self._build_main_graph(body_fn)
            self._save_graph_keys()
            self.custom_restore_from_checkpoint(ckpt_path)

    def custom_restore_from_checkpoint(self, ckpt_path):
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
        model_variables = self.graph.get_collection('variables')
        print("Custom restore")
        for v in model_variables:
            name = v.name.replace(':0', '')
            print(name, v.shape, reader.get_tensor(name).shape)
            self.session.run(v.assign(reader.get_tensor(name)))
        self.restored = True
