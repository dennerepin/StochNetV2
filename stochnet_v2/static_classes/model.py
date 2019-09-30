import numpy as np
import tensorflow as tf
import json
import logging
import os
import pickle
import shutil
from collections import namedtuple
from functools import partial
from sklearn.preprocessing.data import StandardScaler, MinMaxScaler
from tqdm import tqdm

from stochnet_v2.static_classes import nn_bodies
from stochnet_v2.static_classes.top_layers import MixtureOutputLayer
from stochnet_v2.static_classes.top_layers import MIXTURE_COMPONENTS_REGISTRY
from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.utils.errors import NotRestoredVariables
from stochnet_v2.utils.registry import ACTIVATIONS_REGISTRY
from stochnet_v2.utils.registry import CONSTRAINTS_REGISTRY
from stochnet_v2.utils.registry import REGULARIZERS_REGISTRY
from stochnet_v2.utils.util import postprocess_description_dict
from stochnet_v2.utils.util import visualize_description


LOGGER = logging.getLogger('static_classes.model')


ComponentDescription = namedtuple('ComponentDescription', ['name', 'parameters'])


def _get_mixture(config, sample_space_dimension):

    categorical = None
    components = []
    descriptions = [ComponentDescription(name, params) for (name, params) in config]

    for description in descriptions:

        kwargs = {}
        component_class = MIXTURE_COMPONENTS_REGISTRY[description.name]

        for key, val in description.parameters.items():
            if 'activation' in key:
                kwargs[key] = ACTIVATIONS_REGISTRY[val]
            if 'hidden_size' in key:
                kwargs[key] = int(val) if val != 'none' else None
            if 'constraint' in key:
                kwargs[key] = CONSTRAINTS_REGISTRY[val]
            if 'regularizer' in key:
                kwargs[key] = REGULARIZERS_REGISTRY[val]

        if description.name == 'categorical':
            categorical = component_class(number_of_classes=len(descriptions) - 1, **kwargs)
        else:
            component = component_class(sample_space_dimension=sample_space_dimension, **kwargs)
            components.append(component)

    if categorical is None:
        LOGGER.warning(
            "Couldn't find description for Categorical random variable, "
            "will initialize it with default parameters"
        )
        categorical = MIXTURE_COMPONENTS_REGISTRY['categorical'](number_of_classes=len(descriptions))

    return MixtureOutputLayer(categorical, components)


class StochNet:

    def __init__(
            self,
            nb_past_timesteps,
            nb_features,
            project_folder,
            timestep,
            dataset_id,
            model_id,
            body_config_path=None,
            mixture_config_path=None,
            ckpt_path=None,
            mode='normal',
    ):
        self.nb_past_timesteps = nb_past_timesteps
        self.nb_features = nb_features
        self.timestep = timestep

        self.project_explorer = ProjectFileExplorer(project_folder)
        self.dataset_explorer = self.project_explorer.get_dataset_file_explorer(self.timestep, dataset_id)
        self.model_explorer = self.project_explorer.get_model_file_explorer(self.timestep, model_id)

        self._input_placeholder = None
        self._input_placeholder_name = None
        self._pred_tensor = None
        self._pred_tensor_name = None
        self._pred_placeholder = None
        self._pred_placeholder_name = None
        self._sample_shape_placeholder = None
        self._sample_shape_placeholder_name = None
        self._sample_tensor = None
        self._sample_tensor_name = None
        self._description_graphkeys = None
        self.restored = False

        self.graph = tf.compat.v1.Graph()

        with self.graph.as_default():

            self.session = tf.compat.v1.Session()

            if mode == 'normal':
                self._init_normal(body_config_path, mixture_config_path, ckpt_path)
                self._copy_dataset_scaler()

            elif mode == 'inference':
                self._load_model_from_frozen_graph()

            elif mode == 'inference_ckpt':
                self._load_model_from_checkpoint(ckpt_path)

            else:
                raise ValueError(
                    "Unknown keyword for `mode` parameter. Use 'normal', 'inference' or 'inference_ckpt'"
                )

            LOGGER.info(f"Model created in {mode} mode.")

        self.scaler = self._load_scaler()

    def _init_normal(
            self,
            body_config_path,
            mixture_config_path,
            ckpt_path,
    ):
        if body_config_path is None:
            raise ValueError("Should provide `body_config_path` to build model")
        body_config = self._read_config(body_config_path, 'body_config.json')

        if mixture_config_path is None:
            raise ValueError("Should provide `mixture_config_path` to build model")
        mixture_config = self._read_config(mixture_config_path, 'mixture_config.json')

        body_fn = self._get_body_fn(body_config)
        self._build_main_graph(body_fn, mixture_config)
        self._build_sampling_graph()
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
        return partial(nn_bodies.body_main, **body_config)

    def _build_main_graph(self, body_fn, mixture_config_path):
        self.input_placeholder = tf.compat.v1.placeholder(
            tf.float32, (None, self.nb_past_timesteps, self.nb_features), name="input"
        )
        self.rv_output_ph = tf.compat.v1.placeholder(
            tf.float32, (None, self.nb_features), name="random_variable_output"
        )
        body = body_fn(self.input_placeholder)
        self.top_layer_obj = _get_mixture(mixture_config_path, sample_space_dimension=self.nb_features)
        self.pred_tensor = self.top_layer_obj.add_layer_on_top(body)
        self.loss = self.top_layer_obj.loss_function(self.rv_output_ph, self.pred_tensor)

    def _build_sampling_graph(self):
        if self.sample_tensor is not None:
            return
        self.top_layer_obj.build_sampling_graph(graph=self.graph)
        self.pred_placeholder = self.top_layer_obj.pred_placeholder
        self.sample_shape_placeholder = self.top_layer_obj.sample_shape_placeholder
        self.sample_tensor = self.top_layer_obj.sample_tensor
        self.description_graphkeys = self.top_layer_obj.description_graphkeys

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
            'sample_shape_placeholder': self._sample_shape_placeholder_name,
            'sample_tensor':  self._sample_tensor_name,
            'description_graphkeys': self.description_graphkeys
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
        self.sample_shape_placeholder = self.graph.get_tensor_by_name(graph_keys['sample_shape_placeholder'])
        self.sample_tensor = self.graph.get_tensor_by_name(graph_keys['sample_tensor'])
        self.description_graphkeys = graph_keys['description_graphkeys']

    def get_description(
            self,
            nn_prediction_val=None,
            current_state_val=None,
            current_state_rescaled=False,
            visualize=False,
    ):
        if nn_prediction_val is None:
            if current_state_val is None:
                raise ValueError("Should provide either current_state_val or nn_prediction_val")
            if not current_state_rescaled:
                current_state_val = self.rescale(current_state_val)
            nn_prediction_val = self.session.run(
                self.pred_tensor,
                feed_dict={self.input_placeholder: current_state_val}
            )
        description = self.session.run(
            self.description_graphkeys,
            feed_dict={
                self.pred_placeholder: nn_prediction_val,
                self.sample_shape_placeholder: 1
            }
        )

        description = postprocess_description_dict(description)

        if visualize:
            visualize_description(description)

        return description

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
    def sample_shape_placeholder(self):
        return self._sample_shape_placeholder

    @sample_shape_placeholder.setter
    def sample_shape_placeholder(self, tensor):
        self._sample_shape_placeholder = tensor
        self._sample_shape_placeholder_name = tensor.name

    @property
    def sample_tensor(self):
        return self._sample_tensor

    @sample_tensor.setter
    def sample_tensor(self, tensor):
        self._sample_tensor = tensor
        self._sample_tensor_name = tensor.name

    @property
    def dest_nodes(self):
        return [t.split(':')[0] for t in [self._sample_tensor_name, self._pred_tensor_name]]

    @property
    def description_graphkeys(self):
        return self._description_graphkeys

    @description_graphkeys.setter
    def description_graphkeys(self, graphkeys):
        self._description_graphkeys = graphkeys

    def _copy_dataset_scaler(self):
        print('Copying dataset scaler to model dir...')
        scaler_fp = os.path.join(self.model_explorer.model_folder, 'scaler.pickle')
        shutil.copy2(
            self.dataset_explorer.scaler_fp,
            scaler_fp,
        )

    def _load_scaler(self):
        scaler_fp = os.path.join(self.model_explorer.model_folder, 'scaler.pickle')
        with open(scaler_fp, 'rb') as file:
            scaler = pickle.load(file)
        return scaler

    # def load_scaler(self):
    #     with open(self.dataset_explorer.scaler_fp, 'rb') as file:
    #         scaler = pickle.load(file)
    #     return scaler

    # def _save_scaler(self):
    #     scaler_fp = os.path.join(self.model_explorer.model_folder, 'scaler.pickle')
    #     with open(scaler_fp, 'wb') as file:
    #         pickle.dump(self.scaler, file)

    def rescale(self, data):
        if isinstance(self.scaler, StandardScaler):
            data = (data - self.scaler.mean_) / self.scaler.scale_
        elif isinstance(self.scaler, MinMaxScaler):
            data = (data * self.scaler.scale_) + self.scaler.min_
        return data

    def scale_back(self, data):
        if isinstance(self.scaler, StandardScaler):
            data = data * self.scaler.scale_ + self.scaler.mean_
        elif isinstance(self.scaler, MinMaxScaler):
            data = (data - self.scaler.min_) / self.scaler.scale_
        return data

    def predict(self, curr_state_values):

        if not self.restored:
            raise NotRestoredVariables()

        prediction_values = self.session.run(
            self._pred_tensor_name,
            feed_dict={
                self._input_placeholder_name: curr_state_values
            }
        )
        return prediction_values

    def sample(self, prediction_values, sample_shape=()):
        if self.sample_tensor is None:
            self._build_sampling_graph()

        sample = self.session.run(
            self._sample_tensor_name,
            feed_dict={
                self._pred_placeholder_name: prediction_values,
                self._sample_shape_placeholder_name: sample_shape,
            }
        )
        sample = np.expand_dims(sample, -2)
        return sample

    def next_state(
            self,
            curr_state_values,
            curr_state_rescaled=False,
            scale_back_result=True,
            round_result=False,
            n_samples=1,
    ):
        # curr_state_values ~ [n_settings, 1, nb_features]
        if not curr_state_rescaled:
            curr_state_values = self.rescale(curr_state_values)

        nn_prediction_values = self.predict(curr_state_values)
        next_state = self.sample(nn_prediction_values, sample_shape=(n_samples,))

        if scale_back_result:
            next_state = self.scale_back(next_state)
            if round_result:
                next_state = np.around(next_state)

        # next_state ~ [n_samples, n_settings, 1, nb_features]
        return next_state

    def generate_traces(
            self,
            curr_state_values,
            n_steps,
            n_traces=1,
            curr_state_rescaled=False,
            scale_back_result=True,
            round_result=False,
            add_timestamps=False,
            batch_size=150,
    ):
        n_settings, *state_shape = curr_state_values.shape

        traces_final_shape = (n_steps + 1, n_traces, n_settings, *state_shape)
        traces_tmp_shape = (n_steps + 1, n_settings * n_traces, *state_shape)

        traces = np.zeros(traces_tmp_shape, dtype=np.float32)

        if not curr_state_rescaled:
            curr_state_values = self.rescale(curr_state_values)

        curr_state_values = np.tile(curr_state_values, [n_traces, 1, 1])  # (n_settings * n_traces, *state_shape)
        traces[0] = curr_state_values

        zero_level = self.rescale(np.zeros(traces[0].shape))

        n_batches = n_settings * n_traces // batch_size
        remainder = n_settings * n_traces % batch_size != 0

        for step_num in tqdm(range(n_steps)):
            for n in range(n_batches):
                traces[step_num + 1, n * batch_size: (n + 1) * batch_size] = self.next_state(
                    traces[step_num, n * batch_size: (n + 1) * batch_size],
                    curr_state_rescaled=True,
                    scale_back_result=False,
                    round_result=False,
                    n_samples=1,
                )
            if remainder:
                traces[step_num + 1, n_settings * n_traces:] = self.next_state(
                    traces[step_num, n_settings * n_traces:],
                    curr_state_rescaled=True,
                    scale_back_result=False,
                    round_result=False,
                    n_samples=1,
                )
            traces[step_num + 1] = np.maximum(zero_level, traces[step_num + 1])

        traces = np.reshape(traces, traces_final_shape)
        traces = np.squeeze(traces, axis=-2)

        if scale_back_result:
            traces = self.scale_back(traces)
            if round_result:
                traces = np.around(traces)

        # [n_steps, n_traces, n_settings, nb_features] -> [n_settings, n_traces, n_steps, nb_features]
        traces = np.transpose(traces, (2, 1, 0, 3))

        if add_timestamps:
            timespan = np.arange(0, (n_steps + 1) * self.timestep, self.timestep)
            timespan = np.tile(timespan, reps=(n_settings, n_traces, 1))
            timespan = timespan[..., np.newaxis]
            traces = np.concatenate([timespan, traces], axis=-1)

        return traces

