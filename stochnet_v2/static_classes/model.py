import numpy as np
import tensorflow as tf
import json
import logging
import os
import pickle
from collections import namedtuple
from functools import partial

from stochnet_v2.static_classes import nn_bodies
from stochnet_v2.static_classes.top_layers import MixtureOutputLayer
from stochnet_v2.static_classes.top_layers import MIXTURE_COMPONENTS_REGISTRY
from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.utils.errors import NotRestoredVariables
from stochnet_v2.utils.registry import ACTIVATIONS_REGISTRY
from stochnet_v2.utils.registry import CONSTRAINTS_REGISTRY
from stochnet_v2.utils.registry import REGULARIZERS_REGISTRY


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


def get_body_fn(config):

    body_fn_name = config.pop('body_fn', None)
    if body_fn_name is None:
        raise ValueError(f"`body_fn` parameter not specified in config")

    body_fn = getattr(nn_bodies, body_fn_name, None)
    if body_fn is None:
        raise ValueError(f"`body_fn` name {body_fn_name} not found in {nn_bodies.__file__}")

    return partial(body_fn, **config)


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
        self.restored = False

        self.graph = tf.compat.v1.Graph()

        with self.graph.as_default():

            self.session = tf.compat.v1.Session()

            if mode == 'normal':
                if body_config_path is None:
                    raise ValueError("Should provide `body_config_path` to build model")
                body_config = self._read_config(body_config_path, True)

                if mixture_config_path is None:
                    raise ValueError("Should provide `mixture_config_path` to build model")
                mixture_config = self._read_config(mixture_config_path, True)

                body_fn = get_body_fn(body_config)
                self._build_main_graph(body_fn, mixture_config)
                self._build_sampling_graph()
                self._save_graph_keys()
                if ckpt_path:
                    self.restore_from_checkpoint(ckpt_path)

            elif mode == 'inference':
                self._load_model_from_frozen_graph()

            elif mode == 'inference_ckpt':
                if ckpt_path is None:
                    raise ValueError("Should provide `ckpt_path` to build model")
                self._load_model_from_checkpoint(ckpt_path)

            else:
                raise ValueError(
                    "Unknown keyword for `mode` parameter. Use 'normal', 'inference' or 'inference_ckpt'"
                )

            LOGGER.info(f"Model created in {mode} mode.")

        self.scaler = self.load_scaler()

    def _read_config(self, file_path, save_to_model_dir=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, 'r') as f:
            config = json.load(f)

        if save_to_model_dir is True:
            file_name = os.path.basename(file_path)
            save_path = os.path.join(self.model_explorer.model_folder, file_name)
            with open(save_path, 'w') as f:
                json.dump(config, f, indent='\t')

        return config

    def _build_main_graph(self, body_fn, mixture_config_path):
        self.input_placeholder = tf.compat.v1.placeholder(
            tf.float32, (None, self.nb_past_timesteps, self.nb_features))
        self.rv_output_ph = tf.compat.v1.placeholder(tf.float32, (None, self.nb_features))
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
            output_node_names=[
                t.split(':')[0]
                for t in [self._sample_tensor_name, self._pred_tensor_name]
            ],
        )
        return frozen_graph_def

    def _save_graph_keys(self):
        graph_keys_dict = {
            'input_placeholder': self._input_placeholder_name,
            'pred_tensor': self._pred_tensor_name,
            'pred_placeholder':  self._pred_placeholder_name,
            'sample_shape_placeholder': self._sample_shape_placeholder_name,
            'sample_tensor':  self._sample_tensor_name
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

    def load_scaler(self):
        with open(self.dataset_explorer.scaler_fp, 'rb') as file:
            scaler = pickle.load(file)
        return scaler

    def rescale(self, values):
        return (values - self.scaler.mean_) / self.scaler.scale_

    def scale_back(self, values):
        return values * self.scaler.scale_ + self.scaler.mean_

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
            add_timesteps=False,
    ):
        n_settings, *state_shape = curr_state_values.shape
        traces = np.zeros((n_steps + 1, n_traces, n_settings, *state_shape))

        if not curr_state_rescaled:
            curr_state_values = self.rescale(curr_state_values)

        traces[0] = curr_state_values

        next_state_values = self.next_state(
                curr_state_values,
                curr_state_rescaled=True,
                scale_back_result=False,
                round_result=False,
                n_samples=n_traces,
            )
        traces[1] = next_state_values

        # for step in range(2, n_steps + 1):
        #     next_state_values = next_state_values.reshape((-1, *state_shape))
        #     next_state_values = self.next_state(
        #         next_state_values,
        #         curr_state_rescaled=True,
        #         scale_back_result=False,
        #         round_result=False,
        #         n_samples=1,
        #     )
        #     next_state_values = next_state_values.reshape((-1, batch_size, *state_shape))
        #     # next_state_values = np.maximum(0, next_state_values)
        #     traces[step] = next_state_values

        iterate_through_traces = n_traces <= n_settings
        print(f'iterate_through_traces: {iterate_through_traces}')

        if iterate_through_traces:
            for trace_idx in range(n_traces):
                state_values = next_state_values[trace_idx]
                for step in range(2, n_steps + 1):
                    state_values = self.next_state(
                        state_values,
                        curr_state_rescaled=False,
                        scale_back_result=False,
                        round_result=False,
                        n_samples=1,
                    )
                    # state_values = np.squeeze(state_values, 0)
                    # traces[step, trace_idx] = state_values
                    traces[step, trace_idx] = state_values[0]
        else:
            for setting_idx in range(n_settings):
                state_values = next_state_values[:, setting_idx]
                for step in range(2, n_steps + 1):
                    state_values = self.next_state(
                        state_values,
                        curr_state_rescaled=False,
                        scale_back_result=False,
                        round_result=False,
                        n_samples=1,
                    )
                    # state_values = np.squeeze(state_values, 0)
                    # traces[step, :, setting_idx] = state_values
                    traces[step, :, setting_idx] = state_values[0]

        # [n_steps, n_settings, batch_size, 1, nb_features] -> [n_steps, n_traces, n_settings, nb_features]
        traces = np.squeeze(traces, axis=-2)

        if scale_back_result:
            traces = self.scale_back(traces)
            if round_result:
                traces = np.around(traces)

        # [n_steps, n_traces, n_settings, nb_features] -> [n_settings, n_traces, n_steps, nb_features]
        traces = np.transpose(traces, (2, 1, 0, 3))

        if add_timesteps:
            timespan = np.arange(0, (n_steps + 1) * self.timestep, self.timestep)
            timespan = np.tile(timespan, reps=(batch_size, n_traces, 1))
            timespan = timespan[..., np.newaxis]
            traces = np.concatenate([timespan, traces], axis=-1)

        return traces
