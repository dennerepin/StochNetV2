import numpy as np
import tensorflow as tf
import json
import logging
import pickle
from collections import namedtuple

from stochnet_v2.static_classes.top_layers import MixtureOutputLayer
from stochnet_v2.static_classes.top_layers import MIXTURE_COMPONENTS_REGISTRY
from stochnet_v2.utils.file_organisation import ProjectFileExplorer
from stochnet_v2.utils.errors import NotRestoredVariables
from stochnet_v2.utils.registry import ACTIVATIONS_REGISTRY
from stochnet_v2.utils.registry import CONSTRAINTS_REGISTRY
from stochnet_v2.utils.registry import REGULARIZERS_REGISTRY


LOGGER = logging.getLogger('static_classes.model')


ComponentDescription = namedtuple('ComponentDescription', ['name', 'parameters'])


def _get_mixture(config_path, sample_space_dimension):

    with open(config_path, 'r') as f:
        top_layer_conf = json.load(f)

    categorical = None
    components = []
    descriptions = [ComponentDescription(name, params) for (name, params) in top_layer_conf]

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
            timestep,
            dataset_id,
            body_fn,
            mixture_config_path,
            project_folder,
            model_id=None,
            ckpt_path=None,
    ):
        self.nb_past_timesteps = nb_past_timesteps
        self.nb_features = nb_features
        self.timestep = timestep

        self.project_explorer = ProjectFileExplorer(project_folder)
        self.dataset_explorer = self.project_explorer.get_dataset_file_explorer(self.timestep, dataset_id)
        self.model_explorer = self.project_explorer.get_model_file_explorer(self.timestep, model_id or dataset_id)
        self.variables_checkpoint_path = None

        self.graph = tf.compat.v1.Graph()

        with self.graph.as_default():
            self.session = tf.Session()
            self.input_placeholder = tf.compat.v1.placeholder(
                tf.float32, (None, self.nb_past_timesteps, self.nb_features))
            self.rv_output_ph = tf.compat.v1.placeholder(tf.float32, (None, self.nb_features))
            self.body = body_fn(self.input_placeholder)
            self.top_layer_obj = _get_mixture(mixture_config_path, sample_space_dimension=self.nb_features)
            self.pred_tensor = self.top_layer_obj.add_layer_on_top(self.body)
            self.loss = self.top_layer_obj.loss_function(self.rv_output_ph, self.pred_tensor)

        LOGGER.info(f'nn_output shape: {self.pred_tensor.shape}')
        LOGGER.info(f'loss shape: {self.loss.shape}')

        self.scaler = self.load_scaler()

        self.restored = False

        if ckpt_path:
            self.restore_from_checkpoint(ckpt_path)

        self._pred_placeholder = None
        self._pred_placeholder_name = None
        self._sample_shape_placeholder = None
        self._sample_shape_placeholder_name = None
        self._sample_tensor = None
        self._sample_tensor_name = None

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

    def restore_from_checkpoint(self, ckpt_path):
        with self.graph.as_default():
            variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            saver = tf.compat.v1.train.Saver(var_list=variables)
            saver.restore(self.session, ckpt_path)
        self.restored = True

    # def predict(self, curr_state_values):
    #
    #     if not self.restored:
    #         raise NotRestoredVariables()
    #
    #     prediction_values = self.session.run(
    #         self._pred_tensor,
    #         feed_dict={
    #             self._input_placeholder: curr_state_values
    #         }
    #     )
    #     return prediction_values

    # def sample(self, prediction_values, sample_shape=()):
    #     sample = self.top_layer_obj.sample_fast(
    #         prediction_values,
    #         session=self.session,
    #         sample_shape=sample_shape,
    #     )
    #     sample = np.expand_dims(sample, -2)
    #     return sample

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

    def _build_sampling_graph(self):
        if self.sample_tensor is not None:
            return
        self.top_layer_obj.build_sampling_graph(graph=self.graph)
        self.pred_placeholder = self.top_layer_obj.pred_placeholder
        self.sample_shape_placeholder = self.top_layer_obj.sample_shape_placeholder
        self.sample_tensor = self.top_layer_obj.sample_tensor

    # def sample(self, prediction_values, sample_shape=()):
    #     if self.sample_tensor is None:
    #         self._build_sampling_graph()
    #
    #     sample = self.session.run(
    #         self.sample_tensor,
    #         feed_dict={
    #             self.pred_placeholder: prediction_values,
    #             self.sample_shape_placeholder: sample_shape,
    #         }
    #     )
    #     sample = np.expand_dims(sample, -2)
    #     return sample

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
        # curr_state_values ~ [batch_size, 1, nb_features]
        if not curr_state_rescaled:
            curr_state_values = self.rescale(curr_state_values)

        nn_prediction_values = self.predict(curr_state_values)
        next_state = self.sample(nn_prediction_values, sample_shape=(n_samples,))

        if scale_back_result:
            next_state = self.scale_back(next_state)
            if round_result:
                next_state = np.around(next_state)

        # next_state ~ [n_samples, batch_size, 1, nb_features]
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
        batch_size, *state_shape = curr_state_values.shape
        traces = np.zeros((n_steps + 1, n_traces, batch_size, *state_shape))

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

        for step in range(2, n_steps + 1):
            next_state_values = next_state_values.reshape((-1, *state_shape))
            next_state_values = self.next_state(
                next_state_values,
                curr_state_rescaled=True,
                scale_back_result=False,
                round_result=False,
                n_samples=1,
            )
            next_state_values = next_state_values.reshape((-1, batch_size, *state_shape))
            # next_state_values = np.maximum(0, next_state_values)
            traces[step] = next_state_values

        # [n_steps, n_traces, batch_size, 1, nb_features] -> [n_steps, n_traces, batch_size, nb_features]
        traces = np.squeeze(traces, axis=-2)

        if scale_back_result:
            traces = self.scale_back(traces)
            if round_result:
                traces = np.around(traces)

        # [n_steps, n_traces, batch_size, nb_features] -> [batch_size, n_traces, n_steps, nb_features]
        traces = np.transpose(traces, (2, 1, 0, 3))

        if add_timesteps:
            timespan = np.arange(0, (n_steps + 1) * self.timestep, self.timestep)
            timespan = np.tile(timespan, reps=(batch_size, n_traces, 1))
            timespan = timespan[..., np.newaxis]
            traces = np.concatenate([timespan, traces], axis=-1)

        return traces
