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

import stochnet_v2.static_classes.nn_bodies as nn_bodies
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
    """
    Main class containing Mixture Density Network (MDN) neural network.
    After trained by Trainer, can be re-initialized for predicting trajectories.
    """

    def __init__(
            self,
            nb_past_timesteps,
            nb_features,
            nb_randomized_params,
            project_folder,
            timestep,
            dataset_id,
            model_id,
            body_config_path=None,
            mixture_config_path=None,
            ckpt_path=None,
            mode='normal',
    ):
        """
        Initialize model.
        Model can be initialized in three modes:
            * normal - build MDN for training
            * inference - load trained model (from frozen graph). In this mode it can not be trained.
            * inference_ckpt - load model from a training checkpoint. Can be trained further,
              as well as produce predictions. However, as the graph created in this mode is trainable,
              it is not optimised for predictions, i.e. has many redundancies.

        Parameters
        ----------
        nb_past_timesteps : number of time-steps model can observe in past to make a prediction.
            This number is reflected in the input shape: (bs, nb_past_timesteps, nb_features + nb_randomized_params)
        nb_features : number of CRN model features (i.e. species).
        nb_randomized_params : number of CRN model randomized params. MDN takes params values as additional inputs:
            for input of shape (bs, nb_past_timesteps, nb_features + nb_randomized_params) it samples
            outputs of shape   (bs, nb_past_timesteps, nb_features).
        project_folder : root folder for current project (CRN model).
        timestep : time-step between two consecutive states of CRN (which are basically input
            and ground-truth output of MDN). Used to find dataset-related files such as scaler, and
            save/find self model-related files.
        dataset_id : ID number of training dataset. Used to find dataset-related files such as scaler.
        model_id : ID number of the model (self) save/find related files.
        body_config_path : path to a .json configuration file, defining body-part of MDN
            (body_fn_name, block_name, n_blocks, hidden_size, use_batch_norm, activation, constraints and regularisers).
            This config will be copied to the model folder during initialization.
            If None, then it will try to find it in the model folder.
        mixture_config_path : path to a .json configuration file, defining mixture-part of MDN
            (number and types of components, their hidden_size, activation functions, constraints and regularisers).
            This config will be copied to the model folder during initialization.
            If None, then it will try to find it in the model folder.
        ckpt_path : path to a checkpoint file to initialize model parameters (in 'normal' mode),
            or re-create model graph and initialize parameters (in 'inference_ckpt' mode).
        mode : mode to build the model.
        """
        self.nb_past_timesteps = nb_past_timesteps
        self.nb_features = nb_features
        self.nb_randomized_params = nb_randomized_params
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
            body_config = self._read_config(self.model_explorer.body_config_fp)
        else:
            body_config = self._read_config(
                body_config_path,
                os.path.basename(self.model_explorer.body_config_fp)
            )
        if mixture_config_path is None:
            mixture_config = self._read_config(self.model_explorer.mixture_config_fp)
        else:
            mixture_config = self._read_config(
                mixture_config_path,
                os.path.basename(self.model_explorer.mixture_config_fp)
            )
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

    def _build_main_graph(self, body_fn, mixture_config):
        self.input_placeholder = tf.compat.v1.placeholder(
            tf.float32, (None, self.nb_past_timesteps, self.nb_features + self.nb_randomized_params), name="input"
        )
        self.rv_output_ph = tf.compat.v1.placeholder(
            tf.float32, (None, self.nb_features), name="random_variable_output"
        )
        body = body_fn(self.input_placeholder)
        self.top_layer_obj = _get_mixture(
            mixture_config,
            sample_space_dimension=self.nb_features
        )
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
        """Restore model parameters from a training checkpoint.

        Parameters
        ----------
        ckpt_path : path to .ckpt file. Tensorflow checkpoints go in three files typically,
            therefore last part of ckpt file-name ('.index', '.meta', '.XXXX-of-XXXX) should be omitted.
        """
        with self.graph.as_default():
            variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            saver = tf.compat.v1.train.Saver(var_list=variables)
            saver.restore(self.session, ckpt_path)
        self.restored = True

    def save(self):
        """Save model and all related information.
        Saves model frozen-graph and its important graph-keys:
            input_placeholder : name of input tensor
            pred_tensor : name of body-part output tensor
            pred_placeholder : name of mixture-part input tensor
            sample_shape_placeholder : name input tensor specifying sample_shape (i.e. number of samples)
            sample_tensor : name of mixture-part output tensor (samples of mixture distribution)
            description_graphkeys : tensors corresponding to the parameters of the mixture components.
                Used to produce distribution description.

        Returns
        -------

        """
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
        """
        Create description of the mixture distribution.
        For every input (current_state), the model predicts parameters specifying
        mixture distribution (nn_prediction), and then samples from this distribution.
        So one can use already computed nn_prediction values, or current_state values
        (and nn_prediction will be computed by the model naturally.)

        Parameters
        ----------
        nn_prediction_val : values of body-part outputs.
        current_state_val : input values (CRN model state).
        current_state_rescaled : boolean, whether or not inputs are rescaled with dataset scaler.
            If False, inputs will be first rescaled: as model is usually trained on rescaled data,
            inputs should be also rescaled.
        visualize : boolean, if True, will create figures (which can be automatically displayed in jupyter).

        Returns
        -------
        Description dictionary containing parameters of mixture distribution components.

        """
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
        """Apply scaler to data."""
        if isinstance(self.scaler, StandardScaler):
            try:
                data = (data - self.scaler.mean_) / self.scaler.scale_
            except ValueError:
                data = (data - self.scaler.mean_[:self.nb_features]) \
                       / self.scaler.scale_[:self.nb_features]
        elif isinstance(self.scaler, MinMaxScaler):
            try:
                data = (data * self.scaler.scale_) + self.scaler.min_
            except ValueError:
                data = (data * self.scaler.scale_[:self.nb_features]) \
                       + self.scaler.min_[:self.nb_features]
        return data

    def scale_back(self, data):
        """Apply scaler inverse transform to data."""
        if isinstance(self.scaler, StandardScaler):
            try:
                data = data * self.scaler.scale_ + self.scaler.mean_
            except ValueError:
                data = data * self.scaler.scale_[:self.nb_features] \
                       + self.scaler.mean_[:self.nb_features]
        elif isinstance(self.scaler, MinMaxScaler):
            try:
                data = (data - self.scaler.min_) / self.scaler.scale_
            except ValueError:
                data = (data - self.scaler.min_[:self.nb_features]) \
                       / self.scaler.scale_[:self.nb_features]
        return data

    def predict(self, curr_state_values):
        """
        Return prediction values for mixture components.
        This values then can be forwarded to `sample` method for sampling next state.
        Values should be rescaled first.

        Parameters
        ----------
        curr_state_values : input values (CRN model state). Should be rescaled first.

        Returns
        -------
        prediction_values : array of (concatenated) prediction values for mixture components.
        """

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
        """
        Sample from mixture distribution, defined by input prediction_values.

        Parameters
        ----------
        prediction_values : prediction values for mixture components, (returned by `predict` method).
        sample_shape : shape defining the number of samples: sample_shape=(n_samples,)

        Returns
        -------
        sample : array of samples from mixture distribution
        """
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
        """
        Sample next state given values of the current sate.
        The current sate should have shape [n_settings, nb_past_timesteps, nb_features + nb_randomized_params].

        Parameters
        ----------
        curr_state_values : input values (CRN model state).
        curr_state_rescaled : whether or not values are already rescaled. If False, will apply scaler first.
        scale_back_result : whether or not returned values should be scaled back to the original scale.
        round_result : whether or not round returned values (only if scaled back) to imitate
            discrete population dynamics.
        n_samples : number of samples to produce.

        Returns
        -------
        next_state : array of samples of shape [n_samples, n_settings, nb_past_timesteps, nb_features]
        """
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
            keep_params=False,
            batch_size=150,
    ):
        """
        Generate trajectories of the model.
        Trajectories are simulated by consecutive sampling next state for n_steps times.
        The current sate should have shape [n_settings, nb_past_timesteps, nb_features + nb_randomized_params].

        Parameters
        ----------
        curr_state_values : input values (CRN model state).
        n_steps : length of trajectories to simulate.
        n_traces : number of trajectories starting from every initial state
        curr_state_rescaled : whether or not values are already rescaled. If False, will apply scaler first.
        scale_back_result : whether or not returned values should be scaled back to the original scale.
        round_result : whether or not round returned values (only if scaled back) to imitate
            discrete population dynamics.
        add_timestamps : if True, time-step indexes will be added (as 0-th feature)
        keep_params : whether or not to keep randomized parameters in trajectories.
            If False, returned traces will have shape , [n_settings, n_traces, n_steps, nb_features]
            otherwise [n_settings, n_traces, n_steps, nb_features + nb_randomized_params]
        batch_size : batch size to use for simulations. For great number of simulations, it is more
            efficient to feed the neural network with reasonably-sized chunks of data.
        Returns
        -------

        traces : array of shape [n_settings, n_traces, n_steps, nb_features]
            or [n_settings, nb_past_timesteps, nb_features + nb_randomized_params],
            depending on the `keep_params` parameter.
        """
        n_settings, *state_shape = curr_state_values.shape

        traces_final_shape = (n_steps + 1, n_traces, n_settings, *state_shape)
        traces_tmp_shape = (n_steps + 1, n_settings * n_traces, *state_shape)

        traces = np.zeros(traces_tmp_shape, dtype=np.float32)

        if not curr_state_rescaled:
            curr_state_values = self.rescale(curr_state_values)

        curr_state_values = np.tile(curr_state_values, [n_traces, 1, 1])  # (n_settings * n_traces, *state_shape)
        traces[0] = curr_state_values

        zero_level = self.rescale(np.zeros(traces[0, ..., :self.nb_features].shape))

        n_batches = n_settings * n_traces // batch_size
        remainder = n_settings * n_traces % batch_size != 0

        for step_num in tqdm(range(n_steps)):
            for n in range(n_batches):
                next_state = self.next_state(
                    traces[step_num, n * batch_size: (n + 1) * batch_size],
                    curr_state_rescaled=True,
                    scale_back_result=False,
                    round_result=False,
                    n_samples=1)
                params = np.expand_dims(
                    traces[step_num, n * batch_size: (n + 1) * batch_size, ..., self.nb_features:], 0)

                traces[step_num + 1, n * batch_size: (n + 1) * batch_size] = \
                    np.concatenate([next_state, params], -1)

            if remainder:
                next_state = self.next_state(
                    traces[step_num, n_batches * batch_size:],
                    curr_state_rescaled=True,
                    scale_back_result=False,
                    round_result=False,
                    n_samples=1,
                )
                params = np.expand_dims(
                    traces[step_num, n_batches * batch_size:, ..., self.nb_features:], 0)

                traces[step_num + 1, n_batches * batch_size:] =  \
                    np.concatenate([next_state, params], -1)

            traces[step_num + 1, ..., :self.nb_features] = \
                np.maximum(traces[step_num + 1, ..., :self.nb_features], zero_level)

        traces = np.reshape(traces, traces_final_shape)
        traces = np.squeeze(traces, axis=-2)

        if scale_back_result:
            traces = self.scale_back(traces)
            if round_result:
                traces[..., :self.nb_features] = np.around(traces[..., :self.nb_features])

        # [n_steps, n_traces, n_settings, nb_features] -> [n_settings, n_traces, n_steps, nb_features]
        traces = np.transpose(traces, (2, 1, 0, 3))

        if not keep_params:
            traces = traces[..., :self.nb_features]

        if add_timestamps:
            timespan = np.arange(0, (n_steps+1) * self.timestep, self.timestep)[:n_steps+1]
            timespan = np.tile(timespan, reps=(n_settings, n_traces, 1))
            timespan = timespan[..., np.newaxis]
            traces = np.concatenate([timespan, traces], axis=-1)

        return traces

