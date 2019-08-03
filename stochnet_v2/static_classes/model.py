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
            self.input_ph = tf.compat.v1.placeholder(tf.float32, (None, self.nb_past_timesteps, self.nb_features))
            self.rv_output_ph = tf.compat.v1.placeholder(tf.float32, (None, self.nb_features))
            self.body = body_fn(self.input_ph)
            self.top_layer_obj = _get_mixture(mixture_config_path, sample_space_dimension=self.nb_features)
            self.nn_output = self.top_layer_obj.add_layer_on_top(self.body)
            self.loss = self.top_layer_obj.loss_function(self.rv_output_ph, self.nn_output)

        LOGGER.info(f'nn_output shape: {self.nn_output.shape}')
        LOGGER.info(f'loss shape: {self.loss.shape}')

        self.scaler = self.load_scaler()

        self.restored = False

        if ckpt_path:
            self.restore_from_checkpoint(ckpt_path)

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

    def predict(self, curr_state_values):

        if not self.restored:
            raise NotRestoredVariables()

        prediction_values = self.session.run(
            self.nn_output,
            feed_dict={
                self.input_ph: curr_state_values
            }
        )
        return prediction_values

    def sample(self, prediction_values, sample_shape=()):
        sample = self.top_layer_obj.sample_fast(
            prediction_values,
            session=self.session,
            sample_shape=sample_shape,
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

    # def get_dataset(
    #         self,
    #         tf_records_fp,
    #         batch_size,
    #         prefetch_size=None,
    #         shuffle=True,
    # ):
    #     return TFRecordsDataset(
    #         records_paths=tf_records_fp,
    #         batch_size=batch_size,
    #         prefetch_size=prefetch_size,
    #         shuffle=shuffle,
    #         nb_past_timesteps=self.nb_past_timesteps,
    #         nb_features=self.nb_features,
    #     )

    # def train(
    #         self,
    #         save_dir=None,
    #         learning_strategy=None,
    #         batch_size=_DEFAULT_BATCH_SIZE,
    #         prefetch_size=_DEFAULT_PREFETCH_SIZE,
    #         n_epochs=_DEFAULT_NUMBER_OF_EPOCHS,
    # ):
    #
    #     save_dir = save_dir or self.model_explorer.model_folder
    #
    #     if learning_strategy is None:
    #         learning_strategy = _DEFAULT_LEARNING_STRATEGY
    #
    #     trainable_graph, train_operations, train_input_x, train_input_y = \
    #         self._build_trainable_graph(learning_strategy)
    #
    #     with tf.Session(graph=trainable_graph) as session:
    #         session.run(tf.compat.v1.global_variables_initializer())
    #         best_loss_checkpoint_path = self._train(
    #             session=session,
    #             train_operations=train_operations,
    #             train_input_x=train_input_x,
    #             train_input_y=train_input_y,
    #             learning_strategy=learning_strategy,
    #             n_epochs=n_epochs,
    #             batch_size=batch_size,
    #             prefetch_size=prefetch_size,
    #             save_dir=save_dir,
    #         )
    #         self.variables_checkpoint_path = best_loss_checkpoint_path
    #
    #         return best_loss_checkpoint_path
    #
    # def _build_trainable_graph(
    #         self,
    #         learning_strategy,
    # ):
    #     trainable_graph = tf.compat.v1.Graph()
    #     copy_graph(self.graph, trainable_graph)
    #     model_input = get_transformed_tensor(self.input_ph, trainable_graph)
    #     rv_output = get_transformed_tensor(self.rv_output_ph, trainable_graph)
    #     loss = get_transformed_tensor(self.loss, trainable_graph)
    #
    #     with trainable_graph.as_default():
    #
    #         learning_rate = tf.compat.v1.placeholder_with_default(
    #             learning_strategy.initial_lr,
    #             shape=[],
    #             name='learning_rate',
    #         )
    #
    #         optimizer_type = learning_strategy.optimizer_type.lower()
    #         if optimizer_type == 'adam':
    #             optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    #         elif optimizer_type == 'sgd':
    #             optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, _DEFAULT_MOMENTUM)
    #         else:
    #             raise NotImplementedError(f'optimizer "{optimizer_type}" is not supported')
    #
    #         gradients = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
    #         gradients = optimizer.apply_gradients(gradients)
    #
    #         train_operations = TrainOperations(gradients, learning_rate, loss)
    #         train_input_x = model_input
    #         train_input_y = rv_output
    #
    #         return trainable_graph, train_operations, train_input_x, train_input_y
    #
    # def _train(
    #         self,
    #         session,
    #         train_operations,
    #         train_input_x,
    #         train_input_y,
    #         learning_strategy,
    #         n_epochs,
    #         batch_size,
    #         prefetch_size,
    #         save_dir,
    # ):
    #     tensorboard_log_dir = os.path.join(save_dir, 'tensorboard')
    #     checkpoints_save_dir = os.path.join(save_dir, 'checkpoints')
    #     maybe_create_dir(tensorboard_log_dir, erase_existing=True)
    #     maybe_create_dir(checkpoints_save_dir, erase_existing=True)
    #
    #     global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    #     trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    #     optimizer_vars = list(set(global_vars) - set(trainable_vars))
    #
    #     LOGGER.info(f'Total number of trainable vars {len(trainable_vars)}')
    #     LOGGER.info(f'Total number of optimizer vars {len(optimizer_vars)}')
    #
    #     session.run(tf.compat.v1.global_variables_initializer())
    #
    #     regular_checkpoints_saver = tf.compat.v1.train.Saver(
    #         var_list=trainable_vars,
    #         max_to_keep=_NUMBER_OF_REGULAR_CHECKPOINTS,
    #     )
    #     best_loss_checkpoints_saver = tf.compat.v1.train.Saver(
    #         var_list=trainable_vars,
    #         max_to_keep=_NUMBER_OF_BEST_LOSS_CHECKPOINTS,
    #     )
    #
    #     summary_writer = tf.compat.v1.summary.FileWriter(tensorboard_log_dir, session.graph)
    #
    #     train_loss_summary = tf.compat.v1.summary.scalar('train_loss', train_operations.loss)
    #     learning_rate_summary = tf.compat.v1.summary.scalar('train_learning_rate', train_operations.learning_rate)
    #
    #     test_mean_loss_ph = tf.compat.v1.placeholder(tf.float32, ())
    #     test_loss_summary = tf.compat.v1.summary.scalar('test_mean_loss', test_mean_loss_ph)
    #
    #     initial_learning_rate = session.run(train_operations.learning_rate)
    #     next_learning_rate = initial_learning_rate
    #
    #     train_step = 0
    #     decay_step = 0
    #     nans_step_counter = 0
    #
    #     best_loss = float('inf')
    #     best_loss_step = 0
    #
    #     def reset_optimizer():
    #         init_optimizer_vars = tf.compat.v1.variables_initializer(optimizer_vars)
    #         session.run(init_optimizer_vars)
    #
    #     def get_regular_checkpoint_path(step):
    #         return os.path.join(checkpoints_save_dir, f'regular_ckpt_{step}')
    #
    #     def get_best_checkpoint_path(step):
    #         return os.path.join(checkpoints_save_dir, f'best_ckpt_{step}')
    #
    #     train_dataset = self.get_dataset(
    #         self.dataset_explorer.train_records_rescaled_fp,
    #         batch_size=batch_size,
    #         prefetch_size=prefetch_size,
    #     )
    #     test_dataset = self.get_dataset(
    #         self.dataset_explorer.test_records_rescaled_fp,
    #         batch_size=batch_size,
    #     )
    #
    #     regular_checkpoints_saver.save(
    #         session,
    #         get_regular_checkpoint_path(train_step)
    #     )
    #
    #     LOGGER.info('Training...')
    #
    #     for epoch in range(n_epochs):
    #
    #         LOGGER.info(f'Epoch: {epoch + 1}')
    #
    #         for X_train, Y_train in train_dataset:
    #
    #             feed_dict = {
    #                 train_input_x: X_train,
    #                 train_input_y: Y_train,
    #                 train_operations.learning_rate: next_learning_rate,
    #             }
    #             fetches = {
    #                 'gradients': train_operations.gradients,
    #                 'loss': train_operations.loss,
    #                 'learning_rate_summary': learning_rate_summary,
    #                 'loss_summary': train_loss_summary,
    #             }
    #
    #             result = session.run(fetches=fetches, feed_dict=feed_dict)
    #
    #             next_learning_rate = initial_learning_rate.copy()
    #             next_learning_rate *= np.exp(-train_step * learning_strategy.lr_decay)
    #             next_learning_rate *= np.abs(
    #                 np.cos(learning_strategy.lr_cos_phase * decay_step / learning_strategy.lr_cos_steps)
    #             )
    #             next_learning_rate += _MINIMAL_LEARNING_RATE
    #
    #             summary_writer.add_summary(result['learning_rate_summary'], train_step)
    #             summary_writer.add_summary(result['loss_summary'], train_step)
    #
    #             has_nans = any([
    #                 np.isnan(result['loss']),
    #             ])
    #             if has_nans:
    #                 LOGGER.warning(f'Loss is None on step {train_step}, restore previous checkpoint...')
    #                 nans_step_counter += 1
    #                 checkpoint_step = train_step // _REGULAR_CHECKPOINTS_DELTA
    #                 checkpoint_step -= nans_step_counter
    #                 checkpoint_step = max(checkpoint_step, 0)
    #                 checkpoint_step *= _REGULAR_CHECKPOINTS_DELTA
    #
    #                 if checkpoint_step == 0:
    #                     LOGGER.info(f'checkpoint_step is 0, reinitialize all variables...')
    #                     session.run(tf.compat.v1.variables_initializer(global_vars))
    #                 else:
    #                     regular_checkpoints_saver.restore(
    #                         session,
    #                         get_regular_checkpoint_path(checkpoint_step)
    #                     )
    #                     reset_optimizer()
    #                 continue
    #
    #             if result['loss'] < best_loss:
    #                 best_loss_checkpoints_saver.save(session, get_best_checkpoint_path(train_step))
    #                 best_loss, best_loss_step = result['loss'], train_step
    #
    #             if train_step % _REGULAR_CHECKPOINTS_DELTA == 0:
    #                 nans_step_counter = 0
    #                 regular_checkpoints_saver.save(
    #                     session,
    #                     get_regular_checkpoint_path(train_step)
    #                 )
    #
    #             train_step += 1
    #             decay_step += 1
    #
    #             if train_step % learning_strategy.lr_cos_steps == 0:
    #                 LOGGER.info('Reinitialize optimizer...')
    #                 reset_optimizer()
    #                 decay_step = 0
    #
    #         test_losses = []
    #
    #         for X_test, Y_test in test_dataset:
    #
    #             feed_dict = {
    #                 train_input_x: X_test,
    #                 train_input_y: Y_test,
    #             }
    #             fetches = {
    #                 'loss': train_operations.loss,
    #             }
    #
    #             result = session.run(fetches=fetches, feed_dict=feed_dict)
    #             test_losses.append(result['loss'])
    #
    #         test_mean_loss = np.mean(test_losses)
    #         test_loss_summary_val = session.run(
    #             test_loss_summary,
    #             feed_dict={test_mean_loss_ph: test_mean_loss}
    #         )
    #         summary_writer.add_summary(test_loss_summary_val, epoch)
    #
    #         LOGGER.info(f'minimal loss value = {best_loss}')
    #
    #     # best_loss_checkpoints_saver.restore(
    #     #     session,
    #     #     get_best_checkpoint_path(best_loss_step)
    #     # )
    #     best_checkpoint_path = get_best_checkpoint_path(best_loss_step)
    #
    #     return best_checkpoint_path


