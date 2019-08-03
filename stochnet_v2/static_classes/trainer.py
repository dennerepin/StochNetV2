import numpy as np
import tensorflow as tf
import logging
import os
from collections import namedtuple
from time import time

from stochnet_v2.dataset.dataset import TFRecordsDataset
from stochnet_v2.utils.util import maybe_create_dir
from stochnet_v2.utils.util import copy_graph
from stochnet_v2.utils.util import get_transformed_tensor

LOGGER = logging.getLogger('static_classes.trainer')


TrainOperations = namedtuple(
    'TrainOperations',
    [
        'gradients',
        'learning_rate',
        'loss',
        'global_step',
        'increase_global_step',
    ],
)

ExpDecayLearningStrategy = namedtuple(
    'ExpDecayLearningStrategy',
    [
        'optimizer_type',
        'initial_lr',
        'lr_decay',
        'lr_cos_steps',
        'lr_cos_phase',
        'minimal_lr',
    ],
)


ToleranceDropLearningStrategy = namedtuple(
    'ToleranceDropLearningStrategy',
    [
        'optimizer_type',
        'initial_lr',
        'lr_decay',
        'epochs_tolerance',
        'minimal_lr',
    ],
)


_MINIMAL_LEARNING_RATE = 1 * 10 ** - 7
_NUMBER_OF_REGULAR_CHECKPOINTS = 10
_NUMBER_OF_BEST_LOSS_CHECKPOINTS = 5
_REGULAR_CHECKPOINTS_DELTA = 1000
_DEFAULT_NUMBER_OF_EPOCHS = 100
_DEFAULT_BATCH_SIZE = 1024
_DEFAULT_PREFETCH_SIZE = 10
_DEFAULT_MOMENTUM = 0.9

# _DEFAULT_LEARNING_STRATEGY = ExpDecayLearningStrategy(
#     optimizer_type='adam',
#     initial_lr=1e-4,
#     lr_decay=1e-4,
#     lr_cos_steps=None,
#     lr_cos_phase=np.pi / 2,
#     minimal_lr=1e-7,
# )

_DEFAULT_LEARNING_STRATEGY = ToleranceDropLearningStrategy(
    optimizer_type='adam',
    initial_lr=1e-4,
    lr_decay=0.6,
    epochs_tolerance=6,
    minimal_lr=1e-7,
)


class Trainer:

    def train(
            self,
            model,
            save_dir=None,
            learning_strategy=None,
            batch_size=_DEFAULT_BATCH_SIZE,
            n_epochs=_DEFAULT_NUMBER_OF_EPOCHS,
            ckpt_path=None,
    ):
        save_dir = save_dir or model.model_explorer.model_folder

        if learning_strategy is None:
            learning_strategy = _DEFAULT_LEARNING_STRATEGY

        trainable_graph, train_operations, train_input_x, train_input_y = \
            self._build_trainable_graph(model, learning_strategy)

        with tf.Session(graph=trainable_graph) as session:

            if ckpt_path is None:
                session.run(tf.compat.v1.global_variables_initializer())
            else:
                tf.compat.v1.train.Saver().restore(session, ckpt_path)

            clear_train_dir = ckpt_path is None
            tensorboard_log_dir = os.path.join(save_dir, 'tensorboard')
            checkpoints_save_dir = os.path.join(save_dir, 'checkpoints')
            maybe_create_dir(tensorboard_log_dir, erase_existing=clear_train_dir)
            maybe_create_dir(checkpoints_save_dir, erase_existing=clear_train_dir)

            best_loss_checkpoint_path = self._train(
                model=model,
                session=session,
                train_operations=train_operations,
                train_input_x=train_input_x,
                train_input_y=train_input_y,
                learning_strategy=learning_strategy,
                n_epochs=n_epochs,
                batch_size=batch_size,
                checkpoints_save_dir=checkpoints_save_dir,
                tensorboard_log_dir=tensorboard_log_dir,
            )
            return best_loss_checkpoint_path

    def _build_trainable_graph(
            self,
            model,
            learning_strategy,
    ):
        trainable_graph = tf.compat.v1.Graph()
        copy_graph(model.graph, trainable_graph)
        model_input = get_transformed_tensor(model.input_placeholder, trainable_graph)
        rv_output = get_transformed_tensor(model.rv_output_ph, trainable_graph)
        loss = get_transformed_tensor(model.loss, trainable_graph)

        with trainable_graph.as_default():

            learning_rate = tf.compat.v1.placeholder_with_default(
                learning_strategy.initial_lr,
                shape=[],
                name='learning_rate',
            )

            global_step = tf.Variable(0, trainable=False, name='global_step')
            increase_global_step = tf.assign_add(global_step, 1)

            optimizer_type = learning_strategy.optimizer_type.lower()
            if optimizer_type == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            elif optimizer_type == 'sgd':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, _DEFAULT_MOMENTUM)
            else:
                raise NotImplementedError(f'optimizer "{optimizer_type}" is not supported')

            gradients = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
            gradients = optimizer.apply_gradients(gradients)

            train_operations = TrainOperations(
                gradients,
                learning_rate,
                loss,
                global_step,
                increase_global_step,
            )
            train_input_x = model_input
            train_input_y = rv_output

            return trainable_graph, train_operations, train_input_x, train_input_y

    @staticmethod
    def _get_datasets(
            model,
            batch_size,
            kind='tfrecord'
    ):
        if kind == 'tfrecord':
            train_ds = TFRecordsDataset(
                records_paths=model.dataset_explorer.train_records_rescaled_fp,
                batch_size=batch_size,
                prefetch_size=_DEFAULT_PREFETCH_SIZE,
                shuffle=True,
                nb_past_timesteps=model.nb_past_timesteps,
                nb_features=model.nb_features,
            )
            test_ds = TFRecordsDataset(
                records_paths=model.dataset_explorer.test_records_rescaled_fp,
                batch_size=batch_size,
                prefetch_size=_DEFAULT_PREFETCH_SIZE,
                shuffle=True,
                nb_past_timesteps=model.nb_past_timesteps,
                nb_features=model.nb_features,
            )
            return train_ds, test_ds

        raise ValueError(
            f"Could not recognize the 'kind' key: {kind}. "
            f"Should be one of ['tfrecord', ]."
        )

    def _train(
            self,
            model,
            session,
            train_operations,
            train_input_x,
            train_input_y,
            learning_strategy,
            n_epochs,
            batch_size,
            checkpoints_save_dir,
            tensorboard_log_dir,
    ):
        global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        optimizer_vars = list(set(global_vars) - set(trainable_vars) - {train_operations.global_step})

        LOGGER.info(f'Total number of trainable vars {len(trainable_vars)}')
        LOGGER.info(f'Total number of optimizer vars {len(optimizer_vars)}')

        regular_checkpoints_saver = tf.compat.v1.train.Saver(
            # var_list=trainable_vars,
            var_list=global_vars,
            max_to_keep=_NUMBER_OF_REGULAR_CHECKPOINTS,
        )
        best_loss_checkpoints_saver = tf.compat.v1.train.Saver(
            var_list=trainable_vars,
            max_to_keep=_NUMBER_OF_BEST_LOSS_CHECKPOINTS,
        )

        summary_writer = tf.compat.v1.summary.FileWriter(tensorboard_log_dir, session.graph)

        train_loss_summary = tf.compat.v1.summary.scalar('train_loss', train_operations.loss)
        learning_rate_summary = tf.compat.v1.summary.scalar('train_learning_rate', train_operations.learning_rate)

        test_mean_loss_ph = tf.compat.v1.placeholder(tf.float32, ())
        test_loss_summary = tf.compat.v1.summary.scalar('test_mean_loss', test_mean_loss_ph)

        layer_params = {
            var.name.split(':')[0]: var
            for var in trainable_vars
            if any([sub in var.name for sub in {'kernel', 'bias'}])
        }

        histogram_summaries = []
        for name, var in layer_params.items():
            summary = tf.compat.v1.summary.histogram(name, var)
            histogram_summaries.append(summary)

        initial_learning_rate = session.run(train_operations.learning_rate)
        next_learning_rate = initial_learning_rate

        global_step = session.run(train_operations.global_step)
        decay_step = 0
        tolerance_step = 0
        nans_step_counter = 0

        best_loss = float('inf')
        tolerance_best_loss = float('inf')
        best_loss_step = 0

        def reset_optimizer():
            init_optimizer_vars = tf.compat.v1.variables_initializer(optimizer_vars)
            session.run(init_optimizer_vars)

        def get_regular_checkpoint_path(step):
            return os.path.join(checkpoints_save_dir, f'regular_ckpt_{step}')

        def get_best_checkpoint_path(step):
            return os.path.join(checkpoints_save_dir, f'best_ckpt_{step}')

        train_dataset, test_dataset = self._get_datasets(
            model,
            batch_size=batch_size,
            kind='tfrecord',
        )

        regular_checkpoints_saver.save(session, get_regular_checkpoint_path(global_step))

        LOGGER.info('Training...')

        for epoch in range(n_epochs):

            epoch_start_time = time()
            epoch_start_step = global_step

            LOGGER.info(f'Epoch: {epoch + 1}')

            for X_train, Y_train in train_dataset:

                feed_dict = {
                    train_input_x: X_train,
                    train_input_y: Y_train,
                    train_operations.learning_rate: next_learning_rate,
                }
                fetches = {
                    'gradients': train_operations.gradients,
                    'loss': train_operations.loss,
                    'learning_rate_summary': learning_rate_summary,
                    'loss_summary': train_loss_summary,
                    'global_step': train_operations.global_step,
                }

                result = session.run(fetches=fetches, feed_dict=feed_dict)

                global_step = result['global_step']

                # drop lr for next step if train with exponential decay
                if learning_strategy.__class__.__name__ == 'ExpDecayLearningStrategy':
                    next_learning_rate = initial_learning_rate.copy()
                    next_learning_rate *= np.exp(-global_step * learning_strategy.lr_decay)
                    if learning_strategy.lr_cos_steps is not None:
                        next_learning_rate *= np.abs(
                            np.cos(learning_strategy.lr_cos_phase * decay_step / learning_strategy.lr_cos_steps)
                        )
                    next_learning_rate += learning_strategy.minimal_lr

                summary_writer.add_summary(result['learning_rate_summary'], global_step)
                summary_writer.add_summary(result['loss_summary'], global_step)

                # HANDLE NAN VALUES
                has_nans = any([
                    np.isnan(result['loss']),
                ])
                if has_nans:
                    LOGGER.warning(f'Loss is None on step {global_step}, restore previous checkpoint...')
                    nans_step_counter += 1
                    checkpoint_step = global_step // _REGULAR_CHECKPOINTS_DELTA
                    checkpoint_step -= nans_step_counter
                    checkpoint_step = max(checkpoint_step, 0)
                    checkpoint_step *= _REGULAR_CHECKPOINTS_DELTA

                    if checkpoint_step == 0:
                        LOGGER.info(f'checkpoint_step is 0, reinitialize all variables...')
                        session.run(tf.compat.v1.variables_initializer(global_vars))
                    else:
                        try:
                            regular_checkpoints_saver.restore(
                                session,
                                get_regular_checkpoint_path(checkpoint_step)
                            )
                        except ValueError:
                            LOGGER.info(
                                f'Checkpoint for step {checkpoint_step} not found, '
                                f'will restore from the oldest existing checkpoint...'
                            )
                            ckpt_state = tf.compat.v1.train.get_checkpoint_state(checkpoints_save_dir)
                            oldest_ckpt_path = ckpt_state.all_model_checkpoint_paths[0]
                            regular_checkpoints_saver.restore(
                                session,
                                oldest_ckpt_path
                            )
                        reset_optimizer()
                    continue

                # save best checkpoint
                if result['loss'] < best_loss:
                    best_loss_checkpoints_saver.save(session, get_best_checkpoint_path(global_step))
                    best_loss, best_loss_step = result['loss'], global_step

                # save regular checkpoint
                if global_step % _REGULAR_CHECKPOINTS_DELTA == 0:
                    nans_step_counter = 0
                    regular_checkpoints_saver.save(session, get_regular_checkpoint_path(global_step))

                # reset optimizer parameters for next cosine phase
                if learning_strategy.__class__.__name__ == 'ExpDecayLearningStrategy':
                    if learning_strategy.lr_cos_steps is not None:
                        if global_step % learning_strategy.lr_cos_steps == 0 and global_step > 0:
                            LOGGER.info('Reinitialize optimizer...')
                            reset_optimizer()
                            decay_step = 0

                # increase global step
                session.run(train_operations.increase_global_step)
                decay_step += 1

            # drop lr for next epoch if train with tolerance
            if learning_strategy.__class__.__name__ == 'ToleranceDropLearningStrategy':
                if best_loss < tolerance_best_loss:
                    tolerance_best_loss = best_loss
                    tolerance_step = 0
                else:
                    tolerance_step += 1

                if tolerance_step >= learning_strategy.epochs_tolerance:
                    next_learning_rate = np.maximum(
                        learning_strategy.minimal_lr,
                        next_learning_rate * learning_strategy.lr_decay
                    )

            # HISTOGRAMS
            for histogram_summary in histogram_summaries:
                histogram_summary_val = session.run(histogram_summary)
                summary_writer.add_summary(histogram_summary_val, epoch)

            epoch_time = time() - epoch_start_time
            epoch_steps = global_step - epoch_start_step
            avg_step_time = epoch_time / epoch_steps

            # TEST
            test_losses = []

            for X_test, Y_test in test_dataset:

                feed_dict = {
                    train_input_x: X_test,
                    train_input_y: Y_test,
                }
                fetches = {
                    'loss': train_operations.loss,
                }

                result = session.run(fetches=fetches, feed_dict=feed_dict)
                test_losses.append(result['loss'])

            test_mean_loss = np.mean(test_losses)
            test_loss_summary_val = session.run(
                test_loss_summary,
                feed_dict={test_mean_loss_ph: test_mean_loss}
            )
            summary_writer.add_summary(test_loss_summary_val, epoch)

            LOGGER.info(f' = Minimal loss value = {best_loss},\n'
                        f' - {epoch_steps} steps took {epoch_time:.0f} seconds, avg_step_time={avg_step_time:.3f}\n')

        best_checkpoint_path = get_best_checkpoint_path(best_loss_step)

        return best_checkpoint_path
