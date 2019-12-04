import numpy as np
import tensorflow as tf
import h5py
import logging
import os
from collections import namedtuple
from time import time

from stochnet_v2.dataset.dataset import TFRecordsDataset
from stochnet_v2.dataset.dataset import HDF5Dataset
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
        'train_variables',
        'optimizer_variables',
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
#     lr_cos_steps=0,
#     lr_cos_phase=np.pi / 2,
#     minimal_lr=1e-7,
# )


_DEFAULT_LEARNING_STRATEGY = ToleranceDropLearningStrategy(
    optimizer_type='adam',
    initial_lr=1e-4,
    lr_decay=0.3,
    epochs_tolerance=7,
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
            dataset_kind='tfrecord',
            add_noise=False,
            stddev=0.01,
    ):
        save_dir = save_dir or model.model_explorer.model_folder

        if learning_strategy is None:
            learning_strategy = _DEFAULT_LEARNING_STRATEGY

        trainable_graph, train_operations, train_input_x, train_input_y = \
            self._build_trainable_graph(model, learning_strategy)

        with tf.compat.v1.Session(graph=trainable_graph) as session:

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
                dataset_kind=dataset_kind,
                add_noise=add_noise,
                stddev=stddev,
            )

            model.restore_from_checkpoint(best_loss_checkpoint_path)
            model.save()

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
        model_loss = get_transformed_tensor(model.loss, trainable_graph)

        with trainable_graph.as_default():

            regularization_loss = tf.losses.get_regularization_loss()
            LOGGER.debug(f"REGULARIZATION_LOSSES:{regularization_loss}")

            loss = model_loss + regularization_loss

            learning_rate = tf.compat.v1.placeholder_with_default(
                learning_strategy.initial_lr,
                shape=[],
                name='learning_rate',
            )

            global_step = tf.train.get_or_create_global_step()

            optimizer_type = learning_strategy.optimizer_type.lower()
            if optimizer_type == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            elif optimizer_type == 'sgd':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, _DEFAULT_MOMENTUM)
            else:
                raise NotImplementedError(f'optimizer "{optimizer_type}" is not supported')

            trainable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

            gradients = optimizer.compute_gradients(loss, var_list=trainable_variables)
            gradients = optimizer.apply_gradients(
                gradients,
                global_step,
            )

            train_operations = TrainOperations(
                gradients=gradients,
                learning_rate=learning_rate,
                loss=loss,
                global_step=global_step,
                train_variables=trainable_variables,
                optimizer_variables=optimizer.variables(),
            )

            train_input_x = model_input
            train_input_y = rv_output

            return trainable_graph, train_operations, train_input_x, train_input_y

    @staticmethod
    def _get_datasets(
            model,
            batch_size,
            kind='hdf5',
            add_noise=False,
            stddev=0.05,
    ):

        def _add_noise_tf(x, y):
            shape = x.shape.as_list()
            x = x + tf.random.normal(shape=shape, mean=0.0, stddev=stddev)
            return x, y

        def _add_noise_np(x, y):
            shape = x.shape
            x = x + np.random.normal(loc=0.0, scale=stddev, size=shape)
            return x, y

        if kind == 'tfrecord':

            train_ds = TFRecordsDataset(
                records_paths=model.dataset_explorer.train_records_rescaled_fp,
                batch_size=batch_size,
                prefetch_size=_DEFAULT_PREFETCH_SIZE,
                shuffle=True,
                nb_past_timesteps=model.nb_past_timesteps,
                nb_features=model.nb_features,
                preprocess_fn=_add_noise_tf if add_noise else None,
            )
            test_ds = TFRecordsDataset(
                records_paths=model.dataset_explorer.test_records_rescaled_fp,
                batch_size=batch_size,
                prefetch_size=_DEFAULT_PREFETCH_SIZE,
                shuffle=False,
                nb_past_timesteps=model.nb_past_timesteps,
                nb_features=model.nb_features,
                preprocess_fn=None,
            )
            return train_ds, test_ds

        if kind == 'hdf5':
            train_ds = HDF5Dataset(
                model.dataset_explorer.train_rescaled_fp,
                batch_size=batch_size,
                shuffle=True,
                preprocess_fn=_add_noise_np if add_noise else None,
            )
            test_ds = HDF5Dataset(
                model.dataset_explorer.test_rescaled_fp,
                batch_size=batch_size,
                shuffle=False,
                preprocess_fn=None,
            )
            return train_ds, test_ds

        raise ValueError(
            f"Could not recognize the 'kind' key: {kind}. "
            f"Should be one of ['tfrecord', 'hdf5']."
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
            dataset_kind,
            add_noise,
            stddev,
    ):
        global_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        # trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        # optimizer_vars = list(set(global_vars) - set(trainable_vars) - {train_operations.global_step})
        trainable_vars = train_operations.train_variables
        optimizer_vars = train_operations.optimizer_variables

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

        # general summaries:
        summary_writer = tf.compat.v1.summary.FileWriter(tensorboard_log_dir, session.graph)
        learning_rate_summary = tf.compat.v1.summary.scalar('train_learning_rate', train_operations.learning_rate)

        # train_loss_summary = tf.compat.v1.summary.scalar('train_loss', train_operations.loss)
        train_loss_summary = tf.compat.v1.summary.scalar(
            'train_loss', tf.reduce_mean(train_operations.loss))  # TODO: FOR VECTOR LOSS:

        test_mean_loss_ph = tf.compat.v1.placeholder(tf.float32, ())
        test_loss_summary = tf.compat.v1.summary.scalar('test_mean_loss', test_mean_loss_ph)

        # weights and biases summaries
        layer_params = {
            var.name.split(':')[0]: var
            for var in trainable_vars
            if any([sub in var.name for sub in {'kernel', 'bias'}])
        }
        histogram_summaries = []
        for name, var in layer_params.items():
            summary = tf.compat.v1.summary.histogram(name, var)
            histogram_summaries.append(summary)

        # TODO: TMP, categorical logits and probs summaries
        cat_logits = session.graph.get_tensor_by_name('MixtureOutputLayer/CategoricalOutputLayer/logits/BiasAdd:0')
        cat_probs = session.graph.get_tensor_by_name(
            'MixtureOutputLayer_1/random_variable/CategoricalOutputLayer/random_variable/Categorical/probs:0'
        )
        n_classes = cat_logits.shape.as_list()[-1]
        cat_logits_summaries = []
        cat_probs_summaries = []
        for n in range(n_classes):
            cat_logits_summaries.append(tf.compat.v1.summary.histogram('cat_logits', cat_logits[..., n]))
            cat_probs_summaries.append(tf.compat.v1.summary.histogram('cat_probs', cat_probs[..., n]))
        # TODO: end of TMP

        initial_learning_rate = session.run(train_operations.learning_rate)
        learning_rate = initial_learning_rate

        decay_step = 0
        tolerance_step = 0
        nans_step_counter = 0

        best_loss = float('inf')
        tolerance_best_loss = float('inf')
        best_loss_step = 0

        train_dataset, test_dataset = self._get_datasets(
            model,
            batch_size=batch_size,
            kind=dataset_kind,
            add_noise=add_noise,
            stddev=stddev,
        )

        def _reset_optimizer():
            init_optimizer_vars = tf.compat.v1.variables_initializer(optimizer_vars)
            session.run(init_optimizer_vars)

        def _get_regular_checkpoint_path(step):
            return os.path.join(checkpoints_save_dir, f'regular_ckpt_{step}')

        def _get_best_checkpoint_path(step):
            return os.path.join(checkpoints_save_dir, f'best_ckpt_{step}')

        def _run_train_step():

            feed_dict = {
                train_input_x: X_train,
                train_input_y: Y_train,
                train_operations.learning_rate: learning_rate,
            }
            fetches = {
                'gradients': train_operations.gradients,
                'loss': train_operations.loss,
                'learning_rate_summary': learning_rate_summary,
                'loss_summary': train_loss_summary,
                'global_step': train_operations.global_step,
            }

            res = session.run(fetches=fetches, feed_dict=feed_dict)
            res['loss'] = np.mean(res['loss'])  # TODO: FOR VECTOR LOSS
            return res

        def _run_test():
            test_losses = []

            for X_test, Y_test in test_dataset:
                feed_dict = {
                    train_input_x: X_test,
                    train_input_y: Y_test,
                }
                fetches = {
                    'loss': train_operations.loss,
                }

                res = session.run(fetches=fetches, feed_dict=feed_dict)
                res['loss'] = np.mean(res['loss'])  # TODO: FOR VECTOR LOSS
                test_losses.append(res['loss'])

            test_mean_loss = np.mean(test_losses)
            test_loss_summary_val = session.run(
                test_loss_summary,
                feed_dict={test_mean_loss_ph: test_mean_loss}
            )
            summary_writer.add_summary(test_loss_summary_val, epoch)

        def _run_histograms():
            for histogram_summary in histogram_summaries:
                histogram_summary_val = session.run(histogram_summary)
                summary_writer.add_summary(histogram_summary_val, epoch)

        def _run_cat_histograms():
            cat_logits_summaries_values, cat_probs_summaries_values = session.run(
                [cat_logits_summaries, cat_probs_summaries],
                {train_input_x: X_train}
            )
            for summary_val in cat_logits_summaries_values:
                summary_writer.add_summary(summary_val, global_step)
            for summary_val in cat_probs_summaries_values:
                summary_writer.add_summary(summary_val, global_step)

        def _maybe_drop_lr_tolerance(lr, tol_step, tol_best_loss):
            if learning_strategy.__class__.__name__ == 'ToleranceDropLearningStrategy':
                if best_loss < tol_best_loss:
                    tol_best_loss = best_loss
                    tol_step = 0
                else:
                    tol_step += 1
                if tol_step >= learning_strategy.epochs_tolerance:
                    lr = np.maximum(
                        learning_strategy.minimal_lr,
                        lr * learning_strategy.lr_decay
                    )
                    tol_step = 0
                    LOGGER.debug(f"drop lr: {lr}")
            return lr, tol_step, tol_best_loss

        def _maybe_drop_lr_decay(lr):
            if learning_strategy.__class__.__name__ == 'ExpDecayLearningStrategy':
                lr = initial_learning_rate.copy()
                lr *= np.exp(-global_step * learning_strategy.lr_decay)
                if learning_strategy.lr_cos_steps:
                    lr *= np.abs(
                        np.cos(learning_strategy.lr_cos_phase * decay_step / learning_strategy.lr_cos_steps)
                    )
                lr += learning_strategy.minimal_lr
            return lr

        def _handle_nans(counter):
            LOGGER.warning(f'Loss is None on step {global_step}, restore previous checkpoint...')
            counter += 1
            checkpoint_step = global_step // _REGULAR_CHECKPOINTS_DELTA
            checkpoint_step -= counter
            checkpoint_step = max(checkpoint_step, 0)
            checkpoint_step *= _REGULAR_CHECKPOINTS_DELTA

            if checkpoint_step == 0:
                LOGGER.info(f'checkpoint_step is 0, reinitialize all variables...')
                session.run(tf.compat.v1.variables_initializer(global_vars))
            else:
                try:
                    regular_checkpoints_saver.restore(
                        session,
                        _get_regular_checkpoint_path(checkpoint_step)
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
                _reset_optimizer()
            return counter

        LOGGER.info('Training...')

        for epoch in range(n_epochs):

            epoch_start_time = time()
            epoch_steps = 0

            LOGGER.info(f'\nEpoch: {epoch + 1}')

            for X_train, Y_train in train_dataset:

                result = _run_train_step()
                global_step = result['global_step']

                # HANDLE NAN VALUES
                if np.isnan(result['loss']):
                    nans_step_counter = _handle_nans(nans_step_counter)
                    continue

                # drop lr for next step if train with exponential decay
                learning_rate = _maybe_drop_lr_decay(learning_rate)

                summary_writer.add_summary(result['learning_rate_summary'], global_step)
                summary_writer.add_summary(result['loss_summary'], global_step)

                # TODO: TMP, categorical logits and probs summaries:
                if global_step % 300 == 0:
                    _run_cat_histograms()
                # TODO: end of TMP

                # save best checkpoint
                if result['loss'] < best_loss:
                    best_loss_checkpoints_saver.save(session, _get_best_checkpoint_path(global_step))
                    best_loss, best_loss_step = result['loss'], global_step

                # save regular checkpoint
                if global_step % _REGULAR_CHECKPOINTS_DELTA == 0:
                    nans_step_counter = 0
                    regular_checkpoints_saver.save(session, _get_regular_checkpoint_path(global_step))

                # reset optimizer parameters for next cosine phase
                if learning_strategy.__class__.__name__ == 'ExpDecayLearningStrategy':
                    if learning_strategy.lr_cos_steps:
                        if global_step % learning_strategy.lr_cos_steps == 0 and global_step > 0:
                            LOGGER.info('Reinitialize optimizer...')
                            _reset_optimizer()
                            decay_step = 0

                decay_step += 1
                epoch_steps += 1

            # drop lr for next epoch if train with tolerance
            learning_rate, tolerance_step, tolerance_best_loss = _maybe_drop_lr_tolerance(
                learning_rate, tolerance_step, tolerance_best_loss)

            # HISTOGRAMS
            _run_histograms()

            epoch_time = time() - epoch_start_time
            avg_step_time = epoch_time / epoch_steps

            # TEST
            test_start_time = time()
            _run_test()
            test_time = time() - test_start_time

            LOGGER.info(
                f' = Minimal loss value = {best_loss},\n'
                f' - {epoch_steps} steps took {epoch_time:.1f} seconds, avg_step_time={avg_step_time:.3f}\n'
                f' - test time: {test_time:.1f} seconds'
            )

        best_checkpoint_path = _get_best_checkpoint_path(best_loss_step)

        return best_checkpoint_path
