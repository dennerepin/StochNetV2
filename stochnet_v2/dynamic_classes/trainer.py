import numpy as np
import tensorflow as tf
import logging
import os
from collections import namedtuple
from time import time

from stochnet_v2.dataset.dataset import TFRecordsDataset
from stochnet_v2.dataset.dataset import HDF5Dataset
from stochnet_v2.utils.util import maybe_create_dir
from stochnet_v2.utils.util import copy_graph
from stochnet_v2.utils.util import get_transformed_tensor

LOGGER = logging.getLogger('dynamic_classes.trainer')

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
_DEFAULT_NUMBER_OF_EPOCHS_MAIN = 100
_DEFAULT_N_EPOCHS_INTERVAL = 5
_DEFAULT_NUMBER_OF_EPOCHS_ARCH = 5
_DEFAULT_BATCH_SIZE = 1024
_DEFAULT_PREFETCH_SIZE = 10
_DEFAULT_MOMENTUM = 0.9

_REG_LOSS_WEIGHT = 1.0

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
            learning_strategy_main=None,
            learning_strategy_arch=None,
            batch_size=None,
            n_epochs_main=None,
            n_epochs_arch=None,
            n_epochs_interval=None,
            ckpt_path=None,
            dataset_kind='tfrecord',
            add_noise=False,
            stddev=0.01,
    ):
        save_dir = save_dir or model.model_explorer.model_folder

        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

        if n_epochs_main is None:
            n_epochs_main = _DEFAULT_NUMBER_OF_EPOCHS_MAIN

        if n_epochs_arch is None:
            n_epochs_arch = _DEFAULT_NUMBER_OF_EPOCHS_ARCH

        if n_epochs_interval is None:
            n_epochs_interval = _DEFAULT_N_EPOCHS_INTERVAL

        if learning_strategy_main is None:
            learning_strategy_main = _DEFAULT_LEARNING_STRATEGY

        if learning_strategy_arch is None:
            learning_strategy_arch = _DEFAULT_LEARNING_STRATEGY

        trainable_graph, train_operations_main, train_operations_arch, train_input_x, train_input_y = \
            self._build_trainable_graph(model, learning_strategy_main, learning_strategy_arch)

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
                train_operations_main=train_operations_main,
                train_operations_arch=train_operations_arch,
                train_input_x=train_input_x,
                train_input_y=train_input_y,
                learning_strategy_main=learning_strategy_main,
                learning_strategy_arch=learning_strategy_arch,
                n_epochs_main=n_epochs_main,
                n_epochs_arch=n_epochs_arch,
                n_epochs_interval=n_epochs_interval,
                batch_size=batch_size,
                checkpoints_save_dir=checkpoints_save_dir,
                tensorboard_log_dir=tensorboard_log_dir,
                dataset_kind=dataset_kind,
                add_noise=add_noise,
                stddev=stddev,
            )

            model.restore_from_checkpoint(best_loss_checkpoint_path)
            model.save_genotypes()
            model.recreate_from_genome(best_loss_checkpoint_path)
            model.save()

        return best_loss_checkpoint_path

    def _build_trainable_graph(
            self,
            model,
            learning_strategy_main,
            learning_strategy_arch,
    ):
        trainable_graph = tf.compat.v1.Graph()
        copy_graph(model.graph, trainable_graph)
        model_input = get_transformed_tensor(model.input_placeholder, trainable_graph)
        rv_output = get_transformed_tensor(model.rv_output_ph, trainable_graph)
        model_loss = get_transformed_tensor(model.loss, trainable_graph)

        # trainable_graph = model.graph
        # model_input = model.input_placeholder
        # rv_output = model.rv_output_ph
        # model_loss = model.loss

        with trainable_graph.as_default():

            trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            arch_vars = [v for v in trainable_vars if 'architecture_variables' in v.name]
            main_vars = list(set(trainable_vars) - set(arch_vars))

            # print(f"\n\n == Main variables:")
            # for var in main_vars:
            #     print(var)
            #
            # print(f"\n\n == Arch variables:")
            # for var in arch_vars:
            #     print(var)

            # MAIN:
            global_step_main = tf.Variable(0, trainable=False, name='global_step_main')

            reg_losses_main = tf.compat.v1.get_collection('regularization_losses')
            if reg_losses_main:
                reg_loss_main = tf.compat.v1.add_n(reg_losses_main)
                loss_main = model_loss + _REG_LOSS_WEIGHT * reg_loss_main
            else:
                loss_main = model_loss

            learning_rate_main = tf.compat.v1.placeholder_with_default(
                learning_strategy_main.initial_lr,
                shape=[],
                name='learning_rate_main',
            )

            optimizer_type_main = learning_strategy_main.optimizer_type.lower()
            if optimizer_type_main == 'adam':
                optimizer_main = tf.compat.v1.train.AdamOptimizer(learning_rate_main)
            elif optimizer_type_main == 'sgd':
                optimizer_main = tf.compat.v1.train.MomentumOptimizer(learning_rate_main, _DEFAULT_MOMENTUM)
            else:
                raise NotImplementedError(f'optimizer "{optimizer_type_main}" is not supported')

            gradients_main = optimizer_main.compute_gradients(
                loss_main,
                var_list=main_vars
            )

            gradients_main = optimizer_main.apply_gradients(
                gradients_main,
                global_step_main,
            )

            train_operations_main = TrainOperations(
                gradients_main,
                learning_rate_main,
                loss_main,
                global_step_main,
                main_vars,
                optimizer_main.variables(),
            )

            # print(f"Main optimizer: {optimizer_main}")
            # for var in optimizer_main.variables():
            #     print(var)

            # ARCHITECTURE:
            global_step_arch = tf.Variable(0, trainable=False, name='global_step_arch')

            reg_losses_arch = tf.compat.v1.get_collection('architecture_regularization_losses')
            if reg_losses_arch:
                reg_loss_arch = tf.compat.v1.add_n(reg_losses_arch)
                loss_arch = model_loss + _REG_LOSS_WEIGHT * reg_loss_arch
            else:
                loss_arch = model_loss

            learning_rate_arch = tf.compat.v1.placeholder_with_default(
                learning_strategy_arch.initial_lr,
                shape=[],
                name='learning_rate_arch',
            )

            optimizer_type_arch = learning_strategy_arch.optimizer_type.lower()
            if optimizer_type_arch == 'adam':
                optimizer_arch = tf.compat.v1.train.AdamOptimizer(learning_rate_arch)
            elif optimizer_type_arch == 'sgd':
                optimizer_arch = tf.compat.v1.train.MomentumOptimizer(learning_rate_arch, _DEFAULT_MOMENTUM)
            else:
                raise NotImplementedError(f'optimizer "{optimizer_type_arch}" is not supported')

            gradients_arch = optimizer_arch.compute_gradients(
                loss_arch,
                var_list=arch_vars
            )
            gradients_arch = optimizer_arch.apply_gradients(
                gradients_arch,
                global_step_arch,
            )
            train_operations_arch = TrainOperations(
                gradients_arch,
                learning_rate_arch,
                loss_arch,
                global_step_arch,
                arch_vars,
                optimizer_arch.variables(),
            )

            # print(f"Arch optimizer: {optimizer_arch}")
            # for var in optimizer_arch.variables():
            #     print(var)

            train_input_x = model_input
            train_input_y = rv_output

            return trainable_graph, train_operations_main, train_operations_arch, train_input_x, train_input_y

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
            train_operations_main,
            train_operations_arch,
            train_input_x,
            train_input_y,
            learning_strategy_main,
            learning_strategy_arch,
            n_epochs_main,
            n_epochs_arch,
            n_epochs_interval,
            batch_size,
            checkpoints_save_dir,
            tensorboard_log_dir,
            dataset_kind,
            add_noise,
            stddev,
    ):

        LOGGER.info(f'Total number of trainable vars {len(train_operations_main.train_variables)}')
        LOGGER.info(f'Total number of architecture vars {len(train_operations_arch.train_variables)}')
        LOGGER.info(f'Total number of main optimizer vars {len(train_operations_main.optimizer_variables)}')
        LOGGER.info(f'Total number of arch optimizer vars {len(train_operations_arch.optimizer_variables)}')

        regular_checkpoints_saver = tf.compat.v1.train.Saver(
            var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES),
            # var_list=train_operations_main.train_variables
            #          + train_operations_arch.train_variables
            #          + train_operations_main.optimizer_variables
            #          + train_operations_arch.optimizer_variables
            #          + [train_operations_main.global_step]
            #          + [train_operations_arch.global_step],
            max_to_keep=_NUMBER_OF_REGULAR_CHECKPOINTS,
        )
        best_loss_checkpoints_saver = tf.compat.v1.train.Saver(
            var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES),
            # var_list=train_operations_main.train_variables
            #          + train_operations_arch.train_variables,
            max_to_keep=_NUMBER_OF_BEST_LOSS_CHECKPOINTS,
        )

        # general summaries:
        summary_writer = tf.compat.v1.summary.FileWriter(tensorboard_log_dir, session.graph)
        learning_rate_summary_main = tf.compat.v1.summary.scalar(
            'train_main_learning_rate', train_operations_main.learning_rate)
        learning_rate_summary_arch = tf.compat.v1.summary.scalar(
            'train_arch_learning_rate', train_operations_arch.learning_rate)

        train_loss_summary_main = tf.compat.v1.summary.scalar('train_main_loss', train_operations_main.loss)
        train_loss_summary_arch = tf.compat.v1.summary.scalar('train_arch_loss', train_operations_arch.loss)

        # test_mean_loss_ph = tf.compat.v1.placeholder(tf.float32, ())
        # test_loss_summary = tf.compat.v1.summary.scalar('test_mean_loss', test_mean_loss_ph)

        # TODO: TMP, categorical logits and probs summaries
        # cat_logits = session.graph.get_tensor_by_name('MixtureOutputLayer/CategoricalOutputLayer/logits/BiasAdd:0')
        # cat_probs = session.graph.get_tensor_by_name(
        #     'MixtureOutputLayer_1/random_variable/CategoricalOutputLayer/random_variable/Categorical/probs:0'
        # )
        # n_classes = cat_logits.shape.as_list()[-1]
        # cat_logits_summaries = []
        # cat_probs_summaries = []
        # for n in range(n_classes):
        #     cat_logits_summaries.append(tf.compat.v1.summary.histogram('cat_logits', cat_logits[..., n]))
        #     cat_probs_summaries.append(tf.compat.v1.summary.histogram('cat_probs', cat_probs[..., n]))
        # # TODO: end of TMP

        initial_learning_rate_main = session.run(train_operations_main.learning_rate)
        learning_rate_main = initial_learning_rate_main

        initial_learning_rate_arch = session.run(train_operations_arch.learning_rate)
        learning_rate_arch = initial_learning_rate_arch

        decay_step_main = decay_step_arch = 0
        tolerance_step_main = tolerance_step_arch = 0
        nans_step_counter_main = nans_step_counter_arch = 0

        best_loss_main = best_loss_arch = float('inf')
        tolerance_best_loss_main = tolerance_best_loss_arch = float('inf')
        best_loss_step_main = best_loss_step_arch = 0

        train_dataset, test_dataset = self._get_datasets(
            model,
            batch_size=batch_size,
            kind=dataset_kind,
            add_noise=add_noise,
            stddev=stddev,
        )

        def _reset_optimizer(which_optimizer):
            if which_optimizer == 'main':
                optimizer_vars = train_operations_main.optimizer_variables
            elif which_optimizer == 'arch':
                optimizer_vars = train_operations_arch.optimizer_variables
            else:
                raise ValueError(f"`which_optimizer` option not recognized")
            # session.run(tf.compat.v1.variables_initializer(optimizer_vars))
            session.run([var.initializer for var in optimizer_vars])

        def _get_regular_checkpoint_path(step):
            return os.path.join(checkpoints_save_dir, f'regular_ckpt_{step}')

        def _get_best_checkpoint_path(step):
            return os.path.join(checkpoints_save_dir, f'best_ckpt_{step}')

        def _run_train_step(train_operations, X, Y, lr, lr_summary, loss_summary):
            feed_dict = {
                train_input_x: X,
                train_input_y: Y,
                train_operations.learning_rate: lr,
            }
            fetches = {
                'gradients': train_operations.gradients,
                'loss': train_operations.loss,
                'global_step': train_operations.global_step,
                'learning_rate_summary': lr_summary,
                'loss_summary': loss_summary,
            }
            res = session.run(fetches=fetches, feed_dict=feed_dict)
            return res

        # def _run_test():
        #     test_losses = []
        #
        #     for X_test, Y_test in test_dataset:
        #         feed_dict = {
        #             train_input_x: X_test,
        #             train_input_y: Y_test,
        #         }
        #         fetches = {
        #             'loss': train_operations.loss,
        #         }
        #
        #         res = session.run(fetches=fetches, feed_dict=feed_dict)
        #         test_losses.append(res['loss'])
        #
        #     test_mean_loss = np.mean(test_losses)
        #     test_loss_summary_val = session.run(
        #         test_loss_summary,
        #         feed_dict={test_mean_loss_ph: test_mean_loss}
        #     )
        #     summary_writer.add_summary(test_loss_summary_val, epoch)

        # def _run_cat_histograms():
        #     cat_logits_summaries_values, cat_probs_summaries_values = session.run(
        #         [cat_logits_summaries, cat_probs_summaries],
        #         {train_input_x: X_train}
        #     )
        #     for summary_val in cat_logits_summaries_values:
        #         summary_writer.add_summary(summary_val, global_step)
        #     for summary_val in cat_probs_summaries_values:
        #         summary_writer.add_summary(summary_val, global_step)

        def _maybe_drop_lr_tolerance(learning_strategy, lr, best_loss, tol_step, tol_best_loss):
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
                    LOGGER.info(f"drop lr: {lr}")
            return lr, tol_step, tol_best_loss

        def _maybe_drop_lr_decay(learning_strategy, lr, initial_lr, global_step, decay_step):
            if learning_strategy.__class__.__name__ == 'ExpDecayLearningStrategy':
                lr = initial_lr.copy()
                lr *= np.exp(-global_step * learning_strategy.lr_decay)
                if learning_strategy.lr_cos_steps is not None:
                    lr *= np.abs(
                        np.cos(learning_strategy.lr_cos_phase * decay_step / learning_strategy.lr_cos_steps)
                    )
                lr += learning_strategy.minimal_lr
            return lr

        def _handle_nans(counter, global_step):
            LOGGER.warning(f'Loss is None on step {global_step}, restore previous checkpoint...')
            counter += 1
            checkpoint_step = global_step // _REGULAR_CHECKPOINTS_DELTA
            checkpoint_step -= counter
            checkpoint_step = max(checkpoint_step, 0)
            checkpoint_step *= _REGULAR_CHECKPOINTS_DELTA

            if checkpoint_step == 0:
                LOGGER.info(f'checkpoint_step is 0, reinitialize all variables...')
                session.run([
                    var.initializer
                    for var in train_operations_main.train_variables
                               + train_operations_arch.train_variables
                               + [train_operations_main.global_step]
                               + [train_operations_arch.global_step]
                ])
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
            _reset_optimizer('main')
            _reset_optimizer('arch')
            return counter

        LOGGER.info('Training...')

        for epoch in range(n_epochs_main):

            epoch_start_time = time()
            epoch_steps = 0

            LOGGER.info(f'\nEpoch: {epoch + 1}')

            for X_train, Y_train in train_dataset:

                result = _run_train_step(
                    train_operations=train_operations_main,
                    X=X_train,
                    Y=Y_train,
                    lr=learning_rate_main,
                    lr_summary=learning_rate_summary_main,
                    loss_summary=train_loss_summary_main
                )
                global_step_main = result['global_step']

                # HANDLE NAN VALUES
                if np.isnan(result['loss']):
                    nans_step_counter_main = _handle_nans(nans_step_counter_main, global_step_main)
                    continue

                # drop lr for next step if train with exponential decay
                learning_rate_main = _maybe_drop_lr_decay(
                    learning_strategy_main,
                    learning_rate_main,
                    initial_learning_rate_main,
                    global_step_main,
                    decay_step_main
                )

                summary_writer.add_summary(result['learning_rate_summary'], global_step_main)
                summary_writer.add_summary(result['loss_summary'], global_step_main)

                # TODO: TMP, categorical logits and probs summaries:
                # if global_step % 300 == 0:
                #     _run_cat_histograms()
                # TODO: end of TMP

                # save best checkpoint
                if result['loss'] < best_loss_main:
                    best_loss_checkpoints_saver.save(session, _get_best_checkpoint_path(global_step_main))
                    best_loss_main, best_loss_step_main = result['loss'], global_step_main

                # save regular checkpoint
                if global_step_main % _REGULAR_CHECKPOINTS_DELTA == 0:
                    nans_step_counter_main = 0
                    regular_checkpoints_saver.save(session, _get_regular_checkpoint_path(global_step_main))

                # reset optimizer parameters for next cosine phase
                if learning_strategy_main.__class__.__name__ == 'ExpDecayLearningStrategy':
                    if learning_strategy_main.lr_cos_steps is not None:
                        if global_step_main % learning_strategy_main.lr_cos_steps == 0 and global_step_main > 0:
                            LOGGER.info('Reinitialize optimizer...')
                            _reset_optimizer('main')
                            decay_step_main = 0

                decay_step_main += 1
                epoch_steps += 1

            # drop lr for next epoch if train with tolerance
            learning_rate_main, tolerance_step_main, tolerance_best_loss_main = _maybe_drop_lr_tolerance(
                learning_strategy=learning_strategy_main,
                lr=learning_rate_main,
                best_loss=best_loss_main,
                tol_step=tolerance_step_main,
                tol_best_loss=tolerance_best_loss_main,
            )

            epoch_time = time() - epoch_start_time
            avg_step_time = epoch_time / epoch_steps

            # # TEST
            # test_start_time = time()
            # _run_test()
            # test_time = time() - test_start_time

            LOGGER.info(
                f' === Main ==='
                f' = Minimal loss value = {best_loss_main},\n'
                f' - {epoch_steps} steps took {epoch_time:.1f} seconds, avg_step_time={avg_step_time:.3f}\n'
            )

            if epoch % n_epochs_interval == 0 and epoch > 0:

                LOGGER.info(f"\tStart architecture training...")

                for epoch_arch in range(n_epochs_arch):

                    LOGGER.info(f"\tEpoch: {epoch_arch + 1}")

                    epoch_start_time = time()
                    epoch_steps = 0

                    for X_train_arch, Y_train_arch in test_dataset:

                        result = _run_train_step(
                            train_operations=train_operations_arch,
                            X=X_train_arch,
                            Y=Y_train_arch,
                            lr=learning_rate_arch,
                            lr_summary=learning_rate_summary_arch,
                            loss_summary=train_loss_summary_arch
                        )
                        global_step_arch = result['global_step']

                        # HANDLE NAN VALUES
                        if np.isnan(result['loss']):
                            nans_step_counter_main = _handle_nans(nans_step_counter_main, global_step_main)
                            continue

                        # drop lr for next step if train with exponential decay
                        learning_rate_arch = _maybe_drop_lr_decay(
                            learning_strategy_arch,
                            learning_rate_arch,
                            initial_learning_rate_arch,
                            global_step_arch,
                            decay_step_arch
                        )

                        summary_writer.add_summary(result['learning_rate_summary'], global_step_arch)
                        summary_writer.add_summary(result['loss_summary'], global_step_arch)

                        if result['loss'] < best_loss_arch:
                            # best_loss_checkpoints_saver.save(session, _get_best_checkpoint_path(global_step_arch))
                            best_loss_arch, best_loss_step_arch = result['loss'], global_step_arch

                        # reset optimizer parameters for next cosine phase
                        if learning_strategy_arch.__class__.__name__ == 'ExpDecayLearningStrategy':
                            if learning_strategy_arch.lr_cos_steps is not None:
                                if global_step_arch % learning_strategy_arch.lr_cos_steps == 0 and global_step_arch > 0:
                                    LOGGER.info('\tReinitialize optimizer...')
                                    _reset_optimizer('main')
                                    decay_step_arch = 0

                        decay_step_arch += 1
                        epoch_steps += 1

                    # drop lr for next epoch if train with tolerance
                    learning_rate_arch, tolerance_step_arch, tolerance_best_loss_arch = _maybe_drop_lr_tolerance(
                        learning_strategy=learning_strategy_arch,
                        lr=learning_rate_arch,
                        best_loss=best_loss_arch,
                        tol_step=tolerance_step_arch,
                        tol_best_loss=tolerance_best_loss_arch,
                    )

                    epoch_time = time() - epoch_start_time
                    avg_step_time = epoch_time / epoch_steps

                    LOGGER.info(
                        f'\t === Arch ==='
                        f'\t = Minimal loss value = {best_loss_arch},\n'
                        f'\t - {epoch_steps} steps took {epoch_time:.1f} seconds, avg_step_time={avg_step_time:.3f}\n'
                    )

                # variables = session.run(train_operations_arch.train_variables)
                # for v in variables:
                #     print(v)
                variables = train_operations_arch.train_variables
                for v in variables:
                    print(v.name, session.run(v))

        best_checkpoint_path = _get_best_checkpoint_path(best_loss_step_main)

        return best_checkpoint_path
