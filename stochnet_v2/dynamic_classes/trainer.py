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
from stochnet_v2.dynamic_classes.nn_body_search import get_genotypes
from stochnet_v2.utils.util import visualize_genotypes

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
        'gradients_'
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

Summaries = namedtuple(
    'Summaries',
    [
        'summary_writer',
        'loss_summary',
        'learning_rate_summary',
        'test_mean_loss_ph',
        'test_loss_summary',
    ]
)


_MINIMAL_LEARNING_RATE = 10 ** - 7
_NUMBER_OF_REGULAR_CHECKPOINTS = 10
_NUMBER_OF_BEST_LOSS_CHECKPOINTS = 5
_REGULAR_CHECKPOINTS_DELTA = 1000
_DEFAULT_NUMBER_OF_EPOCHS_MAIN = 100
_DEFAULT_NUMBER_OF_EPOCHS_HEAT_UP = 20
_DEFAULT_N_EPOCHS_INTERVAL = 5
_DEFAULT_NUMBER_OF_EPOCHS_ARCH = 5
_DEFAULT_BATCH_SIZE = 256
_DEFAULT_PREFETCH_SIZE = 10
_DEFAULT_MOMENTUM = 0.9

_REG_LOSS_WEIGHT = 1.0

# _DEFAULT_LEARNING_STRATEGY = ExpDecayLearningStrategy(
#     optimizer_type='adam',
#     initial_lr=1e-4,
#     lr_decay=1e-4,
#     lr_cos_steps=0,
#     lr_cos_phase=np.pi / 2,
#     minimal_lr=1e-7,
# )

_DEFAULT_LEARNING_STRATEGY_MAIN = ToleranceDropLearningStrategy(
    optimizer_type='adam',
    initial_lr=1e-4,
    lr_decay=0.3,
    epochs_tolerance=7,
    minimal_lr=1e-7,
)

_DEFAULT_LEARNING_STRATEGY_ARCH = ToleranceDropLearningStrategy(
    optimizer_type='adam',
    initial_lr=1e-3,
    lr_decay=0.5,
    epochs_tolerance=20,
    minimal_lr=1e-7,
)

_DEFAULT_LEARNING_STRATEGY_FINETUNE = ToleranceDropLearningStrategy(
    optimizer_type='adam',
    initial_lr=1e-4,
    lr_decay=0.3,
    epochs_tolerance=5,
    minimal_lr=1e-7,
)


class Trainer:

    def train(
            self,
            model,
            n_epochs_main=None,
            n_epochs_heat_up=None,
            n_epochs_arch=None,
            n_epochs_interval=None,
            n_epochs_finetune=None,
            batch_size=None,
            learning_strategy_main=None,
            learning_strategy_arch=None,
            learning_strategy_finetune=None,
            save_dir=None,
            ckpt_path=None,
            dataset_kind='tfrecord',
            add_noise=False,
            stddev=0.01,
            mode='search_finetune'
    ):
        save_dir = save_dir or model.model_explorer.model_folder
        batch_size = batch_size or _DEFAULT_BATCH_SIZE
        n_epochs_main = n_epochs_main or _DEFAULT_NUMBER_OF_EPOCHS_MAIN
        n_epochs_heat_up = n_epochs_heat_up or _DEFAULT_NUMBER_OF_EPOCHS_HEAT_UP
        n_epochs_arch = n_epochs_arch or _DEFAULT_NUMBER_OF_EPOCHS_ARCH
        n_epochs_interval = n_epochs_interval or _DEFAULT_N_EPOCHS_INTERVAL
        learning_strategy_main = learning_strategy_main or _DEFAULT_LEARNING_STRATEGY_MAIN
        learning_strategy_arch = learning_strategy_arch or _DEFAULT_LEARNING_STRATEGY_ARCH
        learning_strategy_finetune = learning_strategy_finetune or _DEFAULT_LEARNING_STRATEGY_FINETUNE

        if 'search' in mode:
            trainable_graph, train_operations_main, train_operations_arch, train_input_x, train_input_y = \
                self._build_trainable_graph_search(model, learning_strategy_main, learning_strategy_arch)

            train_dataset, test_dataset = self._get_datasets(
                model,
                batch_size=batch_size,
                kind=dataset_kind,
                add_noise=add_noise,
                stddev=stddev,
            )

            with tf.compat.v1.Session(graph=trainable_graph) as session:

                clear_train_dir = ckpt_path is None
                tensorboard_log_dir = os.path.join(save_dir, 'tensorboard')
                checkpoints_save_dir = os.path.join(save_dir, 'checkpoints')
                genotypes_save_dir = os.path.join(save_dir, 'genotypes')
                maybe_create_dir(tensorboard_log_dir, erase_existing=clear_train_dir)
                maybe_create_dir(checkpoints_save_dir, erase_existing=clear_train_dir)
                maybe_create_dir(genotypes_save_dir, erase_existing=clear_train_dir)

                # savers:
                regular_checkpoints_saver = tf.compat.v1.train.Saver(
                    var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES),
                    max_to_keep=_NUMBER_OF_REGULAR_CHECKPOINTS,
                )
                best_loss_checkpoints_saver = tf.compat.v1.train.Saver(
                    var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES),
                    max_to_keep=_NUMBER_OF_BEST_LOSS_CHECKPOINTS,
                )

                # summaries:
                summary_writer = tf.compat.v1.summary.FileWriter(tensorboard_log_dir, session.graph)

                learning_rate_summary_main = tf.compat.v1.summary.scalar(
                    'train_learning_rate_main', train_operations_main.learning_rate)
                loss_summary_main = tf.compat.v1.summary.scalar(
                    # 'train_loss_main', train_operations_main.loss)
                    'train_loss_main', tf.reduce_mean(train_operations_main.loss))  # TODO: for vector loss

                learning_rate_summary_arch = tf.compat.v1.summary.scalar(
                    'train_learning_rate_arch', train_operations_arch.learning_rate)
                loss_summary_arch = tf.compat.v1.summary.scalar(
                    # 'train_loss_arch', train_operations_arch.loss)
                    'train_loss_arch', tf.reduce_mean(train_operations_arch.loss))  # TODO: for vector loss

                test_mean_loss_ph = tf.compat.v1.placeholder(tf.float32, ())
                test_loss_summary_main = tf.compat.v1.summary.scalar(
                    'test_mean_loss_main', test_mean_loss_ph)
                test_loss_summary_arch = tf.compat.v1.summary.scalar(
                    'test_mean_loss_arch', test_mean_loss_ph)

                summaries_main = Summaries(
                    summary_writer=summary_writer,
                    loss_summary=loss_summary_main,
                    learning_rate_summary=learning_rate_summary_main,
                    test_mean_loss_ph=test_mean_loss_ph,
                    test_loss_summary=test_loss_summary_main,

                )
                summaries_arch = Summaries(
                    summary_writer=summary_writer,
                    loss_summary=loss_summary_arch,
                    learning_rate_summary=learning_rate_summary_arch,
                    test_mean_loss_ph=test_mean_loss_ph,
                    test_loss_summary=test_loss_summary_arch,
                )

                if ckpt_path is None:
                    session.run(tf.compat.v1.global_variables_initializer())
                else:
                    regular_checkpoints_saver.restore(session, ckpt_path)

                training_state_main = [
                    session.run(train_operations_main.learning_rate),  # learning_rate
                    0,                                                 # decay_step
                    0,                                                 # tolerance_step
                    0,                                                 # nans_step_counter
                    float('inf'),                                      # best_loss
                    float('inf'),                                      # tolerance_best_loss
                    0,                                                 # best_loss_step
                    0,                                                 # epoch
                ]

                training_state_arch = [
                    session.run(train_operations_arch.learning_rate),  # learning_rate
                    0,                                                 # decay_step
                    0,                                                 # tolerance_step
                    0,                                                 # nans_step_counter
                    float('inf'),                                      # best_loss
                    float('inf'),                                      # tolerance_best_loss
                    0,                                                 # best_loss_step
                    0,                                                 # epoch
                ]

                # n_iterations = n_epochs_main // n_epochs_interval
                # for _ in range(n_iterations):

                iteration = 0
                epoch = 0

                while not epoch >= n_epochs_main:

                    if iteration == 0:
                        train_epochs_main = n_epochs_heat_up
                        train_epochs_arch = n_epochs_arch
                    else:
                        train_epochs_main = min(epoch + n_epochs_interval, n_epochs_main) - epoch
                        train_epochs_arch = n_epochs_arch if train_epochs_main == n_epochs_interval else 0

                    LOGGER.info('\nTraining MAIN...\n')

                    best_loss_checkpoint_path, training_state_main = self._train(
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        session=session,
                        train_operations=train_operations_main,
                        traning_state=training_state_main,
                        train_input_x=train_input_x,
                        train_input_y=train_input_y,
                        learning_strategy=learning_strategy_main,
                        n_epochs=train_epochs_main,
                        checkpoints_save_dir=checkpoints_save_dir,
                        regular_checkpoints_saver=regular_checkpoints_saver,
                        best_loss_checkpoints_saver=best_loss_checkpoints_saver,
                        summaries=summaries_main,
                        save_checkpoints=True,
                    )

                    if train_epochs_arch > 0:

                        LOGGER.info('\nTraining ARCH...\n')

                        _, training_state_arch = self._train(
                            train_dataset=test_dataset,
                            test_dataset=None,
                            session=session,
                            train_operations=train_operations_arch,
                            traning_state=training_state_arch,
                            train_input_x=train_input_x,
                            train_input_y=train_input_y,
                            learning_strategy=learning_strategy_arch,
                            n_epochs=train_epochs_arch,
                            checkpoints_save_dir=checkpoints_save_dir,
                            regular_checkpoints_saver=regular_checkpoints_saver,
                            best_loss_checkpoints_saver=best_loss_checkpoints_saver,
                            summaries=summaries_arch,
                            save_checkpoints=False,
                        )

                        variables = train_operations_arch.train_variables
                        arch_loss = tf.compat.v1.get_collection('architecture_regularization_losses')
                        arch_loss_vals = session.run(arch_loss)
                        for i, v in enumerate(variables):
                            v_softmax_name = v.name.replace('alphas:0', 'Softmax:0')
                            v_val, v_softmax_val = session.run([v, v_softmax_name])
                            LOGGER.debug(
                                f"{v.name} \n"
                                f"\t{v_val},  min={np.min(v_val):.3f}, max={np.max(v_val):.3f} -> \n"
                                f"\t{v_softmax_val}, min={np.min(v_softmax_val):.3f}, max={np.max(v_softmax_val):.3f}, "
                                f"reg_loss={arch_loss_vals[i] if arch_loss_vals else 0}\n"
                            )

                        genotypes = get_genotypes(
                            session,
                            model.n_cells,
                            model.cell_size,
                            model.n_states_reduce,
                        )
                        epoch_arch = training_state_arch[-1]
                        visualize_genotypes(
                            genotypes,
                            os.path.join(genotypes_save_dir, f'epoch_{epoch_arch}_genotype')
                        )

                    epoch += train_epochs_main
                    iteration += 1

                model.restore_from_checkpoint(best_loss_checkpoint_path)
                model.save_genotypes()
                model.recreate_from_genome(best_loss_checkpoint_path)

            if 'finetune' in mode:
                best_loss_checkpoint_path = self.finetune(
                    recreated_model=model,
                    recreated_model_ckpt_path=best_loss_checkpoint_path,
                    learning_strategy_finetune=learning_strategy_finetune,
                    batch_size=batch_size,
                    n_epochs_finetune=n_epochs_finetune,
                    dataset_kind=dataset_kind,
                    add_noise=add_noise,
                    stddev=stddev,
                )
            else:
                model.save()

        elif 'finetune' in mode:
            if ckpt_path is None:
                raise ValueError("Should provide ckp_path (of search algorithm) to recreate model for fine-tuning")
            model.restore_from_checkpoint(ckpt_path)
            model.save_genotypes()
            model.recreate_from_genome(ckpt_path)
            best_loss_checkpoint_path = self.finetune(
                    recreated_model=model,
                    recreated_model_ckpt_path=ckpt_path,
                    learning_strategy_finetune=learning_strategy_finetune,
                    batch_size=batch_size,
                    n_epochs_finetune=n_epochs_finetune,
                    dataset_kind=dataset_kind,
                    add_noise=add_noise,
                    stddev=stddev,
                )
        else:
            raise ValueError(
                f"The `mode` parameter not understood: {mode}."
                f" Expected list or string including optional 'search' and 'finetune' keywords."
            )

        return best_loss_checkpoint_path

    def finetune(
            self,
            recreated_model,
            recreated_model_ckpt_path,
            save_dir=None,
            learning_strategy_finetune=None,
            batch_size=None,
            n_epochs_finetune=None,
            dataset_kind='tfrecord',
            add_noise=False,
            stddev=0.01,
    ):
        save_dir = save_dir or recreated_model.model_explorer.model_folder

        train_dataset, test_dataset = self._get_datasets(
            recreated_model,
            batch_size=batch_size,
            kind=dataset_kind,
            add_noise=add_noise,
            stddev=stddev,
        )

        trainable_graph, train_operations_finetune, train_input_x, train_input_y = \
            self._build_trainable_graph_finetune(recreated_model, learning_strategy_finetune)

        with tf.compat.v1.Session(graph=trainable_graph) as session:

            tensorboard_log_dir = os.path.join(save_dir, 'finetuning', 'tensorboard')
            checkpoints_save_dir = os.path.join(save_dir, 'finetuning', 'checkpoints')
            maybe_create_dir(tensorboard_log_dir, erase_existing=True)
            maybe_create_dir(checkpoints_save_dir, erase_existing=True)

            # savers:
            regular_checkpoints_saver = tf.compat.v1.train.Saver(
                var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES),
                max_to_keep=_NUMBER_OF_REGULAR_CHECKPOINTS,
            )
            best_loss_checkpoints_saver = tf.compat.v1.train.Saver(
                var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES),
                max_to_keep=_NUMBER_OF_BEST_LOSS_CHECKPOINTS,
            )

            tf.compat.v1.train.Saver(
                var_list=list(
                    set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))
                    - {train_operations_finetune.global_step}
                )
            ).restore(session, recreated_model_ckpt_path)
            session.run(train_operations_finetune.global_step.initializer)
            session.run([v.initializer for v in train_operations_finetune.optimizer_variables])

            # summaries:
            summary_writer = tf.compat.v1.summary.FileWriter(tensorboard_log_dir, session.graph)
            learning_rate_summary_finetune = tf.compat.v1.summary.scalar(
                'train_learning_rate_finetune', train_operations_finetune.learning_rate)
            loss_summary_finetune = tf.compat.v1.summary.scalar(
                # 'train_loss_finetune', train_operations_finetune.loss)
                'train_loss_finetune', tf.reduce_mean(train_operations_finetune.loss))  # TODO: for vector loss

            test_mean_loss_ph = tf.compat.v1.placeholder(tf.float32, ())
            test_loss_summary_finetune = tf.compat.v1.summary.scalar(
                'test_mean_loss_finetune', test_mean_loss_ph)

            summaries_finetune = Summaries(
                summary_writer=summary_writer,
                loss_summary=loss_summary_finetune,
                learning_rate_summary=learning_rate_summary_finetune,
                test_mean_loss_ph=test_mean_loss_ph,
                test_loss_summary=test_loss_summary_finetune,

            )

            training_state_finetune = [
                session.run(train_operations_finetune.learning_rate),  # learning_rate
                0,                                                     # decay_step
                0,                                                     # tolerance_step
                0,                                                     # nans_step_counter
                float('inf'),                                          # best_loss
                float('inf'),                                          # tolerance_best_loss
                0,                                                     # best_loss_step
                0,                                                     # epoch
            ]

            best_loss_checkpoint_path, training_state_finetune = self._train(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                session=session,
                train_operations=train_operations_finetune,
                traning_state=training_state_finetune,
                train_input_x=train_input_x,
                train_input_y=train_input_y,
                learning_strategy=learning_strategy_finetune,
                n_epochs=n_epochs_finetune,
                checkpoints_save_dir=checkpoints_save_dir,
                regular_checkpoints_saver=regular_checkpoints_saver,
                best_loss_checkpoints_saver=best_loss_checkpoints_saver,
                summaries=summaries_finetune,
                save_checkpoints=True,
            )
            recreated_model.restore_from_checkpoint(best_loss_checkpoint_path)
            recreated_model.save()

        return best_loss_checkpoint_path

    def _build_trainable_graph_search(
            self,
            model,
            learning_strategy_main,
            learning_strategy_arch,
    ):
        # trainable_graph = tf.compat.v1.Graph()
        # copy_graph(model.graph, trainable_graph)
        # model_input = get_transformed_tensor(model.input_placeholder, trainable_graph)
        # rv_output = get_transformed_tensor(model.rv_output_ph, trainable_graph)
        # model_loss = get_transformed_tensor(model.loss, trainable_graph)

        # tensorflow doesn't copy the custom gradients used by mixed_op_cat,
        # so in this case we work with model graph directly:
        trainable_graph = model.graph
        model_input = model.input_placeholder
        rv_output = model.rv_output_ph
        model_loss = model.loss

        with trainable_graph.as_default():

            trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
            arch_vars = [v for v in trainable_vars if 'architecture_variables' in v.name]
            main_vars = list(set(trainable_vars) - set(arch_vars))

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

            # TODO: gradient clipping
            # gradients_main = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients_main]

            apply_gradients_main = optimizer_main.apply_gradients(
                gradients_main,
                global_step_main,
            )

            train_operations_main = TrainOperations(
                apply_gradients_main,
                learning_rate_main,
                loss_main,
                global_step_main,
                main_vars,
                optimizer_main.variables(),
                gradients_main,
            )

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
            apply_gradients_arch = optimizer_arch.apply_gradients(
                gradients_arch,
                global_step_arch,
            )
            train_operations_arch = TrainOperations(
                apply_gradients_arch,
                learning_rate_arch,
                loss_arch,
                global_step_arch,
                arch_vars,
                optimizer_arch.variables(),
                gradients_arch,
            )

            train_input_x = model_input
            train_input_y = rv_output

            return trainable_graph, train_operations_main, train_operations_arch, train_input_x, train_input_y

    def _build_trainable_graph_finetune(
            self,
            model,
            learning_strategy,
    ):
        # TODO: there is no mixed_op on this stage, so we can safely copy model graph
        # trainable_graph = tf.compat.v1.Graph()
        # copy_graph(model.graph, trainable_graph)
        # model_input = get_transformed_tensor(model.input_placeholder, trainable_graph)
        # rv_output = get_transformed_tensor(model.rv_output_ph, trainable_graph)
        # model_loss = get_transformed_tensor(model.loss, trainable_graph)

        trainable_graph = model.graph
        model_input = model.input_placeholder
        rv_output = model.rv_output_ph
        model_loss = model.loss

        with trainable_graph.as_default():

            trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

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

            gradients = optimizer.compute_gradients(loss, var_list=trainable_vars)
            apply_gradients = optimizer.apply_gradients(
                gradients,
                global_step,
            )

            train_operations = TrainOperations(
                apply_gradients,
                learning_rate,
                loss,
                global_step,
                trainable_vars,
                optimizer.variables(),
                gradients,
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

    @staticmethod
    def _train(
            train_dataset,
            test_dataset,
            session,
            train_operations,
            traning_state,
            train_input_x,
            train_input_y,
            learning_strategy,
            n_epochs,
            checkpoints_save_dir,
            regular_checkpoints_saver,
            best_loss_checkpoints_saver,
            summaries,
            save_checkpoints,
    ):

        LOGGER.debug(f'Total number of trainable vars {len(train_operations.train_variables)}')
        LOGGER.debug(f'Total number of main optimizer vars {len(train_operations.optimizer_variables)}')

        initial_learning_rate = session.run(train_operations.learning_rate)

        [
            learning_rate,
            decay_step,
            tolerance_step,
            nans_step_counter,
            best_loss,
            tolerance_best_loss,
            best_loss_step,
            epoch,
        ] = traning_state

        def _reset_optimizer():
            optimizer_vars = train_operations.optimizer_variables
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
            res['loss'] = np.mean(res['loss'])  # TODO: for vector loss
            return res

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
                if learning_strategy.lr_cos_steps:
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
                    for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
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
            _reset_optimizer()
            return counter

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
                res['loss'] = np.mean(res['loss'])  # TODO: for vector loss
                test_losses.append(res['loss'])

            test_mean_loss = np.mean(test_losses)
            test_loss_summary_val = session.run(
                summaries.test_loss_summary,
                feed_dict={summaries.test_mean_loss_ph: test_mean_loss}
            )
            summaries.summary_writer.add_summary(test_loss_summary_val, epoch)

        for _ in range(n_epochs):

            epoch_start_time = time()
            epoch_steps = 0

            LOGGER.info(f'\nEpoch: {epoch + 1}')

            for X_train, Y_train in train_dataset:

                result = _run_train_step(
                    train_operations=train_operations,
                    X=X_train,
                    Y=Y_train,
                    lr=learning_rate,
                    lr_summary=summaries.learning_rate_summary,
                    loss_summary=summaries.loss_summary
                )
                global_step = result['global_step']

                # HANDLE NAN VALUES
                if np.isnan(result['loss']):
                    nans_step_counter = _handle_nans(nans_step_counter, global_step)
                    continue

                # drop lr for next step if train with exponential decay
                learning_rate = _maybe_drop_lr_decay(
                    learning_strategy,
                    learning_rate,
                    initial_learning_rate,
                    global_step,
                    decay_step
                )

                summaries.summary_writer.add_summary(result['learning_rate_summary'], global_step)
                summaries.summary_writer.add_summary(result['loss_summary'], global_step)

                if result['loss'] < best_loss:
                    best_loss, best_loss_step = result['loss'], global_step
                    if save_checkpoints:
                        best_loss_checkpoints_saver.save(session, _get_best_checkpoint_path(global_step))

                if global_step % _REGULAR_CHECKPOINTS_DELTA == 0:
                    if save_checkpoints:
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
                learning_strategy=learning_strategy,
                lr=learning_rate,
                best_loss=best_loss,
                tol_step=tolerance_step,
                tol_best_loss=tolerance_best_loss,
            )

            epoch_time = time() - epoch_start_time
            avg_step_time = epoch_time / epoch_steps

            LOGGER.info(
                f' = Minimal loss value = {best_loss},\n'
                f' - {epoch_steps} steps took {epoch_time:.1f} seconds, avg_step_time={avg_step_time:.3f}\n'
            )

            # TEST
            if test_dataset is not None:
                test_start_time = time()
                _run_test()
                test_time = time() - test_start_time
                LOGGER.info(f' - test time: {test_time:.1f} seconds')

            epoch += 1

        best_checkpoint_path = _get_best_checkpoint_path(best_loss_step)

        training_state = [
            learning_rate,
            decay_step,
            tolerance_step,
            nans_step_counter,
            best_loss,
            tolerance_best_loss,
            best_loss_step,
            epoch,
        ]

        return best_checkpoint_path, training_state
