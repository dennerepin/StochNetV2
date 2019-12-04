import logging
import numpy as np
import tensorflow as tf

from stochnet_v2.utils.registry import Registry
from stochnet_v2.utils.registry import ACTIVATIONS_REGISTRY
from stochnet_v2.utils.registry import CONSTRAINTS_REGISTRY
from stochnet_v2.utils.registry import REGULARIZERS_REGISTRY


BatchNorm = tf.compat.v1.layers.BatchNormalization
Dense = tf.compat.v1.layers.Dense
# initializer = tf.initializers.glorot_normal
initializer = tf.compat.v1.initializers.variance_scaling(mode='fan_out', distribution="truncated_normal")
# initializer = None

BODY_FN_REGISTRY = Registry(name="BodyFunctionsRegistry")
BLOCKS_REGISTRY = Registry(name="ResidualBlocksRegistry")

LOGGER = logging.getLogger('static_classes.nn_bodies')


@BLOCKS_REGISTRY.register("a")
def block_a(
        x,
        hidden_size,
        activation,
        params_dict,
        use_batch_norm=False,
):
    h1 = Dense(hidden_size, **params_dict)(x)
    h1 = activation(h1)
    if use_batch_norm:
        h1 = BatchNorm()(h1)

    h2 = Dense(hidden_size, **params_dict)(h1)
    h2 = activation(h2)
    if use_batch_norm:
        h2 = BatchNorm()(h2)

    h2 = tf.add(h1, h2)
    return h2


@BLOCKS_REGISTRY.register("b")
def block_b(
        x,
        hidden_size,
        activation,
        params_dict,
        use_batch_norm=False,
):
    if use_batch_norm:
        x = BatchNorm()(x)
    h1 = Dense(hidden_size, **params_dict)(x)
    h1 = activation(h1)

    if use_batch_norm:
        h2 = BatchNorm()(h1)
    else:
        h2 = h1
    h2 = Dense(hidden_size, **params_dict)(h2)
    h2 = activation(h2)

    h2 = tf.add(h1, h2)
    return h2


# FOR body_b
@BLOCKS_REGISTRY.register("c")
def block_c(
        x,
        hidden_size,
        activation,
        params_dict,
        use_batch_norm=False,
):

    h1 = Dense(hidden_size, **params_dict)(x)
    h1 = activation(h1)
    if use_batch_norm:
        h1 = BatchNorm()(h1)

    return tf.add(x, h1)


@BODY_FN_REGISTRY.register("body_a")
def body_a(
        x,
        hidden_size,
        n_blocks,
        block_fn,
        use_batch_norm,
        activation_fn,
        params_dict,
):
    for _ in range(n_blocks):
        x = block_fn(
            x,
            hidden_size,
            activation_fn,
            params_dict,
            use_batch_norm=use_batch_norm
        )

    return x


@BODY_FN_REGISTRY.register("body_b")
def body_b(
        x,
        hidden_size,
        n_blocks,
        block_fn,
        use_batch_norm,
        activation_fn,
        params_dict,
):
    x = Dense(hidden_size, **params_dict)(x)
    x = activation_fn(x)
    if use_batch_norm:
        x = BatchNorm()(x)

    for _ in range(n_blocks):
        x = block_fn(
            x,
            hidden_size,
            activation_fn,
            params_dict,
            use_batch_norm=use_batch_norm
        )

    return x


@BLOCKS_REGISTRY.register('lstm')
def lstm(hidden_size):
    return tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_size)


@BLOCKS_REGISTRY.register('gru')
def gru(hidden_size):
    return tf.compat.v1.nn.rnn_cell.GRUCell(hidden_size)


@BODY_FN_REGISTRY.register("body_lstm")
def body_lstm(
        x,
        hidden_size,
        n_blocks,
        block_fn,
        use_batch_norm,
        activation_fn,
        params_dict,
):
    if n_blocks == 1:
        cell = block_fn(hidden_size)
    else:
        cells = [
            block_fn(hidden_size)
            for _ in range(n_blocks)
        ]
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)

    output, state = tf.compat.v1.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = output[:, -1, :]
    return output


def body_main(
        x,
        hidden_size,
        n_blocks,
        body_fn_name="body_a",
        block_name="a",
        use_batch_norm=False,
        activation="none",
        activity_regularizer="none",
        kernel_constraint="none",
        kernel_regularizer="none",
        bias_constraint="none",
        bias_regularizer="none",
):
    LOGGER.info(
        f"\n"
        f" ** Building '{body_fn_name}' body, hidden size: {hidden_size} \n"
        f"    with {n_blocks} of '{block_name}' block \n"
        f"    activation: {activation} \n"
        f"    activity_regularizer: {activity_regularizer} \n"
        f"    kernel_constraint: {kernel_constraint} \n"
        f"    kernel_regularizer: {kernel_regularizer} \n"
        f"    bias_constraint: {bias_constraint} \n"
        f"    bias_regularizer: {bias_regularizer} \n"
        f"    use BatchNorm: {use_batch_norm} \n"
        f" ** "
        f"\n"
    )
    params_dict = {
        'activity_regularizer': REGULARIZERS_REGISTRY[activity_regularizer],
        'kernel_constraint': CONSTRAINTS_REGISTRY[kernel_constraint],
        'kernel_regularizer': REGULARIZERS_REGISTRY[kernel_regularizer],
        'bias_constraint': CONSTRAINTS_REGISTRY[bias_constraint],
        'bias_regularizer': REGULARIZERS_REGISTRY[bias_regularizer],
        'kernel_initializer': initializer,
    }
    body_fn = BODY_FN_REGISTRY[body_fn_name]
    block_fn = BLOCKS_REGISTRY[block_name]
    activation_fn = ACTIVATIONS_REGISTRY[activation]

    if not 'lstm' in body_fn_name:
        shape = x.shape.as_list()
        x = tf.reshape(x, shape=(shape[0] or -1, np.prod(shape[1:]),))

    x = body_fn(
        x,
        hidden_size,
        n_blocks,
        block_fn=block_fn,
        use_batch_norm=use_batch_norm,
        activation_fn=activation_fn,
        params_dict=params_dict,
    )

    return x


