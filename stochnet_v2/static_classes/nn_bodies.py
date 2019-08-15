import numpy as np
import tensorflow as tf

from stochnet_v2.utils.registry import ACTIVATIONS_REGISTRY
from stochnet_v2.utils.registry import CONSTRAINTS_REGISTRY
from stochnet_v2.utils.registry import REGULARIZERS_REGISTRY

# Dense = tf.keras.layers.Dense
# Add = tf.keras.layers.Add
Dense = tf.layers.Dense


def dummy_body(input_tensor):
    shape = input_tensor.shape.as_list()
    # x = tf.keras.layers.Reshape((np.prod(shape[1:]),))(input_tensor)
    x = tf.reshape(input_tensor, (np.prod(shape[1:]),))
    x = Dense(500, activation='relu')(x)
    return x


def _residual_block_a(
        x,
        hidden_size,
        activation,
        regularization_params,
):
    h1 = Dense(hidden_size, **regularization_params)(x)
    h1 = activation(h1)

    h2 = Dense(hidden_size, **regularization_params)(h1)
    h2 = activation(h2)

    h2 = tf.add(h1, h2)
    return h2


# def _residual_block_b(
#         x,
#         hidden_size,
#         activation_fn,
#         regularization_params,
# ):
#     # TODO: check wtf bad losses compared to _residual_block_a
#     h1 = activation_fn(x)
#     h1 = Dense(hidden_size, **regularization_params)(h1)
#
#     h2 = activation_fn(h1)
#     h2 = Dense(hidden_size, **regularization_params)(h2)
#
#     h2 = tf.keras.layers.tf.add(h1, h2)
#     return h2


def body_a(
        x,
        hidden_size,
        n_residual_blocks,
        activation="none",
        activity_regularizer="none",
        kernel_constraint="none",
        kernel_regularizer="none",
        bias_constraint="none",
        bias_regularizer="none",
):
    regularization_params = {
        'activity_regularizer': REGULARIZERS_REGISTRY[activity_regularizer],
        'kernel_constraint': CONSTRAINTS_REGISTRY[kernel_constraint],
        'kernel_regularizer': REGULARIZERS_REGISTRY[kernel_regularizer],
        'bias_constraint': CONSTRAINTS_REGISTRY[bias_constraint],
        'bias_regularizer': REGULARIZERS_REGISTRY[bias_regularizer],
    }

    activation_fn = ACTIVATIONS_REGISTRY[activation]

    shape = x.shape.as_list()
    # x = tf.keras.layers.Reshape((np.prod(shape[1:]),))(x)
    x = tf.reshape(x, shape=(shape[0] or -1, np.prod(shape[1:]),))

    for _ in range(n_residual_blocks):
        x = _residual_block_a(x, hidden_size, activation_fn, regularization_params)

    return x


