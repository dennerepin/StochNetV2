import numpy as np
import tensorflow as tf

from stochnet_v2.utils.registry import Registry
from stochnet_v2.utils.registry import ACTIVATIONS_REGISTRY
from stochnet_v2.utils.registry import CONSTRAINTS_REGISTRY
from stochnet_v2.utils.registry import REGULARIZERS_REGISTRY


BatchNorm = tf.compat.v1.layers.BatchNormalization
Dense = tf.layers.Dense
# initializer = tf.initializers.glorot_normal
initializer = tf.compat.v1.initializers.variance_scaling(mode='fan_out', distribution="truncated_normal")
# initializer = None

BODY_FN_REGISTRY = Registry(name="BodyFunctionsRegistry")
RESIDUAL_BLOCKS_REGISTRY = Registry(name="ResidualBlocksRegistry")


# def dummy_body(input_tensor):
#     shape = input_tensor.shape.as_list()
#     # x = tf.keras.layers.Reshape((np.prod(shape[1:]),))(input_tensor)
#     x = tf.reshape(input_tensor, (np.prod(shape[1:]),))
#     x = Dense(500, activation='relu')(x)
#     return x


# @RESIDUAL_BLOCKS_REGISTRY.register("a")
# def _residual_block_a(
#         x,
#         hidden_size,
#         activation,
#         params_dict,
# ):
#     h1 = Dense(hidden_size, **params_dict)(x)
#     h1 = activation(h1)
#
#     h2 = Dense(hidden_size, **params_dict)(h1)
#     h2 = activation(h2)
#
#     h2 = tf.add(h1, h2)
#     return h2


@RESIDUAL_BLOCKS_REGISTRY.register("a")
def residual_block_a(
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


@RESIDUAL_BLOCKS_REGISTRY.register("b")
def residual_block_b(
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
@RESIDUAL_BLOCKS_REGISTRY.register("c")
def residual_block_c(
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
        n_residual_blocks,
        residual_block_fn,
        use_batch_norm,
        activation_fn,
        params_dict,
):
    for _ in range(n_residual_blocks):
        x = residual_block_fn(
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
        n_residual_blocks,
        residual_block_fn,
        use_batch_norm,
        activation_fn,
        params_dict,
):
    x = Dense(hidden_size, **params_dict)(x)
    x = activation_fn(x)
    if use_batch_norm:
        x = BatchNorm()(x)

    for _ in range(n_residual_blocks):
        x = residual_block_fn(
            x,
            hidden_size,
            activation_fn,
            params_dict,
            use_batch_norm=use_batch_norm
        )

    return x


def body_main(
        x,
        hidden_size,
        n_residual_blocks,
        body_fn_name="body_a",
        residual_block_name="a",
        use_batch_norm=False,
        activation="none",
        activity_regularizer="none",
        kernel_constraint="none",
        kernel_regularizer="none",
        bias_constraint="none",
        bias_regularizer="none",
):
    print(
        f"\n"
        f" ** Building '{body_fn_name}' body, hidden size: {hidden_size} \n"
        f"    with {n_residual_blocks} of '{residual_block_name}' residual block \n"
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
        'kernel_initializer': initializer,  # TODO
    }
    body_fn = BODY_FN_REGISTRY[body_fn_name]
    residual_block_fn = RESIDUAL_BLOCKS_REGISTRY[residual_block_name]
    activation_fn = ACTIVATIONS_REGISTRY[activation]

    shape = x.shape.as_list()
    x = tf.reshape(x, shape=(shape[0] or -1, np.prod(shape[1:]),))

    x = body_fn(
        x,
        hidden_size,
        n_residual_blocks,
        residual_block_fn=residual_block_fn,
        use_batch_norm=use_batch_norm,
        activation_fn=activation_fn,
        params_dict=params_dict,
    )

    return x
