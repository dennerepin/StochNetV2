import tensorflow as tf
from stochnet_v2.utils.registry import Registry


OP_REGISTRY = Registry(name="OpRegistry")


def expand_identity(x, expansion_coeff=2):
    n_dims = x.shape.ndims
    x_shape = x.shape.as_list()
    final_shape = [i for i in x_shape[:-1]] + [x_shape[-1] * expansion_coeff]
    final_shape = [i or -1 for i in final_shape[::-1]]
    x = tf.transpose(x, [i for i in range(n_dims-1, -1, -1)])
    x = tf.tile(x, [1 for _ in range(n_dims - 1)] + [expansion_coeff])
    x = tf.reshape(x, final_shape)
    x = tf.transpose(x, [i for i in range(n_dims-1, -1, -1)])
    return x


@OP_REGISTRY.register('dense')
def dense(x, expansion_coeff):
    n_units = x.shape.as_list()[-1] * expansion_coeff
    return tf.compat.v1.layers.Dense(n_units)(x)


@OP_REGISTRY.register('none')
def none(x, expansion_coeff):
    n_dims = x.shape.ndims
    return tf.tile(tf.compat.v1.zeros_like(x), [1 for _ in range(n_dims - 1)] + [expansion_coeff])


@OP_REGISTRY.register('skip_connect')
def skip_connect(x, expansion_coeff):
    return tf.compat.v1.identity(x) if expansion_coeff == 1 else expand_identity(x, expansion_coeff)


@OP_REGISTRY.register('relu')
def relu(x, expansion_coeff):
    if expansion_coeff != 1:
        x = expand_identity(x, expansion_coeff)
    return tf.compat.v1.nn.relu(x)


# @OP_REGISTRY.register('relu6')
# def relu6(x):
#     return tf.compat.v1.nn.relu6(x)
#
#
# @OP_REGISTRY.register('elu')
# def elu(x):
#     return tf.compat.v1.keras.layers.ELU(1.0)(x)
