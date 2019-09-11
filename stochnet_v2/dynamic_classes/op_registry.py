import tensorflow as tf
from stochnet_v2.utils.registry import Registry
from stochnet_v2.dynamic_classes.util import expand_identity
from stochnet_v2.dynamic_classes.util import expand_element_wise
from stochnet_v2.dynamic_classes.util import relu_dense_bn
from stochnet_v2.dynamic_classes.util import bn_dense_relu


OP_REGISTRY = Registry(name="OpRegistry")


@OP_REGISTRY.register('simple_dense')
def simple_dense(x, expansion_coeff):
    n_units = x.shape.as_list()[-1] * expansion_coeff
    return tf.compat.v1.layers.Dense(n_units)(x)


@OP_REGISTRY.register('rich_dense_1')
def rich_dense_1(x, expansion_coeff):
    return relu_dense_bn(x, expansion_coeff)


@OP_REGISTRY.register('rich_dense_2')
def rich_dense_2(x, expansion_coeff):
    return bn_dense_relu(x, expansion_coeff)


@OP_REGISTRY.register('none')
def none(x, expansion_coeff):
    n_dims = x.shape.ndims
    with tf.compat.v1.variable_scope("none"):
        x = tf.tile(tf.compat.v1.zeros_like(x), [1 for _ in range(n_dims - 1)] + [expansion_coeff])
    return x


@OP_REGISTRY.register('skip_connect')
def skip_connect(x, expansion_coeff):
    return tf.compat.v1.identity(x) if expansion_coeff == 1 else expand_identity(x, expansion_coeff)


@OP_REGISTRY.register('relu')
def relu(x, expansion_coeff):
    if expansion_coeff != 1:
        x = expand_identity(x, expansion_coeff)
    return tf.compat.v1.nn.relu(x)


@OP_REGISTRY.register('element_wise')
def element_wise(x, expansion_coeff):
    return expand_element_wise(x, expansion_coeff)
