import tensorflow as tf
from stochnet_v2.utils.registry import Registry
from stochnet_v2.dynamic_classes.util import _expand_identity
from stochnet_v2.dynamic_classes.util import _expand_element_wise
from stochnet_v2.dynamic_classes.util import _simple_dense
from stochnet_v2.dynamic_classes.util import _gated_linear_unit
from stochnet_v2.dynamic_classes.util import _activated_dense


OP_REGISTRY = Registry(name="OpRegistry")


@OP_REGISTRY.register('simple_dense')
def simple_dense(x, expansion_coeff, **kwargs):
    return _simple_dense(x, expansion_coeff, **kwargs)


@OP_REGISTRY.register('activated_dense')
def activated_dense(x, expansion_coeff, **kwargs):
    activation_type = 'swish'
    return _activated_dense(x, expansion_coeff, activation_type, **kwargs)


@OP_REGISTRY.register('none')
def none(x, expansion_coeff, **kwargs):
    n_dims = x.shape.ndims
    with tf.compat.v1.variable_scope("none"):
        x = tf.tile(tf.compat.v1.zeros_like(x), [1 for _ in range(n_dims - 1)] + [expansion_coeff])
    return x


@OP_REGISTRY.register('skip_connect')
def skip_connect(x, expansion_coeff, **kwargs):
    return _expand_identity(x, expansion_coeff)


@OP_REGISTRY.register('element_wise')
def element_wise(x, expansion_coeff, **kwargs):
    return _expand_element_wise(x, expansion_coeff, **kwargs)


@OP_REGISTRY.register('gated_linear_unit')
def gated_linear_unit(x, expansion_coeff, **kwargs):
    return _gated_linear_unit(x, expansion_coeff, **kwargs)


@OP_REGISTRY.register('relu')
def relu(x, expansion_coeff, **kwargs):
    if expansion_coeff > 1:
        x = _expand_identity(x, expansion_coeff)
    return tf.compat.v1.nn.relu(x)


@OP_REGISTRY.register('swish')
def swish(x, expansion_coeff, **kwargs):
    if expansion_coeff > 1:
        x = _expand_identity(x, expansion_coeff)
    return tf.compat.v1.nn.swish(x)
