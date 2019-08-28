import tensorflow as tf
from stochnet_v2.utils.registry import Registry


OP_REGISTRY = Registry(name="OpRegistry")


@OP_REGISTRY.register('dense')
def dense(x):
    n_units = x.shape.as_list()[-1]
    return tf.layers.Dense(n_units)(x)


@OP_REGISTRY.register('none')
def none(x):
    return tf.zeros_like(x)


@OP_REGISTRY.register('skip_connect')
def skip_connect(x):
    return tf.identity(x)


@OP_REGISTRY.register('relu')
def relu(x):
    return tf.nn.relu(x)


@OP_REGISTRY.register('relu6')
def relu6(x):
    return tf.nn.relu6(x)


@OP_REGISTRY.register('elu')
def elu(x):
    return tf.keras.layers.ELU(1.0)(x)
