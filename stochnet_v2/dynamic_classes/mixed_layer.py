import numpy as np
import tensorflow as tf

from stochnet_v2.dynamic_classes.op_registry import OP_REGISTRY

PRIMITIVES = [
    'dense',
    'relu',
    'skip_connect',
    'none',
]

null_scope = tf.compat.v1.VariableScope("")


def l2_regularizer(x):
    return 0.01 * tf.compat.v1.nn.l2_loss(x)


def expand_dense(x):
    out_dim = x.shape.as_list()[-1] * 4
    return tf.compat.v1.layers.Dense(out_dim)(x)


def regular_dense(x):
    out_dim = x.shape.as_list()[-1]
    return tf.compat.v1.layers.Dense(out_dim)(x)


def mixed_op(x, index, expand):

    with tf.compat.v1.variable_scope(null_scope):
        alphas = tf.compat.v1.get_variable(
            name=f"alpha_{2 if expand else 1}_{index}",
            shape=[len(PRIMITIVES)],
            initializer=tf.compat.v1.keras.initializers.random_normal,
            trainable=False,
            collections=['architecture_variables'],
        )
        alphas_regularizer = l2_regularizer(alphas)
        tf.compat.v1.add_to_collection('architecture_regularization_losses', alphas_regularizer)

    alphas = tf.nn.softmax(alphas)

    outputs = []
    for idx, primitive in enumerate(PRIMITIVES):

        out = OP_REGISTRY[primitive](x)
        mask = [idx == i for i in range(len(PRIMITIVES))]
        mask = tf.compat.v1.constant(mask, tf.bool)
        alpha = tf.compat.v1.boolean_mask(alphas, mask)

        outputs.append(alpha * out)

    return tf.compat.v1.add_n(outputs)


def cell(
        s0,
        s1,
        cell_size,
        multiplier,
        expand,
        expand_prev,
):
    if expand_prev:
        s0 = expand_dense(s0)
    else:
        s0 = regular_dense(s0)

    s1 = regular_dense(s1)

    state = [s0, s1]
    offset = 0

    for i in range(cell_size):
        tmp = []
        for j in range(i + 2):
            tmp.append(mixed_op(state[j], offset + j, expand))

        offset += len(state)
        state.append(tf.add_n(tmp))

    out = tf.concat(state[-multiplier:], axis=-1)

    return out


def body(x, n_cells=4, cell_size=4, multiplier=4):
    out_dim = x.shape.as_list()[-1]
    s0 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    s1 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    expand_prev = False

    for n in range(n_cells):
        if n in [n_cells // 3, 2 * n_cells // 3]:
            expand = True
        else:
            expand = False

        s0, s1 = s1, cell()