import numpy as np
import tensorflow as tf

from stochnet_v2.dynamic_classes.op_registry import OP_REGISTRY
from stochnet_v2.dynamic_classes.op_registry import dense


PRIMITIVES = [
    'dense',
    'skip_connect',
    'none',
    'relu',
]

null_scope = tf.compat.v1.VariableScope("")


def l2_regularizer(x):
    return 0.01 * tf.compat.v1.nn.l2_loss(x)


def mixed_op(x, index, expand, expansion_coeff):

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

    outputs = []

    with tf.compat.v1.variable_scope(f"mixed_op_{2 if expand else 1}_{index}"):

        alphas = tf.nn.softmax(alphas)

        for idx, primitive in enumerate(PRIMITIVES):

            out = OP_REGISTRY[primitive](x, expansion_coeff)
            mask = [idx == i for i in range(len(PRIMITIVES))]
            mask = tf.compat.v1.constant(mask, tf.bool)
            alpha = tf.compat.v1.boolean_mask(alphas, mask)

            outputs.append(alpha * out)

        out = tf.compat.v1.add_n(outputs)

    return out


def cell(
        s0,
        s1,
        cell_size,
        expand,
        expand_prev,
        expansion_multiplier,
        cell_index,
):
    print(
        f"{'expand' if expand else 'normal'} cell, "
        f"{'expand' if expand_prev else 'normal'}_prev cell"
    )
    print(f"Inputs: s0: {s0.shape}, s1: {s1.shape}")

    with tf.compat.v1.variable_scope(f"{'expand' if expand else 'normal'}_cell_{cell_index}"):

        with tf.variable_scope("state_0"):
            if expand_prev:
                s0 = dense(s0, expansion_multiplier)
            else:
                s0 = dense(s0, 1)

        with tf.variable_scope("state_1"):
            s1 = dense(s1, 1)

        print(f"State: s0: {s0.shape}, s1: {s1.shape}")

        state = [s0, s1]
        offset = 0

        for i in range(cell_size):
            tmp = []
            for j in range(i + 2):
                expansion_coeff = expansion_multiplier if expand and j < 2 else 1
                mix = mixed_op(state[j], offset + j, expand, expansion_coeff)
                print(i, j, mix.shape)
                tmp.append(mix)

            offset += len(state)
            with tf.variable_scope(f"state_{i+2}"):
                new_state = tf.add_n(tmp)
            state.append(new_state)

        print([t.shape.as_list() for t in state])

        out = tf.concat(state[-1:], axis=-1)

    return out


def body(x, n_cells=4, cell_size=4, expansion_multiplier=4):
    out_dim = x.shape.as_list()[-1]
    s0 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    s1 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    expand_prev = False

    for n in range(n_cells):
        if n in [n_cells // 3, 2 * n_cells // 3]:
            expand = True
        else:
            expand = False

        s0, s1 = s1, cell(s0, s1, cell_size, expand, expand_prev, expansion_multiplier, n)
        expand_prev = expand

    return s1
