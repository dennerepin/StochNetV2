import numpy as np
import tensorflow as tf

from stochnet_v2.dynamic_classes.op_registry import OP_REGISTRY

PRIMITIVES = [
    'dense',
    'relu',
    'skip_connect',
    'none',
]

null_scope = tf.VariableScope("")
l2_regularizer = lambda var: 0.01 * tf.nn.l2_loss(var)


def expand_dense(x):
    out_dim = x.shape.as_list()[-1] * 4
    return tf.compat.v1.layers.Dense(out_dim)(x)


def regular_dense(x):
    out_dim = x.shape.as_list()[-1]
    return tf.compat.v1.layers.Dense(out_dim)(x)


def mixed_op(x, index, expand):

    with tf.compat.v1.variable_scope(null_scope):
        alphas = tf.get_variable(
            name=f"alpha_{2 if expand else 1}_{index}",
            shape=[len(PRIMITIVES)],
            initializer=tf.keras.initializers.random_normal,
            trainable=False,
            collections=['architecture_variables'],
        )
        alphas_regularizer = l2_regularizer(alphas)
        tf.add_to_collection('architecture_regularization_losses', alphas_regularizer)

    alphas = tf.nn.softmax(alphas)

    outputs = []
    for idx, primitive in enumerate(PRIMITIVES):

        out = OP_REGISTRY[primitive](x)
        mask = [idx == i for i in range(len(PRIMITIVES))]
        mask = tf.constant(mask, tf.bool)
        alpha = tf.boolean_mask(alphas, mask)

        outputs.append(alpha * out)

    return tf.add_n(outputs)


def cell(
        s0,
        s1,
        cells_num,
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

    for i in range(cells_num):
        tmp = []
        for j in range(i + 2):
            tmp.append(mixed_op(state[j], offset + j, expand))

        offset += len(state)
        state.append(tf.add_n(tmp))

    # out = tf.concat(state[-multiplier:], axis=-1)

    return state


# class MixedLayer:
#
#     def __init__(
#             self,
#             n_units,
#     ):
#         self._n_units = n_units
#         self._n_branches = 4
#         self._branch_vars = list()
#         self._build()
#
#     def _build(self, alphas=None):
#
#         if alphas is None:
#             alphas = np.ones(shape=(1, self._n_branches), dtype=np.float32)
#
#         if not alphas.shape[-1] == self._n_branches:
#             raise ValueError("len(alphas) is not equal to self.n_branches")
#
#         self._alphas = tf.compat.v1.Variable(
#             initial_value=alphas,
#             trainable=False,
#             collections=["architecture_variables"]
#         )
#         gates = tf.random.categorical(logits=self._alphas, num_samples=1)
#         gates = tf.one_hot(gates, self._n_branches)
#         gates = tf.squeeze(gates, 0)
#         gates = tf.transpose(gates)
#         self._gates = gates
#
#     def __call__(
#             self,
#             inputs,
#             *args,
#             **kwargs,
#     ):
#
#         out_0, vars_0 = dense_layer(inputs, n_units=self._n_units)
#         # out_0 = self._gates[0] * out_0
#         self._branch_vars.append(vars_0)
#
#         out_1, vars_1 = activation_layer(inputs, activation_kind='relu')
#         # out_1 = self._gates[1] * out_1
#         self._branch_vars.append(vars_1)
#
#         out_2, vars_2 = tf.identity(inputs), []
#         # out_2 = self._gates[2] * out_2
#         self._branch_vars.append(vars_2)
#
#         out_3, vars_3 = dense_layer(inputs, n_units=self._n_units)
#         out_3 = inputs + out_3
#         # out_3 = self._gates[3] * out_3
#         self._branch_vars.append(vars_3)
#
#         result = tf.stack([out_0, out_1, out_2, out_3], axis=0)
#         result = tf.multiply(self._gates, result)
#         result = tf.reduce_sum(result, axis=0)
#
#         return result


