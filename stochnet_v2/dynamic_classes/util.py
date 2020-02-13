import tensorflow as tf
from stochnet_v2.utils.util import apply_regularization
from stochnet_v2.utils.layer_prepostprocess import layer_preprocess
from stochnet_v2.utils.layer_prepostprocess import layer_postprocess
from stochnet_v2.utils.registry import ACTIVATIONS_REGISTRY

initializer = tf.initializers.glorot_normal
# initializer = tf.compat.v1.initializers.variance_scaling(mode='fan_out', distribution="truncated_normal")
# initializer = None


def preprocess(layer_input):
    return layer_preprocess(layer_input, 'none', 0.1, 'l2', 1e-4)  # 'none'


def postprocess(layer_output):
    return layer_postprocess(0., layer_output, 'n', 0.1, 'l2', 1e-4)  # 'n'


# def postprocess_residual(layer_input, layer_output):
#     return layer_postprocess(layer_input, layer_output, 'n', 0.1, 'layer', 1e-4)  # 'an'


def cell_is_expanding(n, n_cells):
    if n_cells >= 4:
        if n in [n_cells // 3, 2 * n_cells // 3]:
            expand = True
        else:
            expand = False

    elif n_cells == 3:
        if n in [0]:  # TODO: if n in [0, 1]:
            expand = True
        else:
            expand = False

    elif n_cells <= 2:
        if n == 0:
            expand = True
        else:
            expand = False
    else:
        raise ValueError(f"Incorrect `n_cells` parameter: {n_cells}")
    return expand


def l2_regularizer(x, scale=0.01):
    return scale * tf.compat.v1.nn.l2_loss(x)


def l1_regularizer(x, scale=0.01):
    return scale * tf.reduce_sum(tf.abs(x))


def _expand_identity(x, expansion_coeff, **kwargs):
    if expansion_coeff == 1:
        return tf.compat.v1.identity(x)
    with tf.compat.v1.variable_scope('expansion_identity'):
        n_dims = x.shape.ndims
        x_shape = x.shape.as_list()
        final_shape = [i for i in x_shape[:-1]] + [x_shape[-1] * expansion_coeff]
        final_shape = [i or -1 for i in final_shape[::-1]]
        x = tf.transpose(x, [i for i in range(n_dims-1, -1, -1)])
        x = tf.tile(x, [1 for _ in range(n_dims - 1)] + [expansion_coeff])
        x = tf.reshape(x, final_shape)
        x = tf.transpose(x, [i for i in range(n_dims-1, -1, -1)])
    return x


def _expand_element_wise(
        x,
        expansion_coeff,
        kernel_initializer=initializer,
        kernel_constraint=None,
        kernel_regularizer=None,
        bias_initializer=tf.compat.v1.initializers.zeros,
        bias_constraint=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
):
    with tf.variable_scope("ElementWise"):

        if expansion_coeff > 1:
            x = _expand_identity(x, expansion_coeff)

        with tf.compat.v1.variable_scope("kernel"):
            kernel = tf.compat.v1.get_variable(
                name=f"kernel",
                shape=[1] + x.shape.as_list()[1:],
                initializer=kernel_initializer,
                constraint=kernel_constraint,
                regularizer=kernel_regularizer,
                trainable=True,
            )

        with tf.compat.v1.variable_scope("bias"):
            bias = tf.compat.v1.get_variable(
                name=f"bias",
                shape=x.shape.as_list()[-1],
                initializer=bias_initializer,
                constraint=bias_constraint,
                regularizer=bias_regularizer,
                trainable=True,
            )
        residual = x
        x = preprocess(x)

        x = tf.compat.v1.multiply(x, kernel) + bias
        if activity_regularizer:
            apply_regularization(activity_regularizer, x)

        # x = postprocess_residual(residual, x)
        x = postprocess(x)

    return x


def _simple_dense(
        x,
        expansion_coeff,
        kernel_initializer=initializer,
        kernel_constraint=None,
        kernel_regularizer=None,
        bias_initializer=tf.compat.v1.initializers.zeros,
        bias_constraint=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
):
    n_units = int(x.shape.as_list()[-1] * expansion_coeff)

    with tf.variable_scope("Dense"):
        x = preprocess(x)
        x = tf.compat.v1.layers.Dense(
            n_units,
            kernel_initializer=kernel_initializer,
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            bias_constraint=bias_constraint,
            bias_regularizer=bias_regularizer,
        )(x)
        if activity_regularizer:
            apply_regularization(activity_regularizer, x)

        x = postprocess(x)

    return x


def _activated_dense(
        x,
        expansion_coeff,
        activation_type,
        kernel_initializer=initializer,
        kernel_constraint=None,
        kernel_regularizer=None,
        bias_initializer=tf.compat.v1.initializers.zeros,
        bias_constraint=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
):
    n_units = int(x.shape.as_list()[-1] * expansion_coeff)

    with tf.variable_scope("ActivatedDense"):
        x = preprocess(x)
        x = tf.compat.v1.layers.Dense(
            n_units,
            kernel_initializer=kernel_initializer,
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            bias_constraint=bias_constraint,
            bias_regularizer=bias_regularizer,
        )(x)
        if activity_regularizer:
            apply_regularization(activity_regularizer, x)

        activation_fn = ACTIVATIONS_REGISTRY[activation_type]
        x = activation_fn(x)
        x = postprocess(x)

    return x


def _gated_linear_unit(
        x,
        expansion_coeff,
        kernel_initializer=initializer,
        kernel_constraint=None,
        kernel_regularizer=None,
        bias_initializer=tf.compat.v1.initializers.zeros,
        bias_constraint=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
):
    n_units = int(x.shape.as_list()[-1] * expansion_coeff)

    with tf.variable_scope("GatedLinearUnit"):
        if expansion_coeff > 1:
            residual = _expand_identity(x, expansion_coeff)
        else:
            residual = x
        x = preprocess(x)

        values = tf.compat.v1.layers.Dense(
            n_units,
            kernel_initializer=kernel_initializer,
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            bias_constraint=bias_constraint,
            bias_regularizer=bias_regularizer,
        )(x)
        gates = tf.compat.v1.layers.Dense(
            n_units,
            kernel_initializer=kernel_initializer,
            kernel_constraint=kernel_constraint,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            bias_constraint=bias_constraint,
            bias_regularizer=bias_regularizer,
        )(x)
        gates = tf.nn.sigmoid(gates)
        x = values * gates

        # x = postprocess_residual(residual, x)
        x = postprocess(x)

    return x
