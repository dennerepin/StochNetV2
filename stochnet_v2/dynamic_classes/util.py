import tensorflow as tf


initializer = tf.initializers.glorot_normal
# initializer = tf.compat.v1.initializers.variance_scaling(mode='fan_out', distribution="truncated_normal")
# initializer = None


def expand_cell(n, n_cells):
    if n_cells >= 4:
        if n in [n_cells // 3, 2 * n_cells // 3]:
            expand = True
        else:
            expand = False

    elif n_cells == 3:
        if n in [0, 1]:
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


def expand_identity(x, expansion_coeff):
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


def expand_element_wise(
        x,
        expansion_coeff,
        kernel_initializer=tf.compat.v1.initializers.glorot_normal,
        kernel_regularizer=l2_regularizer,
        bias_initializer=tf.compat.v1.initializers.zeros,
        bias_regularizer=l2_regularizer,
):
    with tf.variable_scope("ElementWise"):

        if expansion_coeff > 1:
            x = expand_identity(x, expansion_coeff)

        with tf.compat.v1.variable_scope("kernel"):
            kernel = tf.compat.v1.get_variable(
                name=f"kernel",
                shape=[1] + x.shape.as_list()[1:],
                initializer=kernel_initializer,
                trainable=True,
            )
            if kernel_regularizer:
                kernel_reg_loss = kernel_regularizer(kernel)
                tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, kernel_reg_loss)

        with tf.compat.v1.variable_scope("bias"):
            bias = tf.compat.v1.get_variable(
                name=f"bias",
                shape=x.shape.as_list()[-1],
                initializer=bias_initializer,
                trainable=True,
            )
            if bias_regularizer:
                bias_reg_loss = l2_regularizer(bias)
                tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, bias_reg_loss)

        x = tf.compat.v1.multiply(x, kernel) + bias

    return x


def dense(x, expansion_coeff):
    n_units = x.shape.as_list()[-1] * expansion_coeff
    return tf.compat.v1.layers.Dense(n_units, kernel_initializer=initializer)(x)


def relu_dense_bn(x, expansion_coeff):
    n_units = x.shape.as_list()[-1] * expansion_coeff
    with tf.variable_scope("ReluDenseBN"):
        x = tf.compat.v1.nn.relu(x)
        x = tf.compat.v1.layers.Dense(n_units, kernel_initializer=initializer)(x)
        x = tf.compat.v1.layers.BatchNormalization()(x)
    return x


def bn_dense_relu(x, expansion_coeff):
    n_units = x.shape.as_list()[-1] * expansion_coeff
    with tf.variable_scope("BNDenseRelu"):
        x = tf.compat.v1.layers.BatchNormalization()(x)
        x = tf.compat.v1.layers.Dense(n_units, kernel_initializer=initializer)(x)
        x = tf.compat.v1.nn.relu(x)
    return x


def dense_relu(x, expansion_coeff):
    n_units = x.shape.as_list()[-1] * expansion_coeff
    with tf.variable_scope("DenseRelu"):
        x = tf.compat.v1.layers.Dense(n_units, kernel_initializer=initializer)(x)
        x = tf.compat.v1.nn.relu(x)
    return x
