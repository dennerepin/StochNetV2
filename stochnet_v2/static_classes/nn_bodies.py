import numpy as np
import tensorflow as tf

maxnorm = tf.keras.constraints.MaxNorm
Dense = tf.keras.layers.Dense
LeakyReLU = tf.keras.layers.LeakyReLU
Add = tf.keras.layers.Add


def dummy_body(input_tensor):
    shape = input_tensor.shape.as_list()
    x = tf.keras.layers.Reshape((np.prod(shape[1:]),))(input_tensor)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    return x


def body_a(x, hidden_size=1024):
    shape = x.shape.as_list()
    x = tf.keras.layers.Reshape((np.prod(shape[1:]),))(x)

    h1 = tf.keras.layers.Dense(
        hidden_size,
        # kernel_constraint=maxnorm(3),
    )(x)
    #h1 = tf.keras.layers.LeakyReLU(alpha=0.1)(h1)
    #h1 = tf.keras.layers.ReLU()(h1)

    h2 = tf.keras.layers.Dense(
        hidden_size,
        # kernel_constraint=maxnorm(3),
    )(h1)

    #h2 = tf.keras.layers.LeakyReLU(alpha=0.1)(h2)
    h2 = tf.keras.layers.Add()([h1, h2])

    # h2 = tf.keras.layers.ReLU()(h2)

    h3 = tf.keras.layers.Dense(
        hidden_size,
        # kernel_constraint=maxnorm(3),
    )(h2)

    # h3 = tf.keras.layers.LeakyReLU(alpha=0.1)(h3)
    h3 = tf.keras.layers.Add()([h2, h3])

    # h3 = tf.keras.layers.ReLU()(h3)

    # h4 = tf.keras.layers.Dense(
    #     hidden_size,
    #     kernel_constraint=maxnorm(3)
    # )(h3)
    # h4 = tf.keras.layers.LeakyReLU(alpha=0.1)(h4)
    # h4 = tf.keras.layers.Add()([h3, h4])

    # h4 = tf.keras.layers.Dense(hidden_size*4)(h3)

    return h3


def body_good(x):
    shape = x.shape.as_list()
    x = tf.keras.layers.Reshape((np.prod(shape[1:]),))(x)

    h1 = Dense(512, kernel_constraint=maxnorm(3))(x)
    h1 = LeakyReLU(alpha=0.1)(h1)

    h2 = Dense(512, kernel_constraint=maxnorm(3))(h1)
    h2 = LeakyReLU(alpha=0.1)(h2)
    h2 = Add()([h1, h2])

    h3 = Dense(512, kernel_constraint=maxnorm(3))(h2)
    h3 = LeakyReLU(alpha=0.1)(h3)
    h3 = Add()([h2, h3])

    h4 = Dense(512, kernel_constraint=maxnorm(3))(h3)
    h4 = LeakyReLU(alpha=0.1)(h4)
    h4 = Add()([h3, h4])
    return h4

