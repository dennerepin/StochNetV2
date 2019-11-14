import tensorflow as tf


def comma_separated_string_to_integer_list(s):
    return [int(i) for i in s.split(",") if i]


def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    return cast_x


def layer_norm(x, filters=None, epsilon=1e-6):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.shape.as_list()[-1]
    with tf.variable_scope("layer_norm"):
        scale = tf.get_variable("layer_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable("layer_norm_bias", [filters], initializer=tf.zeros_initializer())
        epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(
            tf.math.squared_difference(x, mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        output = norm_x * scale + bias
        return output


def noam_norm(x, epsilon=1.0, name=None):
    """One version of layer normalization."""
    with tf.name_scope(name, default_name="noam_norm", values=[x]):
        shape = x.get_shape()
        ndims = len(shape)
        return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon)
                * tf.sqrt(tf.cast(shape[-1], tf.float32)))


def l2_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
    """Layer normalization with l2 norm."""
    if filters is None:
        filters = x.shape.as_list()[-1]
    with tf.variable_scope(name, default_name="l2_norm", values=[x], reuse=reuse):
        scale = tf.compat.v1.get_variable(
            "l2_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.compat.v1.get_variable(
            "l2_norm_bias", [filters], initializer=tf.zeros_initializer())
        epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        l2norm = tf.reduce_sum(
            tf.math.squared_difference(x, mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(l2norm + epsilon)
        return norm_x * scale + bias


def apply_norm(x, norm_type, depth=None, epsilon=None):
    """Apply Normalization."""
    # TODO: add l1 norm
    if norm_type == "layer":
        return layer_norm(x, filters=depth, epsilon=epsilon)
    if norm_type == "batch":
        return tf.compat.v1.layers.BatchNormalization(epsilon=epsilon)(x)
    if norm_type == "noam":
        return noam_norm(x, epsilon)
    if norm_type == "l2":
        return l2_norm(x, filters=depth, epsilon=epsilon)
    if norm_type == "none":
        return x
    raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch', 'none'.")


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
    """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.

    Instead of specifying noise_shape, this function takes broadcast_dims -
    a list of dimension numbers in which noise_shape should be 1.  The random
    keep/drop tensor has dimensionality 1 along these dimensions.

    Args:
      x: a floating point tensor.
      keep_prob: A scalar Tensor with the same type as x.
        The probability that each element is kept.
      broadcast_dims: an optional list of integers
        the dimensions along which to broadcast the keep/drop flags.
      **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".

    Returns:
      Tensor of the same shape as x.
    """
    assert "noise_shape" not in kwargs
    if broadcast_dims:
        shape = tf.shape(x)
        ndims = len(x.get_shape())
        # Allow dimensions like "-1" as well.
        broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
        kwargs["noise_shape"] = [
            1 if i in broadcast_dims else shape[i] for i in range(ndims)
        ]
    return tf.nn.dropout(x, keep_prob, **kwargs)


def layer_prepostprocess(
        previous_value,
        x,
        sequence,
        dropout_rate,
        norm_type,
        depth,
        epsilon,
        default_name,
        name=None,
        dropout_broadcast_dims=None,
):
    """Apply a sequence of functions to the input or output of a layer.

    The sequence is specified as a string which may contain the following
    characters:
      a: add previous_value
      n: apply normalization
      d: apply dropout
      z: zero add

    For example, if sequence=="dna", then the output is
      previous_value + normalize(dropout(x))

    Args:
      previous_value: A Tensor, to be added as a residual connection ('a')
      x: A Tensor to be transformed.
      sequence: a string.
      dropout_rate: a float
      norm_type: a string (see apply_norm())
      depth: an integer (size of last dimension of x).
      epsilon: a float (parameter for normalization)
      default_name: a string
      name: a string
      dropout_broadcast_dims:  an optional list of integers less than 3
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.
    Returns:
      a Tensor
    """
    with tf.variable_scope(name, default_name=default_name):
        if sequence == "none":
            return x
        for c in sequence:
            if c == "a":
                x += previous_value
            elif c == "n":
                x = apply_norm(
                    x, norm_type, depth, epsilon)
            else:
                assert c == "d", ("Unknown sequence step %s" % c)
                x = dropout_with_broadcast_dims(
                    x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
        return x


def layer_preprocess(
        layer_input,
        sequence,
        dropout,
        norm_type,
        norm_epsilon=1e-4,
        dropout_broadcast_dims="",
):
    """Apply layer preprocessing."""
    assert "a" not in sequence, (
        "No residual connections allowed in hparams.layer_preprocess_sequence")
    assert "z" not in sequence, (
        "No residual connections allowed in hparams.layer_preprocess_sequence")
    return layer_prepostprocess(
        None,
        layer_input,
        sequence=sequence,
        dropout_rate=dropout,
        norm_type=norm_type,
        depth=None,
        epsilon=norm_epsilon,
        dropout_broadcast_dims=comma_separated_string_to_integer_list(dropout_broadcast_dims),
        default_name="layer_preprocess"
    )


def layer_postprocess(
        layer_input,
        layer_output,
        sequence,
        dropout,
        norm_type,
        norm_epsilon=1e-4,
        dropout_broadcast_dims="",
):
    """Apply layer postprocessing."""
    return layer_prepostprocess(
        layer_input,
        layer_output,
        sequence=sequence,
        dropout_rate=dropout,
        norm_type=norm_type,
        depth=None,
        epsilon=norm_epsilon,
        dropout_broadcast_dims=comma_separated_string_to_integer_list(dropout_broadcast_dims),
        default_name="layer_postprocess")
