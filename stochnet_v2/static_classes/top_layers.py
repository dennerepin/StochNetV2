import abc
import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from stochnet_v2.static_classes.random_variables import Categorical
from stochnet_v2.static_classes.random_variables import MultivariateNormalDiag
from stochnet_v2.static_classes.random_variables import MultivariateNormalTriL
from stochnet_v2.static_classes.random_variables import MultivariateLogNormalTriL
from stochnet_v2.static_classes.random_variables import Mixture
from stochnet_v2.utils.errors import ShapeError
from stochnet_v2.utils.errors import DimensionError
from stochnet_v2.utils.registry import Registry
from stochnet_v2.utils.util import apply_regularization

tfd = tfp.distributions
Dense = tf.compat.v1.layers.Dense
initializer = tf.initializers.glorot_normal
# initializer = tf.compat.v1.initializers.variance_scaling(mode='fan_out', distribution="truncated_normal")
# initializer = None


_DIAG_MIN = 0.01
_DIAG_MAX = np.inf
_SUB_DIAG_MAX = np.inf
# _DIAG_MAX = 1.0
# _SUB_DIAG_MAX = 1.0


MIXTURE_COMPONENTS_REGISTRY = Registry(name='MixtureComponentsDescriptionsRegistry')


LOGGER = logging.getLogger('static_classes.top_layers')


def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.math.log(tf.math.expm1(x))


def softplus_activation(x):
    """Softplus activation"""
    LOGGER.debug("Using softplus activation for diagonal")
    return tf.nn.softplus(x + _softplus_inverse(1.0))


def nn_elu_activation(x):
    """Computes the Non-Negative Exponential Linear Unit"""
    LOGGER.debug("Using non-nnegative elu activation for diagonal")
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(x))


class RandomVariableOutputLayer(abc.ABC):

    def __init__(self):
        self._pred_placeholder = None
        self._random_variable = None
        self._sample_shape_placeholder = None
        self._sample_tensor = None
        self._sampling_graph = None

        self._sample_space_dimension = None
        self._number_of_output_neurons = None

        self.name_scope = self.__class__.__name__

    @abc.abstractmethod
    def add_layer_on_top(self, base):
        """Adds a layer on top of an existing neural network model which allows
        to learn those parameters which are needed to instantiate a random
        variable of the family indicated in the class name.
        The top layer output tensor has the following shape:
        [batch_size, self.number_of_output_neurons]
        """

    @abc.abstractmethod
    def get_random_variable(self, nn_prediction):
        """Given the values of a tensor produced by a neural network with a layer
        on top of the form of the one provided by the add_layer_on_top method,
        it returns an instance of the corresponding tensor random variable class
        initialized using the parameters provided by the neural network output.
        Additional checks might be needed for certain families of random variables.
        """

    @property
    @abc.abstractmethod
    def number_of_output_neurons(self):
        pass

    def check_input_shape(self, nn_prediction):
        """The method check that nn_prediction has the following shape:
        [batch_size, number_of_output_neurons]
        """
        shape = list(nn_prediction.shape)
        if len(shape) != 2 or shape[1] != self.number_of_output_neurons:
            raise ShapeError(
                f"The neural network predictions passed as input for"
                f" {self.__class__.__name__} should be of shape:\n"
                f"[batch_size, {self.number_of_output_neurons}], got shape:\n"
                f"{shape}"
            )

    def get_description(self, nn_prediction):
        """Return a string containing a description of the random variables initialized
        using the parameters in nn_prediction"""
        # TODO: fixme: get description w/o re-creating graph for random variable?
        random_variable = self.get_random_variable(nn_prediction)
        description = random_variable.get_description()
        return description

    def build_sampling_graph(self, graph=None):

        self._sampling_graph = graph or tf.compat.v1.Graph()
        with self._sampling_graph.as_default():
            with tf.variable_scope(self.name_scope + '/sample'):
                self._pred_placeholder = tf.compat.v1.placeholder(
                    dtype=tf.float32,
                    shape=(None, self.number_of_output_neurons),
                    name='pred_placeholder',
                )
                self._random_variable = self.get_random_variable(self._pred_placeholder)
                self._sample_shape_placeholder = tf.compat.v1.placeholder(
                    tf.int32, None, name='sample_shape_placeholder'
                )
                self._sample_tensor = self._random_variable.sample(self._sample_shape_placeholder)

    def sample(self, nn_prediction_np, sample_shape=()):

        if self._sampling_graph is None:
            self.build_sampling_graph()

        with tf.Session(graph=self._sampling_graph) as sess:
            res = sess.run(
                self._sample_tensor,
                feed_dict={
                    self._pred_placeholder: nn_prediction_np,
                    self._sample_shape_placeholder: sample_shape
                }
            )
        return res

    @property
    def pred_placeholder(self):
        return self._pred_placeholder

    @property
    def sample_shape_placeholder(self):
        return self._sample_shape_placeholder

    @property
    def sample_tensor(self):
        return self._sample_tensor

    @property
    def description_graphkeys(self):
        return self._random_variable.description_graphkeys


@MIXTURE_COMPONENTS_REGISTRY.register('categorical')
class CategoricalOutputLayer(RandomVariableOutputLayer):

    def __init__(
            self,
            number_of_classes,
            hidden_size=None,
            activation=None,
            coeff_regularizer=None,
            kernel_constraint=None,
            kernel_regularizer=None,
            bias_constraint=None,
            bias_regularizer=None,
    ):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.random_variable = None
        self.hidden_size = hidden_size
        self._activation_fn = activation
        self._coeff_regularizer = coeff_regularizer

        self._layer_params = {
            'kernel_constraint': kernel_constraint,
            'kernel_regularizer': kernel_regularizer,
            'bias_constraint': bias_constraint,
            'bias_regularizer': bias_regularizer,
            'kernel_initializer': initializer,
        }

    @property
    def number_of_classes(self):
        return self._number_of_classes

    @property
    def number_of_output_neurons(self):
        return self._number_of_output_neurons

    @number_of_classes.setter
    def number_of_classes(self, new_number_of_classes):
        if new_number_of_classes > 0:
            self._number_of_classes = new_number_of_classes
            self._number_of_output_neurons = new_number_of_classes
        else:
            raise ValueError(
                "Number of classes for Categorical random variable "
                "should be at least 1."
            )

    def add_layer_on_top(self, base):

        with tf.variable_scope(self.name_scope):

            if self.hidden_size:
                with tf.variable_scope('residual'):
                    base = Dense(self.hidden_size, **self._layer_params)(base)
                    base = self._activation_fn(base)
                    base1 = Dense(self.hidden_size, **self._layer_params)(base)
                    base1 = self._activation_fn(base1)
                    base = tf.add(base, base1)

            logits = Dense(
                self.number_of_output_neurons,
                activation=None,
                name='logits',
                **self._layer_params
            )(base)

            if self._coeff_regularizer:
                apply_regularization(self._coeff_regularizer, logits)

            return logits

    def get_random_variable(self, nn_prediction_tensor):
        self.check_input_shape(nn_prediction_tensor)
        with tf.variable_scope(self.name_scope):
            with tf.variable_scope('random_variable'):
                return Categorical(nn_prediction_tensor)

    @staticmethod
    def loss_function(y_true, y_pred):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return loss

    @staticmethod
    def log_likelihood_function(y_true, y_pred):
        log_likelihood = -tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        return log_likelihood


@MIXTURE_COMPONENTS_REGISTRY.register('normal_diag')
class MultivariateNormalDiagOutputLayer(RandomVariableOutputLayer):

    def __init__(
            self,
            sample_space_dimension,
            hidden_size=None,
            activation=None,
            mu_regularizer=None,
            diag_regularizer=None,
            kernel_constraint=None,
            kernel_regularizer=None,
            bias_constraint=None,
            bias_regularizer=None,
    ):
        super().__init__()
        self.sample_space_dimension = sample_space_dimension
        self.hidden_size = hidden_size
        self._activation_fn = activation
        self._mu_regularizer = mu_regularizer
        self._diag_regularizer = diag_regularizer

        self._mu_layer_params = {
            'kernel_constraint': kernel_constraint,
            'kernel_regularizer': kernel_regularizer,
            'bias_constraint': bias_constraint,
            'bias_regularizer': bias_regularizer,
            'kernel_initializer': initializer,
        }
        self._diag_layer_params = {
            'kernel_constraint': kernel_constraint,
            'kernel_regularizer': kernel_regularizer,
            'bias_constraint': bias_constraint,
            'bias_regularizer': bias_regularizer,
            'kernel_initializer': initializer,
        }

    @property
    def sample_space_dimension(self):
        return self._sample_space_dimension

    @sample_space_dimension.setter
    def sample_space_dimension(self, new_sample_space_dimension):
        if new_sample_space_dimension > 1:
            self._sample_space_dimension = new_sample_space_dimension
            self._number_of_output_neurons = 2 * self._sample_space_dimension
        else:
            raise ValueError(
                "The sample space dimension for MultivariateNormalDiag random variable "
                "should be at least 2."
            )

    @property
    def number_of_output_neurons(self):
        return self._number_of_output_neurons

    def add_layer_on_top(self, base):

        with tf.name_scope(self.name_scope) as scope:
            with tf.variable_scope(scope):

                s = base.shape.as_list()[-1]

                mu, diag = tf.split(base, num_or_size_splits=[s//2, s - s//2], axis=-1)

                if self.hidden_size:

                    with tf.variable_scope('residual'):

                        with tf.variable_scope('mu'):
                            mu = Dense(self.hidden_size, **self._mu_layer_params)(mu)
                            mu = self._activation_fn(mu)
                            mu1 = Dense(self.hidden_size, **self._mu_layer_params)(mu)
                            mu1 = self._activation_fn(mu1)
                            mu = tf.add(mu, mu1)

                        with tf.variable_scope('diag'):
                            diag = Dense(self.hidden_size, **self._diag_layer_params)(diag)
                            diag = self._activation_fn(diag)
                            diag1 = Dense(self.hidden_size, **self._diag_layer_params)(diag)
                            diag1 = self._activation_fn(diag1)
                            diag = tf.add(diag, diag1)

                mu = Dense(
                    self._sample_space_dimension,
                    activation=None,
                    name='mu',
                    **self._mu_layer_params,
                )(mu)

                diag = Dense(
                    self._sample_space_dimension,
                    activation=nn_elu_activation,  # or softplus_activation
                    name='diag',
                    **self._diag_layer_params,
                )(diag)

                if self._mu_regularizer:
                    apply_regularization(self._mu_regularizer, mu)

                if self._diag_regularizer:
                    apply_regularization(self._diag_regularizer, diag)

                diag = tf.compat.v1.clip_by_value(diag, _DIAG_MIN, _DIAG_MAX)

                return tf.concat([mu, diag], axis=-1)

    def get_random_variable(self, nn_prediction_tensor):
        self.check_input_shape(nn_prediction_tensor)

        with tf.variable_scope(self.name_scope):
            with tf.variable_scope('random_variable'):
                mu = tf.slice(
                    nn_prediction_tensor,
                    [0, 0],
                    [-1, self._sample_space_dimension],
                    name='mu',
                )
                diag = tf.slice(
                    nn_prediction_tensor,
                    [0, self._sample_space_dimension],
                    [-1, self._sample_space_dimension],
                    name='diag',
                )
                return MultivariateNormalDiag(mu, diag)

    def loss_function(self, y_true, y_pred):
        loss = - self.log_likelihood(y_true, y_pred)
        # loss = tf.math.reduce_mean(loss)
        loss = tf.reshape(loss, [-1])
        return loss

    def log_likelihood(self, y_true, y_pred):
        return self.get_random_variable(y_pred).log_prob(y_true)


@MIXTURE_COMPONENTS_REGISTRY.register('normal_tril')
class MultivariateNormalTriLOutputLayer(RandomVariableOutputLayer):

    random_variable_class = MultivariateNormalTriL

    def __init__(
            self,
            sample_space_dimension,
            hidden_size=None,
            activation=None,
            mu_regularizer=None,
            diag_regularizer=None,
            sub_diag_regularizer=None,
            kernel_constraint=None,
            kernel_regularizer=None,
            bias_constraint=None,
            bias_regularizer=None,
    ):
        super().__init__()
        self.sample_space_dimension = sample_space_dimension
        self.hidden_size = hidden_size
        self._activation_fn = activation
        self._mu_regularizer = mu_regularizer
        self._diag_regularizer = diag_regularizer
        self._sub_diag_regularizer = sub_diag_regularizer

        self._mu_layer_params = {
            'kernel_constraint': kernel_constraint,
            'kernel_regularizer': kernel_regularizer,
            'bias_constraint': bias_constraint,
            'bias_regularizer': bias_regularizer,
            'kernel_initializer': initializer,
        }
        self._diag_layer_params = {
            'kernel_constraint': kernel_constraint,
            'kernel_regularizer': kernel_regularizer,
            'bias_constraint': bias_constraint,
            'bias_regularizer': bias_regularizer,
            'kernel_initializer': initializer,
        }
        self._sub_diag_layer_params = {
            'kernel_constraint': kernel_constraint,
            'kernel_regularizer': kernel_regularizer,
            'bias_constraint': bias_constraint,
            'bias_regularizer': bias_regularizer,
            'kernel_initializer': initializer,
        }

    @property
    def sample_space_dimension(self):
        return self._sample_space_dimension

    @sample_space_dimension.setter
    def sample_space_dimension(self, new_sample_space_dimension):
        if new_sample_space_dimension > 1:
            self._sample_space_dimension = new_sample_space_dimension
            self._number_of_sub_diag_entries = self._sample_space_dimension * (self._sample_space_dimension - 1) // 2
            self._number_of_output_neurons = 2 * self._sample_space_dimension + self._number_of_sub_diag_entries
        else:
            raise ValueError(
                "The sample space dimension for MultivariateNormalTriL random variable "
                "should be at least 2."
            )

    @property
    def number_of_output_neurons(self):
        return self._number_of_output_neurons

    def add_layer_on_top(self, base):

        with tf.name_scope(self.name_scope) as scope:
            with tf.variable_scope(scope):

                shape = base.shape.as_list()
                s = shape[-1] // 3

                mu, diag, sub_diag = tf.split(base, num_or_size_splits=[s, s, shape[-1] - 2 * s], axis=-1)

                if self.hidden_size:

                    with tf.variable_scope('residual'):

                        with tf.variable_scope('mu'):
                            mu = Dense(self.hidden_size, **self._mu_layer_params)(mu)
                            mu = self._activation_fn(mu)
                            mu1 = Dense(self.hidden_size, **self._mu_layer_params)(mu)
                            mu1 = self._activation_fn(mu1)
                            mu = tf.add(mu, mu1)

                        with tf.variable_scope('diag'):
                            diag = Dense(self.hidden_size, **self._diag_layer_params)(diag)
                            diag = self._activation_fn(diag)
                            diag1 = Dense(self.hidden_size, **self._diag_layer_params)(diag)
                            diag1 = self._activation_fn(diag1)
                            diag = tf.add(diag, diag1)

                        with tf.variable_scope('sub_diag'):
                            sub_diag = Dense(self.hidden_size, **self._diag_layer_params)(sub_diag)
                            sub_diag = self._activation_fn(sub_diag)
                            sub_diag1 = Dense(self.hidden_size, **self._diag_layer_params)(sub_diag)
                            sub_diag1 = self._activation_fn(sub_diag1)
                            sub_diag = tf.add(sub_diag, sub_diag1)

                mu = Dense(
                    self._sample_space_dimension,
                    activation=None,
                    name='mu',
                    **self._mu_layer_params,
                )(mu)

                diag = Dense(
                    self._sample_space_dimension,
                    activation=nn_elu_activation,   # or softplus_activation
                    name='diag',
                    **self._diag_layer_params,
                )(diag)

                sub_diag = Dense(
                    self._number_of_sub_diag_entries,
                    activation=None,
                    name='sub_diag',
                    **self._sub_diag_layer_params,
                )(sub_diag)

                if self._mu_regularizer:
                    apply_regularization(self._mu_regularizer, mu)

                if self._diag_regularizer:
                    apply_regularization(self._diag_regularizer, diag)

                if self._sub_diag_regularizer:
                    apply_regularization(self._sub_diag_regularizer, sub_diag)

                diag = tf.compat.v1.clip_by_value(diag, _DIAG_MIN, _DIAG_MAX)
                sub_diag = tf.compat.v1.clip_by_value(sub_diag, -_SUB_DIAG_MAX, _SUB_DIAG_MAX)

                return tf.concat([mu, diag, sub_diag], axis=-1)

    def get_random_variable(self, nn_prediction_tensor):

        def _batch_to_tril(flat_diag, flat_sub_diag):
            diag_matr = tf.compat.v1.linalg.diag(flat_diag)
            sub_diag_matr = tfd.fill_triangular(flat_sub_diag)
            sub_diag_matr = tf.pad(sub_diag_matr, paddings=tf.constant([[0, 0], [1, 0], [0, 1]]))
            return diag_matr + sub_diag_matr

        self.check_input_shape(nn_prediction_tensor)

        with tf.variable_scope(self.name_scope):
            with tf.variable_scope('random_variable'):

                mu = tf.slice(
                    nn_prediction_tensor,
                    [0, 0],
                    [-1, self._sample_space_dimension],
                    name='mu',
                )
                diag = tf.slice(
                    nn_prediction_tensor,
                    [0, self._sample_space_dimension],
                    [-1, self._sample_space_dimension],
                    name='diag',
                )
                sub_diag = tf.slice(
                    nn_prediction_tensor,
                    [0, 2 * self._sample_space_dimension],
                    [-1, self._number_of_sub_diag_entries],
                    name='sub_diag',
                )
                with tf.control_dependencies([tf.assert_positive(diag)]):
                    tril = _batch_to_tril(diag, sub_diag)

        return self.random_variable_class(mu, tril)

    def loss_function(self, y_true, y_pred):
        loss = - self.log_likelihood(y_true, y_pred)
        # loss = tf.math.reduce_mean(loss)
        loss = tf.reshape(loss, [-1])
        return loss

    def log_likelihood(self, y_true, y_pred):
        return self.get_random_variable(y_pred).log_prob(y_true)


@MIXTURE_COMPONENTS_REGISTRY.register('log_normal_tril')
class MultivariateLogNormalTriLOutputLayer(MultivariateNormalTriLOutputLayer):

    random_variable_class = MultivariateLogNormalTriL


class MixtureOutputLayer(RandomVariableOutputLayer):

    def __init__(
            self,
            categorical,
            components,
    ):
        super().__init__()
        if len(components) != categorical.number_of_classes:
            raise ValueError(
                f"Number of classes of Categorical is not equal "
                f"to the number of components provided for Mixture."
            )
        self.number_of_components = len(components)
        self.categorical = categorical
        self.components = list(components)
        self.set_sample_space_dimension()
        self.set_number_of_output_neurons()

    @property
    def sample_space_dimension(self):
        return self._sample_space_dimension

    @property
    def number_of_output_neurons(self):
        return self._number_of_output_neurons

    def set_sample_space_dimension(self):
        sample_space_dims = [component.sample_space_dimension for component in self.components]

        if not all(x == sample_space_dims[0] for x in sample_space_dims):
            raise DimensionError(
                f"The random variables which have been passed "
                f"as mixture components sample from spaces with "
                f"different dimensions.\n"
                f"This is the list of sample spaces dimensions:\n"
                f"{sample_space_dims}"
            )

        self._sample_space_dimension = sample_space_dims[0]

    def set_number_of_output_neurons(self):
        self._number_of_output_neurons = self.categorical.number_of_output_neurons
        for component in self.components:
            self._number_of_output_neurons += component.number_of_output_neurons

    def add_layer_on_top(self, base):
        return self._add_layer_on_top_share(base)
        # return self._add_layer_on_top_individual(base)
        # return self._add_layer_on_top_individual_cat(base)

    def _add_layer_on_top_share(self, base):
        # all components onto the same base
        LOGGER.debug("Mixture components share nn outputs")
        LOGGER.debug(f'base shape: {base.shape.as_list()}')
        with tf.variable_scope(self.name_scope):
            categorical_layer = self.categorical.add_layer_on_top(base)
            components_layers = [component.add_layer_on_top(base) for component in self.components]
            mixture_layers = [categorical_layer] + components_layers
            return tf.concat(mixture_layers, axis=-1)

    def _add_layer_on_top_individual(self, base):
        LOGGER.debug("Mixture components use individual slices of nn outputs")
        # individual slice for each component
        n_slices = len(self.components) + 1
        slice_dim = base.shape.as_list()[-1]
        slice_size = slice_dim // n_slices
        cat_slice_size = slice_size + slice_dim % n_slices

        LOGGER.debug(f'base shape: {base.shape.as_list()}')

        components_outputs = []

        with tf.variable_scope(self.name_scope):

            categorical_slice = tf.slice(
                base,
                [0, 0],
                [-1, cat_slice_size],
                name='categorical_slice',
            )
            LOGGER.debug(f'categorical: {self.categorical.__class__.__name__} '
                         f'from 0 for {cat_slice_size} - {categorical_slice.shape.as_list()}')
            categorical_output = self.categorical.add_layer_on_top(categorical_slice)

            for i, component in enumerate(self.components):
                component_slice = tf.slice(
                    base,
                    [0, cat_slice_size + i * slice_size],
                    [-1, slice_size],
                    name=f'component_{i}_slice'
                )
                LOGGER.debug(f'component {i+1}: {component.__class__.__name__} '
                             f'from {cat_slice_size + i * slice_size} for {slice_size}'
                             f' - {component_slice.shape.as_list()}')
                component_output = component.add_layer_on_top(component_slice)
                components_outputs.append(component_output)

            mixture_outputs = [categorical_output] + components_outputs
            return tf.concat(mixture_outputs, axis=-1)

    def _add_layer_on_top_individual_cat(self, base):
        # separate slice for categorical, shared for other
        slice_dim = base.shape.as_list()[-1]
        n_slices = len(self.components) + 1
        cat_slice_size = slice_dim // n_slices  # * 2

        LOGGER.debug("Mixture components share nn outputs and categorical has individual slice")
        LOGGER.debug(f'base shape: {base.shape.as_list()}')

        components_outputs = []

        with tf.variable_scope(self.name_scope):

            categorical_slice = tf.slice(
                base,
                [0, 0],
                [-1, cat_slice_size],
            )
            LOGGER.debug(f'categorical: slice '
                         f'from 0 for {cat_slice_size} - {categorical_slice.shape.as_list()}')
            categorical_output = self.categorical.add_layer_on_top(categorical_slice)

            components_slice = tf.slice(
                base,
                [0, cat_slice_size],
                [-1, slice_dim - cat_slice_size],
            )
            LOGGER.debug(f'components slice: '
                         f'from {cat_slice_size} for {slice_dim - cat_slice_size}'
                         f' - {components_slice.shape.as_list()}')

            for i, component in enumerate(self.components):
                component_output = component.add_layer_on_top(components_slice)
                components_outputs.append(component_output)

            mixture_outputs = [categorical_output] + components_outputs
            return tf.concat(mixture_outputs, axis=-1)

    def get_random_variable(self, nn_prediction_tensor):
        self.check_input_shape(nn_prediction_tensor)

        with tf.variable_scope(self.name_scope):
            with tf.variable_scope('random_variable'):

                categorical_predictions = tf.slice(
                    nn_prediction_tensor,
                    [0, 0],
                    [-1, self.categorical.number_of_output_neurons],
                    name='categorical_predictions',
                )
                categorical_random_variable = self.categorical.get_random_variable(categorical_predictions)

                components_random_variables = []
                start_slicing_index = self.categorical.number_of_output_neurons

                for i, component in enumerate(self.components):
                    component_predictions = tf.slice(
                        nn_prediction_tensor,
                        [0, start_slicing_index],
                        [-1, component.number_of_output_neurons],
                        name=f'component_{i}_predictions',
                    )
                    component_random_variable = component.get_random_variable(component_predictions)
                    components_random_variables.append(component_random_variable)
                    start_slicing_index += component.number_of_output_neurons
                return Mixture(categorical_random_variable, components_random_variables)

    def loss_function(self, y_true, y_pred):
        loss = - self.log_likelihood(y_true, y_pred)
        # loss = tf.math.reduce_mean(loss)
        loss = tf.reshape(loss, [-1])
        return loss

    def log_likelihood(self, y_true, y_pred):
        return self.get_random_variable(y_pred).log_prob(y_true)
