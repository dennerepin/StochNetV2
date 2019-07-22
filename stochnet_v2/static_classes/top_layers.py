import abc
import tensorflow as tf

from stochnet_v2.utils.errors import ShapeError
from stochnet_v2.utils.errors import DimensionError
from stochnet_v2.static_classes.random_variables import Categorical
from stochnet_v2.static_classes.random_variables import MultivariateNormalDiag
from stochnet_v2.static_classes.random_variables import MultivariateNormalTriL
from stochnet_v2.static_classes.random_variables import MultivariateLogNormalTriL
from stochnet_v2.static_classes.random_variables import Mixture


class RandomVariableOutputLayer(abc.ABC):

    def __init__(self):
        self._pred_placeholder = None
        self._random_variable = None
        self._sample_shape_placeholder = None
        self._sample_tensor = None

        self._sample_space_dimension = None
        self._number_of_output_neurons = None

    # def __new__(cls, *args, **kwargs):
    #     cls._pred_placeholder = tf.placeholder(
    #         dtype=tf.float32,
    #         shape=(None, cls.number_of_output_neurons),
    #         name='pred_placeholder',
    #     )
    #     cls._random_variable = cls.get_random_variable(cls._pred_placeholder)
    #     cls._sample_shape_placeholder = tf.placeholder(tf.int32, None, name='sample_shape_placeholder')
    #     cls._sample_tensor = cls._random_variable.sample(cls._sample_shape_placeholder)

    @abc.abstractmethod
    def add_layer_on_top(self, base_model):
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
        initialized using the parameters provived by the neural network output.
        Additional checks might be needed for certain families of random variables.
        """

    @property
    @abc.abstractmethod
    def number_of_output_neurons(self):
        pass

    # @abc.abstractmethod
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

    # @abc.abstractmethod
    def get_description(self, nn_prediction):
        """Return a string containing a description of the random variables inizialized
        using the parameters in nn_prediction"""
        # TODO: fixme: get description w/o re-creating graph for random variable?
        random_variable = self.get_random_variable(nn_prediction)
        description = random_variable.get_description()
        return description

    # @abc.abstractmethod
    def _build_sampling_graph(self):
        self._pred_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.number_of_output_neurons),
            name='pred_placeholder',
        )
        self._random_variable = self.get_random_variable(self._pred_placeholder)
        self._sample_shape_placeholder = tf.placeholder(tf.int32, None, name='sample_shape_placeholder')
        self._sample_tensor = self._random_variable.sample(self._sample_shape_placeholder)

    # @abc.abstractmethod
    def sample(self, nn_prediction_np, sample_shape=(), sess=None):

        if self._sample_tensor is None:
            self._build_sampling_graph()

        if sess is None:
            with tf.Session() as sess:
                res = sess.run(
                    self._sample_tensor,
                    feed_dict={
                        self._pred_placeholder: nn_prediction_np,
                        self._sample_shape_placeholder: sample_shape
                    }
                )
        else:
            res = sess.run(
                self._sample_tensor,
                feed_dict={
                    self._pred_placeholder: nn_prediction_np,
                    self._sample_shape_placeholder: sample_shape
                }
            )
        return res


class CategoricalOutputLayer(RandomVariableOutputLayer):

    def __init__(
            self,
            number_of_classes,
            hidden_size=None,
            regularizer=None,
            kernel_constraint=None,
    ):
        super().__init__()
        self.number_of_classes = number_of_classes
        self.random_variable = None
        self.hidden_size = hidden_size
        self._kernel_constraint = kernel_constraint
        self._regularizer = regularizer

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

    def add_layer_on_top(self, base_model):

        if self.hidden_size:
            base_model = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._regularizer,
            )(base_model)
            base_model1 = tf.keras.layers.LeakyReLU(0.2)(base_model)
            base_model1 = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._regularizer,
            )(base_model1)
            base_model = tf.keras.layers.Add()([base_model, base_model1])

        logits = tf.keras.layers.Dense(
            self.number_of_output_neurons,
            activation=None,
            kernel_constraint=self._kernel_constraint,
            activity_regularizer=self._regularizer,
        )(base_model)

        return logits

    def get_random_variable(self, nn_prediction_tensor):
        self.check_input_shape(nn_prediction_tensor)
        return Categorical(nn_prediction_tensor)

    def loss_function(self, y_true, y_pred):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=self.get_random_variable(y_pred)
        )
        return loss

    # def check_input_shape(self, input_):
    #     super().check_input_shape(input_)

    # def get_description(self, nn_prediction):
    #     return super().get_description(nn_prediction)

    # def _build_sampling_graph(self):
    #     super()._build_sampling_graph()

    # def sample(self, nn_prediction, sample_shape=(), sess=None):
    #     return super().sample(nn_prediction, sample_shape, sess)


class MultivariateNormalDiagOutputLayer(RandomVariableOutputLayer):

    def __init__(
            self,
            sample_space_dimension,
            hidden_size=None,
            mu_regularizer=None,
            diag_regularizer=None,
            kernel_constraint=None,
    ):
        super().__init__()
        self.sample_space_dimension = sample_space_dimension
        self.hidden_size = hidden_size
        self._mu_regularizer = mu_regularizer
        self._diag_regularizer = diag_regularizer
        self._kernel_constraint = kernel_constraint

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

    def add_layer_on_top(self, base_model):
        mu, diag = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=-1)
        )(base_model)

        if self.hidden_size:
            mu = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._mu_regularizer,
            )(mu)
            mu1 = tf.keras.layers.LeakyReLU(0.2)(mu)
            mu1 = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._mu_regularizer,
            )(mu1)
            mu = tf.keras.layers.Add()([mu, mu1])

            diag = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._diag_regularizer,
            )(diag)
            diag1 = tf.keras.layers.LeakyReLU(0.2)(diag)
            diag1 = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._diag_regularizer,
            )(diag1)
            diag = tf.keras.layers.Add()([diag, diag1])

        mu = tf.keras.layers.Dense(
            self._sample_space_dimension,
            activation=None,
            kernel_constraint=self._kernel_constraint,
            activity_regularizer=self._mu_regularizer,
        )(mu)

        diag = tf.keras.layers.Dense(
            self._sample_space_dimension,
            activation=tf.exp,
            kernel_constraint=self._kernel_constraint,
            activity_regularizer=self._diag_regularizer,
        )(diag)

        return tf.keras.layers.Concatenate(axis=-1)([mu, diag])

    def get_random_variable(self, nn_prediction_tensor):
        self.check_input_shape(nn_prediction_tensor)
        mu = tf.slice(
            nn_prediction_tensor,
            [0, 0],
            [-1, self._sample_space_dimension]
        )
        diag = tf.slice(
            nn_prediction_tensor,
            [0, self._sample_space_dimension],
            [-1, self._sample_space_dimension]
        )
        return MultivariateNormalDiag(mu, diag)

    def loss_function(self, y_true, y_pred):
        loss = - self.log_likelihood(y_true, y_pred)
        loss = tf.reshape(loss, [-1])
        return loss

    def log_likelihood(self, y_true, y_pred):
        return self.get_random_variable(y_pred).log_prob(y_true)


class MultivariateNormalTriLOutputLayer(RandomVariableOutputLayer):

    def __init__(
            self,
            sample_space_dimension,
            hidden_size=None,
            mu_regularizer=None,
            diag_regularizer=None,
            sub_diag_regularizer=None,
            kernel_constraint=None,
    ):
        super().__init__()
        self.sample_space_dimension = sample_space_dimension
        self.hidden_size = hidden_size
        self._mu_regularizer = mu_regularizer
        self._diag_regularizer = diag_regularizer
        self._sub_diag_regularizer = sub_diag_regularizer
        self._kernel_constraint = kernel_constraint

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

    def add_layer_on_top(self, base_model):

        shape = base_model.shape.as_list()
        s = shape[-1] // 3

        mu, diag, sub_diag = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=[s, s, shape[-1] - 2 * s], axis=-1)
        )(base_model)

        if self.hidden_size:
            mu = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._mu_regularizer,
            )(mu)
            mu1 = tf.keras.layers.LeakyReLU(0.2)(mu)
            mu1 = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._mu_regularizer,
            )(mu1)
            mu = tf.keras.layers.Add()([mu, mu1])

            diag = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._diag_regularizer,
            )(diag)
            diag1 = tf.keras.layers.LeakyReLU(0.2)(diag)
            diag1 = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._diag_regularizer,
            )(diag1)
            diag = tf.keras.layers.Add()([diag, diag1])

            sub_diag = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._sub_diag_regularizer,
            )(sub_diag)
            sub_diag1 = tf.keras.layers.LeakyReLU(0.2)(sub_diag)
            sub_diag1 = tf.keras.layers.Dense(
                self.hidden_size,
                kernel_constraint=self._kernel_constraint,
                activity_regularizer=self._sub_diag_regularizer,
            )(sub_diag1)
            sub_diag = tf.keras.layers.Add()([sub_diag, sub_diag1])

        mu = tf.keras.layers.Dense(
            self._sample_space_dimension,
            activation=None,
            kernel_constraint=self._kernel_constraint,
            activity_regularizer=self._mu_regularizer,
        )(mu)

        diag = tf.keras.layers.Dense(
            self._sample_space_dimension,
            activation=tf.exp,
            kernel_constraint=self._kernel_constraint,
            activity_regularizer=self._diag_regularizer,
        )(diag)

        sub_diag = tf.keras.layers.Dense(
            self._number_of_sub_diag_entries,
            activation=None,
            kernel_constraint=self._kernel_constraint,
            activity_regularizer=self._sub_diag_regularizer,
        )(sub_diag)

        return tf.keras.layers.Concatenate(axis=-1)([mu, diag, sub_diag])

    def get_random_variable(self, nn_prediction_tensor):
        self.check_input_shape(nn_prediction_tensor)
        mu = tf.slice(
            nn_prediction_tensor,
            [0, 0],
            [-1, self._sample_space_dimension],
        )
        diag = tf.slice(
            nn_prediction_tensor,
            [0, self._sample_space_dimension],
            [-1, self._sample_space_dimension],
        )
        sub_diag = tf.slice(
            nn_prediction_tensor,
            [0, 2 * self._sample_space_dimension],
            [-1, self._number_of_sub_diag_entries],
        )
        with tf.control_dependencies([tf.assert_positive(diag)]):
            flat_tril = tf.concat([diag, sub_diag], axis=-1)
        return MultivariateNormalTriL(mu, flat_tril)

    def loss_function(self, y_true, y_pred):
        loss = - self.log_likelihood(y_true, y_pred)
        loss = tf.reshape(loss, [-1])
        return loss

    def log_likelihood(self, y_true, y_pred):
        return self.get_random_variable(y_pred).log_prob(y_true)


class MultivariateLogNormalTriLOutputLayer(MultivariateNormalTriLOutputLayer):

    def get_random_variable(self, nn_prediction_tensor):
        self.check_input_shape(nn_prediction_tensor)
        mu = tf.slice(
            nn_prediction_tensor,
            [0, 0],
            [-1, self._sample_space_dimension],
        )
        diag = tf.slice(
            nn_prediction_tensor,
            [0, self._sample_space_dimension],
            [-1, self._sample_space_dimension],
        )
        sub_diag = tf.slice(
            nn_prediction_tensor,
            [0, 2 * self._sample_space_dimension],
            [-1, self._number_of_sub_diag_entries],
        )
        with tf.control_dependencies([tf.assert_positive(diag)]):
            flat_tril = tf.concat([diag, sub_diag], axis=-1)
        return MultivariateLogNormalTriL(mu, flat_tril)


class MixtureOutputLayer(RandomVariableOutputLayer):

    def __init__(
            self,
            components,
            coeff_regularizer=None
    ):
        super().__init__()
        self.number_of_components = len(components)
        self.categorical = CategoricalOutputLayer(self.number_of_components, coeff_regularizer)
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
            raise DimensionError(f"The random variables which have been passed "
                                 f"as mixture components sample from spaces with "
                                 f"different dimensions.\n"
                                 f"This is the list of sample spaces dimensions:\n"
                                 f"{sample_space_dims}")

        self._sample_space_dimension = sample_space_dims[0]

    def set_number_of_output_neurons(self):
        self._number_of_output_neurons = self.categorical.number_of_output_neurons
        for component in self.components:
            self._number_of_output_neurons += component.number_of_output_neurons

    def add_layer_on_top(self, base_model):
        # TODO: split to slice for each component?
        categorical_layer = self.categorical.add_layer_on_top(base_model)
        components_layers = [component.add_layer_on_top(base_model) for component in self.components]
        mixture_layers = [categorical_layer] + components_layers
        return tf.keras.layers.Concatenate(axis=-1)(mixture_layers)

    def get_random_variable(self, nn_prediction_tensor):
        self.check_input_shape(nn_prediction_tensor)

        categorical_predictions = tf.slice(
            nn_prediction_tensor,
            [0, 0],
            [-1, self.categorical.number_of_output_neurons],
        )
        categorical_random_variable = self.categorical.get_random_variable(categorical_predictions)

        components_random_variables = []
        start_slicing_index = self.categorical.number_of_output_neurons

        for component in self.components:
            component_predictions = tf.slice(
                nn_prediction_tensor,
                [0, start_slicing_index],
                [-1, component.number_of_output_neurons]
            )
            component_random_variable = component.get_random_variable(component_predictions)
            components_random_variables.append(component_random_variable)
            start_slicing_index += component.number_of_output_neurons
        return Mixture(categorical_random_variable, components_random_variables)

    def loss_function(self, y_true, y_pred):
        loss = - self.log_likelihood(y_true, y_pred)
        loss = tf.reshape(loss, [-1])
        return loss

    def log_likelihood(self, y_true, y_pred):
        return self.get_random_variable(y_pred).log_prob(y_true)
