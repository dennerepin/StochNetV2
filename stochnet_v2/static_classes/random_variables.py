import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

tfd = tfp.distributions
bijectors = tfp.bijectors

np.set_printoptions(suppress=True, precision=3)


class Categorical:

    def __init__(
            self,
            logits,
            validate_args=False,
    ):
        self.distribution_obj = tfd.Categorical(
            logits=logits,
            validate_args=validate_args,
        )
        self.number_of_classes = self.distribution_obj.num_categories

    def sample(self, sample_shape=()):
        return self.distribution_obj.sample(sample_shape=sample_shape)

    @property
    def nb_of_independent_random_variables(self):
        return int(np.array(self.distribution_obj.batch_shape).prod())

    def get_description(self):
        descriptions = []

        with tf.Session():
            if isinstance(self.number_of_classes, tf.Tensor):
                self.number_of_classes = self.number_of_classes.eval()

            flattened_class_probabilities = tf.reshape(
                self.distribution_obj.probs,
                [-1, self.number_of_classes]
            ).eval()

        description_preamble = f"Categorical random variable with {self.number_of_classes} classes.\n\n"

        for j in range(self.nb_of_independent_random_variables):
            description = f"{j+1}." + description_preamble + \
                          f"  Class probabilities: \n " \
                          f"  {flattened_class_probabilities[j, :]}\n\n"
            descriptions.append(description)

        return descriptions


class MultivariateNormalDiag:

    def __init__(
            self,
            mu,
            diag,
            validate_args=False,
    ):
        self.distribution_obj = tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=diag,
            validate_args=validate_args,
        )

    def log_prob(self, value):
        return self.distribution_obj.log_prob(value)

    @property
    def mean(self):
        return self.distribution_obj.mean()

    @property
    def covariance(self):
        return self.distribution_obj.covariance()

    @property
    def sample_space_dimension(self):
        return self.distribution_obj.event_shape.as_list()[0]

    def sample(self, sample_shape=()):
        return self.distribution_obj.sample(sample_shape=sample_shape)

    @property
    def nb_of_independent_random_variables(self):
        return int(np.array(self.distribution_obj.batch_shape).prod())

    def get_description(self):
        descriptions = []
        with tf.Session():
            flattened_means = tf.reshape(
                self.mean,
                [-1, self.sample_space_dimension]
            ).eval()
            flattened_sigmas = tf.reshape(
                self.covariance,
                [-1, self.sample_space_dimension, self.sample_space_dimension]
            ).eval()

        description_preamble = "Multivariate Normal random variable.\n\n"

        for j in range(self.nb_of_independent_random_variables):

            description = f"{j+1}." + description_preamble + \
                          f"  Mean:\n{flattened_means[j, :]}\n\n " \
                          f"  Covariance matrix:\n{flattened_sigmas[j, ...]}\n"
            descriptions.append(description)

        return descriptions


class MultivariateNormalTriL:

    def __init__(
            self,
            mu,
            tril_flat,
            validate_args=False
    ):
        tril = tfd.fill_triangular(tril_flat)
        self.distribution_obj = tfd.MultivariateNormalTriL(
            loc=mu,
            scale_tril=tril,
            validate_args=validate_args,
        )

    def log_prob(self, value):
        return self.distribution_obj.log_prob(value)

    @property
    def mean(self):
        return self.distribution_obj.mean()

    @property
    def covariance(self):
        return self.distribution_obj.covariance()

    @property
    def sample_space_dimension(self):
        return self.distribution_obj.event_shape.as_list()[0]

    def sample(self, sample_shape=()):
        return self.distribution_obj.sample(sample_shape=sample_shape)

    @property
    def nb_of_independent_random_variables(self):
        return int(np.array(self.distribution_obj.batch_shape).prod())

    def get_description(self):
        descriptions = []
        with tf.Session():
            flattened_means = tf.reshape(
                self.mean,
                [-1, self.sample_space_dimension]
            ).eval()
            flattened_sigmas = tf.reshape(
                self.covariance,
                [-1, self.sample_space_dimension, self.sample_space_dimension]
            ).eval()

        description_preamble = "Multivariate Normal random variable.\n\n"

        for j in range(self.nb_of_independent_random_variables):

            description = f"{j+1}." + description_preamble + \
                          f"  Mean:\n{flattened_means[j, :]}\n\n " \
                          f"  Covariance matrix:\n{flattened_sigmas[j, ...]}\n"
            descriptions.append(description)

        return descriptions


class MultivariateLogNormalTriL:

    def __init__(
            self,
            mu,
            tril,
            validate_args=False,
    ):
        tril = tfd.fill_triangular(tril)
        self.distribution_obj = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalTriL(
                loc=mu,
                scale_tril=tril,
            ),
            bijector=bijectors.Inline(
                forward_fn=tf.math.exp,
                inverse_fn=tf.math.log,
                inverse_log_det_jacobian_fn=(lambda y: -tf.reduce_sum(tf.log(y), reduction_indices=-1)),
                forward_min_event_ndims=1,
            ),
            name="LogNormalTransformedDistribution",
            validate_args=validate_args,
        )
        self._normal_mean = mu
        self._normal_tril = tril

    def log_prob(self, value):
        with tf.control_dependencies([tf.assert_positive(value)]):
            return self.distribution_obj.log_prob(value)

    @property
    def mean(self):
        return self._normal_mean

    @property
    def covariance(self):
        n_dims = self._normal_tril.shape.ndims
        perm = [i for i in range(n_dims - 2)] + [n_dims - 1, n_dims - 2]
        scale_tril_transposed = tf.transpose(self._normal_tril, perm=perm)
        return tf.matmul(self._normal_tril, scale_tril_transposed)

    @property
    def sample_space_dimension(self):
        return self.distribution_obj.event_shape.as_list()[0]

    def sample(self, sample_shape=()):
        return self.distribution_obj.sample(sample_shape=sample_shape)

    @property
    def nb_of_independent_random_variables(self):
        return int(np.array(self.distribution_obj.batch_shape).prod())

    def get_description(self):
        descriptions = []
        with tf.Session():
            flattened_means = tf.reshape(
                self.mean,
                [-1, self.sample_space_dimension]
            ).eval()
            flattened_sigmas = tf.reshape(
                self.covariance,
                [-1, self.sample_space_dimension, self.sample_space_dimension]
            ).eval()

        description_preamble = "Multivariate Normal random variable.\n\n"

        for j in range(self.nb_of_independent_random_variables):

            description = f"{j+1}." + description_preamble + \
                          f"  Mean:\n{flattened_means[j, :]}\n\n " \
                          f"  Covariance matrix:\n{flattened_sigmas[j, ...]}\n"
            descriptions.append(description)

        return descriptions


class Mixture:

    def __init__(
            self,
            cat,
            components,
            validate_args=False,
    ):
        self.cat = cat
        self.components = list(components)
        self.number_of_components = len(components)

        tf_cat = cat.distribution_obj
        tf_components = [component.distribution_obj for component in components]

        self.distribution_obj = tfd.Mixture(
            cat=tf_cat,
            components=tf_components,
            validate_args=validate_args
        )

    def log_prob(self, value):
        return self.distribution_obj.log_prob(value)

    def sample(self, sample_shape=()):
        return self.distribution_obj.sample(sample_shape=sample_shape)

    @property
    def nb_of_indipendent_random_variables(self):
        return np.array(self.distribution_obj.batch_shape).prod()

    def get_description(self):
        descriptions = []
        description_preamble = f"Mixture random variable with {self.number_of_components} components.\n\n"
        cat_descriptions = self.cat.get_description()
        component_descriptions = [component.get_description() for component in self.components]

        for j in range(self.nb_of_indipendent_random_variables):
            description = description_preamble + cat_descriptions[j]

            for component_description in component_descriptions:
                description += component_description[j]

            descriptions.append(description)

        return descriptions
