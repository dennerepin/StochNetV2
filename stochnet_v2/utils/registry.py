import tensorflow as tf


def normalize_to_list(data):
    if data is None:
        return []

    if isinstance(data, list):
        return data

    return [data]


class Registry(dict):

    def __init__(self, key_type=str, value_type=None, name='Registry'):
        self._key_type = key_type
        self._value_type = value_type
        self._name = name
        super().__init__()

    def _check_key_type(self, key):
        if self._key_type is None or isinstance(key, self._key_type):
            return

        raise TypeError(f'registry key must be type {self._key_type}')

    def _check_value_type(self, value):
        if self._value_type is None or isinstance(value, self._value_type):
            return

        raise TypeError(f'registry value must be type {self._value_type}')

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as exc:
            raise KeyError(f'key "{key}" not registered in {self._name}') from exc

    def __setitem__(self, key, value):
        self._check_key_type(key)
        self._check_value_type(value)
        if key in self:
            raise ValueError(f'item with key "{key}" already exists')

        super().__setitem__(key, value)

    def register(self, aliases):
        aliases = normalize_to_list(aliases)

        def add_item(item):
            for alias in aliases:
                self.__setitem__(alias, item)

            return item

        return add_item


CONSTRAINTS_REGISTRY = Registry(name='KernelConstraintsRegistry')
REGULARIZERS_REGISTRY = Registry(name='RegularizersRegistry')
ACTIVATIONS_REGISTRY = Registry(name='ActivationsRegistry')

CONSTRAINTS_REGISTRY['maxnorm'] = tf.keras.constraints.MaxNorm(3.)
CONSTRAINTS_REGISTRY['minmaxnorm'] = tf.keras.constraints.MinMaxNorm(0., 3.)
CONSTRAINTS_REGISTRY['unitnorm'] = tf.keras.constraints.UnitNorm()
CONSTRAINTS_REGISTRY['none'] = None

REGULARIZERS_REGISTRY['l1'] = tf.keras.regularizers.l1(0.001)
REGULARIZERS_REGISTRY['l2'] = tf.keras.regularizers.l2(0.001)
REGULARIZERS_REGISTRY['l1_l2'] = tf.keras.regularizers.l1_l2(0.001, 0.001)
REGULARIZERS_REGISTRY['none'] = None

ACTIVATIONS_REGISTRY['relu'] = tf.compat.v1.nn.relu
ACTIVATIONS_REGISTRY['relu6'] = tf.compat.v1.nn.relu6
ACTIVATIONS_REGISTRY['swish'] = tf.compat.v1.nn.swish
ACTIVATIONS_REGISTRY['none'] = tf.keras.activations.linear
ACTIVATIONS_REGISTRY['leakyrelu'] = tf.keras.layers.LeakyReLU(0.2)
ACTIVATIONS_REGISTRY['prelu'] = tf.keras.layers.PReLU()
ACTIVATIONS_REGISTRY['elu'] = tf.keras.layers.ELU(1.0)
