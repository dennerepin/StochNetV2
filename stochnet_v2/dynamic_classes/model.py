import os
import pickle
import numpy as np
import logging
import tensorflow as tf
from functools import partial

import stochnet_v2.dynamic_classes.nn_body_search as nn_body_search
import stochnet_v2.dynamic_classes.nn_body as nn_body
from stochnet_v2.static_classes.model import StochNet
from stochnet_v2.dynamic_classes.nn_body_search import get_genotypes
from stochnet_v2.utils.registry import CONSTRAINTS_REGISTRY
from stochnet_v2.utils.registry import REGULARIZERS_REGISTRY


LOGGER = logging.getLogger('dynamic_classes.model')


def _body_search(x, **kwargs):
    shape = x.shape.as_list()
    x = tf.reshape(x, shape=(shape[0] or -1, np.prod(shape[1:]),))
    return nn_body_search.body(x, **kwargs)


def _body_trained(x, **kwargs):
    shape = x.shape.as_list()
    x = tf.reshape(x, shape=(shape[0] or -1, np.prod(shape[1:]),))
    return nn_body.body(x, **kwargs)


class NASStochNet(StochNet):

    def _get_body_fn(self, body_config):

        self.n_cells = body_config['n_cells']
        self.expansion_multiplier = body_config['expansion_multiplier']
        self.cell_size = body_config['cell_size']
        self.n_states_reduce = body_config['n_states_reduce']

        kernel_constraint = body_config.pop("kernel_constraint", "none")
        body_config["kernel_constraint"] = CONSTRAINTS_REGISTRY[kernel_constraint]

        kernel_regularizer = body_config.pop("kernel_regularizer", "none")
        body_config["kernel_regularizer"] = REGULARIZERS_REGISTRY[kernel_regularizer]

        bias_constraint = body_config.pop("bias_constraint", "none")
        body_config["bias_constraint"] = CONSTRAINTS_REGISTRY[bias_constraint]

        bias_regularizer = body_config.pop("bias_regularizer", "none")
        body_config["bias_regularizer"] = REGULARIZERS_REGISTRY[bias_regularizer]

        activity_regularizer = body_config.pop("activity_regularizer", "none")
        body_config["activity_regularizer"] = REGULARIZERS_REGISTRY[activity_regularizer]

        return partial(_body_search, **body_config)

    def save_genotypes(self, n_cells=None, cell_size=None, n_states_reduce=None):
        n_cells = n_cells or self.n_cells
        cell_size = cell_size or self.cell_size
        n_states_reduce = n_states_reduce or self.n_states_reduce
        genotypes = get_genotypes(self.session, n_cells=n_cells, cell_size=cell_size, n_states_reduce=n_states_reduce)
        with open(os.path.join(self.model_explorer.model_folder, 'genotypes.pickle'), 'wb') as f:
            pickle.dump(genotypes, f)

    def _load_genotypes(self):
        with open(os.path.join(self.model_explorer.model_folder, 'genotypes.pickle'), 'rb') as f:
            genotypes = pickle.load(f)
        return genotypes

    def recreate_from_genome(self, ckpt_path):
        self.session.close()
        del self.graph
        del self.session
        self._sample_tensor = None

        body_config_path = os.path.join(self.model_explorer.model_folder, 'body_config.json')
        mixture_config_path = os.path.join(self.model_explorer.model_folder, 'mixture_config.json')
        body_config = self._read_config(body_config_path)
        mixture_config = self._read_config(mixture_config_path)

        expansion_multiplier = body_config['expansion_multiplier']
        genotypes = self._load_genotypes()

        body_fn = partial(
            _body_trained,
            genotypes=genotypes,
            expansion_multiplier=expansion_multiplier
        )

        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            self.session = tf.compat.v1.Session()
            self._build_main_graph(body_fn, mixture_config)
            self._build_sampling_graph()
            self._save_graph_keys()
            self.custom_restore_from_checkpoint(ckpt_path)

    def custom_restore_from_checkpoint(self, ckpt_path):
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
        model_variables = self.graph.get_collection('variables')
        # print("Custom restore")
        for v in model_variables:
            name = v.name.replace(':0', '')
            # print(name, v.shape, reader.get_tensor(name).shape)
            self.session.run(v.assign(reader.get_tensor(name)))
        self.restored = True
