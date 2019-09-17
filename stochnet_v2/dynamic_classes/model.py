import os
import pickle
import numpy as np
import logging
import tensorflow as tf
from functools import partial

from stochnet_v2.static_classes.model import StochNet
from stochnet_v2.dynamic_classes import nn_body_search
from stochnet_v2.dynamic_classes import nn_body
from stochnet_v2.dynamic_classes.nn_body_search import get_genotypes

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
        self.n_summ_states = body_config['n_summ_states']
        return partial(_body_search, **body_config)

    def save_genotypes(self, n_cells=None, cell_size=None, n_summ_states=None):
        n_cells = n_cells or self.n_cells
        cell_size = cell_size or self.cell_size
        n_summ_states = n_summ_states or self.n_summ_states
        genotypes = get_genotypes(self.session, n_cells=n_cells, cell_size=cell_size, n_summ_states=n_summ_states)
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

    # def finalize(self):
    #     graph = tf.compat.v1.Graph()
    #     with graph.as_default():
    #         session = tf.compat.v1.Session()
    #         graph_def = tf.compat.v1.graph_util.extract_sub_graph(
    #             graph_def=self.graph.as_graph_def(),
    #             dest_nodes=self.dest_nodes
    #         )
    #         tf.compat.v1.import_graph_def(graph_def, name='')
    #         self.input_placeholder = graph.get_tensor_by_name(self._input_placeholder_name)
    #         self.pred_tensor = graph.get_tensor_by_name(self._pred_tensor_name)
    #         self.pred_tensor = graph.get_tensor_by_name(self._pred_tensor_name)
    #         self.pred_placeholder = graph.get_tensor_by_name(self._pred_placeholder_name)
    #         self.sample_shape_placeholder = graph.get_tensor_by_name(self._sample_shape_placeholder_name)
    #         self.sample_tensor = graph.get_tensor_by_name(self._sample_tensor_name)
    # 
    #     self.session.close()
    #     del self.graph
    #     del self.session
    #     self.graph = graph
    #     self.session = session

    def custom_restore_from_checkpoint(self, ckpt_path):
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
        model_variables = self.graph.get_collection('variables')
        print("Custom restore")
        for v in model_variables:
            name = v.name.replace(':0', '')
            print(name, v.shape, reader.get_tensor(name).shape)
            self.session.run(v.assign(reader.get_tensor(name)))
        self.restored = True
