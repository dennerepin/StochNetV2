import numpy as np
import os
import shutil
import tensorflow as tf


def maybe_create_dir(dir_path, erase_existing=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        if erase_existing:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)


def copy_graph(src_graph, dst_graph, dst_scope=None):
    src_meta_graph = tf.compat.v1.train.export_meta_graph(graph=src_graph)
    with dst_graph.as_default():
        tf.compat.v1.train.import_meta_graph(src_meta_graph, import_scope=dst_scope)


def get_transformed_tensor(src_tensor, dst_graph, dst_scope=''):
    dst_tensor_name = src_tensor.name
    if dst_scope:
        dst_tensor_name = f'{dst_scope}/{dst_tensor_name}'

    return dst_graph.get_tensor_by_name(dst_tensor_name)


def count_parameters(model):
    with model.graph.as_default():
        trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    return np.sum([np.prod(v.shape.as_list()) for v in trainable_vars])


def count_dense_flop(layer, x):
    return (x.shape.as_list()[-1] + 1) * layer.units


def graph_def_to_graph(graph_def):
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph