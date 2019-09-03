import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import shutil
import tensorflow as tf
from functools import partial


def maybe_create_dir(dir_path, erase_existing=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        if erase_existing:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)


def copy_graph(src_graph, dst_graph, dst_scope=None, **kwargs):
    src_meta_graph = tf.compat.v1.train.export_meta_graph(graph=src_graph)
    with dst_graph.as_default():
        tf.compat.v1.train.import_meta_graph(src_meta_graph, import_scope=dst_scope, **kwargs)


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


def plot_trace(trace, show=True, labels=None, **kwargs):
    n_species = trace.shape[-1] - 1
    cmap = get_cmap(n_species+1)
    for i in range(n_species):
        lab = labels[i] if labels else None
        plt.plot(trace[:, i+1], c=cmap(i), label=lab, **kwargs)
    if show is True:
        plt.show()


def plot_traces(traces, show=False, labels=None, **kwargs):
    if len(traces.shape) == 2:
        plot_trace(traces, show=False, labels=labels, **kwargs)
    else:
        for i, trace in enumerate(traces):
            plot_trace(trace, show=False, labels=labels if i == 0 else None, **kwargs)
    if show is True:
        plt.show()


def random_pick_traces(data_shape, n):
    return np.random.choice(data_shape[0], n)


def plot_random_traces(data, n, show=False, **kwargs):
    for i in random_pick_traces(data.shape, n):
        plot_trace(data[i], show=show, **kwargs)


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def _single_trace(
        setting,
        gillespy_model,
        traj_per_setting,
):
    gillespy_model.set_species_initial_value(setting)
    traces = gillespy_model.run(number_of_trajectories=traj_per_setting, show_labels=False)
    traces = np.array(traces)
    return traces


def generate_gillespy_traces(settings, step_to, timestep, gillespy_model, traj_per_setting=10):
    endtime = int(step_to * timestep)
    nb_of_steps = int(math.ceil((endtime / timestep))) + 1
    gillespy_model.timespan(np.linspace(0, endtime, nb_of_steps))

    count = multiprocessing.cpu_count() * 3 // 4
    pool = multiprocessing.Pool(
        processes=count
    )

    task = partial(
        _single_trace,
        gillespy_model=gillespy_model,
        traj_per_setting=traj_per_setting,
    )

    simulated_traces = pool.map(task, settings)
    return np.stack(simulated_traces)
