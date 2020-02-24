import graphviz
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import shutil
import tensorflow as tf
from functools import partial
from time import time


def str_to_bool(arg):
    arg_upper = str(arg).upper()
    if 'TRUE'.startswith(arg_upper):
        return True
    elif 'FALSE'.startswith(arg_upper):
        return False
    else:
        pass


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
        params_to_randomize,
        traj_per_setting,
):
    # start = time()
    nb_randomized_params = len(params_to_randomize)
    if nb_randomized_params > 0:
        params = setting[-nb_randomized_params:]
        setting = setting[:-nb_randomized_params]
        param_dict = dict(zip(params_to_randomize, params))

        gillespy_model.set_species_initial_value(setting)
        gillespy_model.set_parameters(param_dict)

    else:
        gillespy_model.set_species_initial_value(setting)

    traces = gillespy_model.run(number_of_trajectories=traj_per_setting, show_labels=False)
    traces = np.array(traces)
    # elapsed = time() - start
    # print(f'..single trace: shape={traces.shape}, elapsed {elapsed:.2f}')
    return traces


def generate_gillespy_traces(
        settings,
        n_steps,
        timestep,
        gillespy_model,
        params_to_randomize,
        traj_per_setting=10,
):
    # start = time()

    # V1
    # endtime = int(step_to * timestep)
    # nb_of_steps = int(math.ceil((endtime / timestep))) + 1

    # V2
    endtime = n_steps * timestep
    nb_of_steps = int(endtime // timestep) + 1

    gillespy_model.timespan(np.linspace(0, endtime, nb_of_steps))

    count = multiprocessing.cpu_count() * 3 // 4
    pool = multiprocessing.Pool(
        processes=count
    )

    task = partial(
        _single_trace,
        gillespy_model=gillespy_model,
        params_to_randomize=params_to_randomize,
        traj_per_setting=traj_per_setting,
    )

    simulated_traces = pool.map(task, settings)
    simulated_traces = np.stack(simulated_traces)

    # elapsed = time() - start
    # print(f'..single trace: shape={simulated_traces.shape}, elapsed {elapsed:.2f}')

    return simulated_traces


def apply_regularization(regularizer_fn, tensor):
    with tf.compat.v1.variable_scope("regularization"):
        loss = regularizer_fn(tensor)
    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, loss)


def numpy_softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e, axis=-1, keepdims=True)


def postprocess_description_dict(description):
    if 'logits' in description['cat']:
        logits = description['cat'].pop('logits')
        probs = numpy_softmax(logits)
        description['cat']['probs'] = probs

    for component_dict in description['components']:

        if 'diag' in component_dict:
            diag = component_dict.pop('diag')
            batch_size, diag_size = diag.shape
            cov = np.zeros([batch_size, diag_size, diag_size])
            for i in range(batch_size):
                np.fill_diagonal(cov[i], diag[i])
            component_dict['cov'] = cov

        elif 'tril' in component_dict:
            tril = component_dict.pop('tril')
            cov = np.matmul(tril, np.transpose(tril, [0, 2, 1]))
            component_dict['cov'] = cov
    return description


def visualize_description(description, save_figs_to=None, prefix='distribution_visualization'):
    cat = description['cat']['probs']
    batch_size = cat.shape[0]
    n_components = len(description['components'])

    figures = []

    for i in range(batch_size):

        fig, axes = plt.subplots(nrows=n_components + 1, ncols=2, figsize=(16, 8 * (n_components + 1)))

        p = cat[i:i + 1]
        ax = axes[0, 0]
        im = ax.imshow(p)
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)
        ax.set_title("Probs")
        ax.get_xaxis().set_ticks(range(n_components))
        ax.get_yaxis().set_ticks([])

        ax = axes[0, 1]
        ax.set_axis_off()

        for j, component_dict in enumerate(description['components']):
            m = component_dict['mean']
            ax = axes[j + 1, 0]
            im = ax.imshow(m[i:i + 1])
            plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)
            ax.set_title("Mean")
            ax.get_yaxis().set_ticks([])

            c = component_dict['cov']
            ax = axes[j + 1, 1]
            im = ax.imshow(c[i])
            plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)
            ax.set_title("Covariance")

        fig.tight_layout()
        fig.suptitle(f"Component {i}", fontsize=16, y=1.0)
        figures.append(fig)

        if isinstance(save_figs_to, str):
            maybe_create_dir(save_figs_to)
            [fig.savefig(os.path.join(save_figs_to, f'{prefix}_{i}')) for i, fig in enumerate(figures)]

    return figures


def visualize_genotypes(genotypes, filename, view=False):

    gg = graphviz.Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(
            style='filled',
            shape='rect',
            align='center',
            fontsize='20',
            height='0.5',
            width='0.5',
            penwidth='2',
            fontname="times"),
        engine='dot')

    minus_two_cell = "C-2"
    minus_one_cell = "C-1"

    gg.body.extend(['rankdir=LR'])

    gg.node("input", fillcolor='darkseagreen2')
    gg.node(minus_two_cell, fillcolor='gray')
    gg.node(minus_one_cell, fillcolor='gray')
    gg.edge("input", minus_two_cell, label='identity', fillcolor="gray")
    gg.edge("input", minus_one_cell, label='identity', fillcolor="gray")

    for cell_num, genotype in enumerate(genotypes, 1):

        g = graphviz.Digraph(
            format='pdf',
            edge_attr=dict(fontsize='20', fontname="times"),
            node_attr=dict(
                style='filled',
                shape='rect',
                align='center',
                fontsize='20',
                height='0.5',
                width='0.5',
                penwidth='2',
                fontname="times"),
            engine='dot')

        if not len(genotype.normal) == 0:
            reduce_indexes = genotype.normal_reduce
            genotype = genotype.normal
        else:
            assert not len(genotype.expand) == 0
            reduce_indexes = genotype.expand_reduce
            genotype = genotype.expand

        if cell_num == 1:
            v_pprev = minus_two_cell
            v_prev = minus_one_cell
        elif cell_num == 2:
            v_pprev = minus_one_cell
            v_prev = f"C{cell_num - 1}_out"
        else:
            v_pprev = f"C{cell_num - 2}_out"
            v_prev = f"C{cell_num - 1}_out"

        g.node(v_pprev, fillcolor='darkseagreen2')
        g.node(v_prev, fillcolor='darkseagreen2')

        assert len(genotype) % 2 == 0
        steps = len(genotype) // 2

        for i in range(2):
            g.node(f"C{cell_num}_s{i}", fillcolor='gray')

        for i in range(2, steps + 2):
            g.node(f"C{cell_num}_s{i}", fillcolor='lightblue')

        for i in range(2):
            op = "expand_op"
            if i == 0:
                u = v_pprev
            else:
                u = v_prev
            v = f"C{cell_num}_s{i}"
            g.edge(u, v, label=op, fillcolor="gray")

        for i in range(steps):
            for k in [2 * i, 2 * i + 1]:
                op, j = genotype[k]
                u = f"C{cell_num}_s{j}"
                v = f"C{cell_num}_s{i+2}"
                g.edge(u, v, label=op, fillcolor="gray")

        g.node(f"C{cell_num}_out", fillcolor='palegoldenrod')

        for i in reduce_indexes:
            g.edge(f"C{cell_num}_s{i}", f"C{cell_num}_out", fillcolor="gray")

        gg.subgraph(g)

    gg.render(filename, view=view)
    return gg


def merge_species_and_param_settings(species_settings, randomized):
    n_settings = len(species_settings)
    res = []
    for i in range(n_settings):
        d = {}
        for key in randomized:
            d[key] = randomized[key][i]
        res.append(d)

    return np.concatenate([species_settings, [list(p.values()) for p in res]], -1)
