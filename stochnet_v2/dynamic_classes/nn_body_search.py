import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from stochnet_v2.dynamic_classes.op_registry import OP_REGISTRY
# from stochnet_v2.dynamic_classes.op_registry import simple_dense as expand_op
# from stochnet_v2.dynamic_classes.op_registry import _expand_identity as expand_op
from stochnet_v2.dynamic_classes.op_registry import simple_dense
from stochnet_v2.dynamic_classes.op_registry import _expand_identity
from stochnet_v2.dynamic_classes.genotypes import Genotype
from stochnet_v2.dynamic_classes.genotypes import PRIMITIVES
from stochnet_v2.dynamic_classes.util import cell_is_expanding
from stochnet_v2.dynamic_classes.util import l1_regularizer
from stochnet_v2.dynamic_classes.util import l2_regularizer


tfd = tfp.distributions

LOGGER = logging.getLogger('dynamic_classes.nn_body_search')


def expand_op(x, multiplier):
    if multiplier != 1.:
        return simple_dense(x, multiplier)
    return _expand_identity(x, multiplier)


def mixed_op(x, expansion_coeff, **kwargs):

    with tf.compat.v1.variable_scope("architecture_variables"):
        alphas = tf.compat.v1.get_variable(
            name=f"alphas",
            shape=[len(PRIMITIVES)],
            initializer=tf.compat.v1.zeros_initializer,
            trainable=True,
        )

        alphas = tf.nn.softmax(alphas)
        alphas_reg_loss = - l2_regularizer(alphas, float(len(PRIMITIVES)))
        tf.compat.v1.add_to_collection('architecture_regularization_losses', alphas_reg_loss)

    outputs = []
    for idx, primitive in enumerate(PRIMITIVES):
        out = OP_REGISTRY[primitive](x, expansion_coeff, **kwargs)

        # alpha = alphas[idx]

        mask = [i == idx for i in range(len(PRIMITIVES))]
        alphas_mask = tf.constant(mask, tf.bool)
        alpha = tf.boolean_mask(alphas, alphas_mask)

        outputs.append(alpha * out)

    out = tf.compat.v1.add_n(outputs)

    return out


@tf.custom_gradient
def cat_onehot(a):
    categorical = tfd.Categorical(probs=a)
    cat = categorical.sample()
    one_hot = tf.one_hot(cat, depth=a.shape.as_list()[0])

    def grad(dy):
        return dy * one_hot  # * a

    return one_hot, grad


def mixed_op_cat(x, expansion_coeff, **kwargs):

    with tf.compat.v1.variable_scope("architecture_variables"):
        alphas = tf.compat.v1.get_variable(
            name=f"alphas",
            shape=[len(PRIMITIVES)],
            initializer=tf.compat.v1.zeros_initializer,
            trainable=True,
        )
        alphas = tf.nn.softmax(alphas)
        alphas_reg_loss = - l2_regularizer(alphas, 0.005)
        tf.compat.v1.add_to_collection('architecture_regularization_losses', alphas_reg_loss)

    outputs = []
    for idx, primitive in enumerate(PRIMITIVES):
        tmp = OP_REGISTRY[primitive](x, expansion_coeff, **kwargs)
        outputs.append(tmp)

    out = tf.compat.v1.stack(outputs, axis=0)

    one_hot = cat_onehot(alphas)
    one_hot = tf.expand_dims(tf.expand_dims(one_hot, -1), -1)
    out = one_hot * out
    out = tf.reduce_sum(out, axis=0)

    return out


def cell(
        s0,
        s1,
        cell_size,
        expand,
        expand_prev,
        expansion_multiplier,
        cell_index,
        n_states_reduce,
        **kwargs,
):

    with tf.compat.v1.variable_scope(f"{'expand' if expand else 'normal'}_cell_{cell_index}"):

        prev_multiplier = 1
        if expand_prev:
            prev_multiplier *= expansion_multiplier
        if expand:
            prev_multiplier *= expansion_multiplier

        curr_multiplier = expansion_multiplier if expand else 1

        with tf.variable_scope("state_0"):
            s0 = expand_op(s0, prev_multiplier)

        with tf.variable_scope("state_1"):
            s1 = expand_op(s1, curr_multiplier)

        state = [s0, s1]

        for i in range(cell_size):
            tmp = []
            output_state_idx = i + 2
            for j in range(output_state_idx):
                expansion_coeff = 1
                with tf.compat.v1.variable_scope(f"mixed_op_{j}_{output_state_idx}"):
                    # mixed_op makes it harder for the final (pruned) model
                    # to perform the same way: layer activations are different
                    # because of weighted sum of op candidates;
                    # mixed_op_cat avoids this by picking only one edge per time
                    # mix = mixed_op(state[j], expansion_coeff, **kwargs)
                    mix = mixed_op_cat(state[j], expansion_coeff, **kwargs)
                tmp.append(mix)

            with tf.variable_scope(f"state_{output_state_idx}"):
                new_state = tf.add_n(tmp)
            state.append(new_state)

        out = tf.compat.v1.reduce_mean(state[-n_states_reduce:], 0)

    return out


def body(
        x,
        n_cells,
        cell_size,
        expansion_multiplier,
        n_states_reduce,
        **kwargs
):
    # out_dim = x.shape.as_list()[-1]
    # s0 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    # s1 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    s0 = tf.compat.v1.identity(x)
    s1 = tf.compat.v1.identity(x)
    expand_prev = False

    for n in range(n_cells):
        expand = cell_is_expanding(n, n_cells)
        s0, s1 = s1, cell(
            s0,
            s1,
            cell_size=cell_size,
            expand=expand,
            expand_prev=expand_prev,
            expansion_multiplier=expansion_multiplier,
            cell_index=n,
            n_states_reduce=n_states_reduce,
            **kwargs
        )
        expand_prev = expand

    return s1


def get_genotypes(
        session,
        n_cells,
        cell_size,
        n_states_reduce,
):

    def _parse(_expand, cell_index):

        result = []
        arch_variables = [
            v
            for v in session.graph.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            if 'architecture_variables' in v.name
        ]

        for i in range(cell_size):
            edges = []
            edges_scores = []
            output_state_idx = i + 2
            LOGGER.debug(f"STATE {output_state_idx}:")

            for j in range(output_state_idx):

                alpha_name = f"{'expand' if _expand else 'normal'}_cell_{cell_index}/mixed_op_{j}_{output_state_idx}"
                alpha = [v for v in arch_variables if alpha_name in v.name and 'alphas:' in v.name]
                if len(alpha) != 1:
                    if len(alpha) > 1:
                        msg = f"Too many alphas ({alpha_name}) found: {alpha}, should be exactly one."
                    else:
                        msg = f"Alpha ({alpha_name}) not found."
                    raise ValueError(msg)
                alpha = alpha[0]

                alpha_value = session.run(alpha)
                LOGGER.debug(f"\t{alpha_name}: {alpha_value}")

                alpha_value_argsort = alpha_value.argsort()

                # if we keep zero-connections as candidates, we may obtain much smaller graphs,
                # as it removes a some of connections. BUT when more than one zero-connections
                # have large scores they can dominate all other candidate edges which results
                # in no connection to the state from any another
                # max_index = alpha_value_argsort[-1]

                max_index = \
                    alpha_value_argsort[-1] \
                    if alpha_value_argsort[-1] != PRIMITIVES.index('none') \
                    else alpha_value_argsort[-2]

                edge_candidate = PRIMITIVES[max_index]
                edge_score = alpha_value[max_index]
                LOGGER.debug(f"\t{j}-th state via {edge_candidate.upper()}"
                      f" (max_index={max_index}, score={edge_score:.2f})")
                edges.append((edge_candidate, j))
                edges_scores.append(edge_score)

            edges_scores = np.array(edges_scores)
            LOGGER.debug(f"edges_scores = {edges_scores}")

            edges_scores_argsort = np.argsort(edges_scores)[::-1]  # decreasing order
            edges_sorted = [edges[i] for i in edges_scores_argsort]
            max_edges = edges_sorted[:2]

            LOGGER.debug(f"max_edges = {max_edges}\n")
            result.extend(max_edges)

        return result

    genotypes = []

    for n in range(n_cells):
        expand = cell_is_expanding(n, n_cells)
        states_reduce = list(range(cell_size + 2 - n_states_reduce, cell_size + 2))
        gene = _parse(expand, n)
        if expand:
            genotype = Genotype(
                normal=(), normal_reduce=(),
                expand=gene, expand_reduce=states_reduce
            )
        else:
            genotype = Genotype(
                normal=gene, normal_reduce=states_reduce,
                expand=(), expand_reduce=()
            )
        genotypes.append(genotype)

    return genotypes

