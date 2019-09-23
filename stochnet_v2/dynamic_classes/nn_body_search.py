import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from stochnet_v2.dynamic_classes.op_registry import OP_REGISTRY
from stochnet_v2.dynamic_classes.op_registry import simple_dense as dense  # TODO: ?
# from stochnet_v2.dynamic_classes.op_registry import dense_relu as dense  # TODO: ?
# from stochnet_v2.dynamic_classes.op_registry import bn_dense_relu as dense  # TODO: ?
# from stochnet_v2.dynamic_classes.op_registry import relu_dense_bn as dense  # TODO: ?
from stochnet_v2.dynamic_classes.genotypes import Genotype
from stochnet_v2.dynamic_classes.genotypes import PRIMITIVES
from stochnet_v2.dynamic_classes.util import expand_cell
from stochnet_v2.dynamic_classes.util import l1_regularizer
from stochnet_v2.dynamic_classes.util import l2_regularizer


tfd = tfp.distributions


def mixed_op(x, expansion_coeff):

    with tf.compat.v1.variable_scope("architecture_variables"):
        alphas = tf.compat.v1.get_variable(
            name=f"alphas",
            shape=[len(PRIMITIVES)],
            initializer=tf.compat.v1.ones_initializer,
            trainable=True,
        )

        alphas = tf.nn.softmax(alphas)
        # reg_scale = float(len(PRIMITIVES))
        reg_scale = 1.0
        alphas_reg_loss = - l2_regularizer(alphas, reg_scale)
        tf.compat.v1.add_to_collection('architecture_regularization_losses', alphas_reg_loss)

    outputs = []
    for idx, primitive in enumerate(PRIMITIVES):
        out = OP_REGISTRY[primitive](x, expansion_coeff)
        alpha = alphas[idx]
        outputs.append(alpha * out)

    out = tf.compat.v1.add_n(outputs)

    return out


@tf.custom_gradient
def cat_onehot(a):
    categorical = tfd.Categorical(probs=a)
    cat = categorical.sample()
    one_hot = tf.one_hot(cat, depth=a.shape.as_list()[0])

    def grad(dy):
        return dy * one_hot * a

    return one_hot, grad


def mixed_op_cat(x, expansion_coeff):

    with tf.compat.v1.variable_scope("architecture_variables"):
        alphas = tf.compat.v1.get_variable(
            name=f"alphas",
            shape=[len(PRIMITIVES)],
            initializer=tf.compat.v1.ones_initializer,
            trainable=True,
        )

        alphas = tf.nn.softmax(alphas)
        alphas_reg_loss = - l2_regularizer(alphas, 0.01)
        tf.compat.v1.add_to_collection('architecture_regularization_losses', alphas_reg_loss)

    outputs = []
    for idx, primitive in enumerate(PRIMITIVES):
        tmp = OP_REGISTRY[primitive](x, expansion_coeff)
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
        n_summ_states=2,
):

    with tf.compat.v1.variable_scope(f"{'expand' if expand else 'normal'}_cell_{cell_index}"):

        prev_multiplier = 1
        if expand_prev:
            prev_multiplier *= expansion_multiplier
        if expand:
            prev_multiplier *= expansion_multiplier

        curr_multiplier = expansion_multiplier if expand else 1

        with tf.variable_scope("state_0"):
            s0 = dense(s0, prev_multiplier)

        with tf.variable_scope("state_1"):
            s1 = dense(s1, curr_multiplier)

        state = [s0, s1]

        for i in range(cell_size):
            tmp = []
            for j in range(i + 2):
                # expansion_coeff = expansion_multiplier if expand and j < 2 else 1
                expansion_coeff = 1
                with tf.compat.v1.variable_scope(f"mixed_op_{j}_{i + 2}"):
                    mix = mixed_op_cat(state[j], expansion_coeff)
                tmp.append(mix)

            with tf.variable_scope(f"state_{i + 2}"):
                new_state = tf.add_n(tmp)
            state.append(new_state)

        out = tf.compat.v1.add_n(state[-n_summ_states:])

    return out


def body(x, n_cells=4, cell_size=4, expansion_multiplier=4, n_summ_states=2):
    # out_dim = x.shape.as_list()[-1]
    # s0 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    # s1 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    s0 = tf.compat.v1.identity(x)
    s1 = tf.compat.v1.identity(x)
    expand_prev = False

    for n in range(n_cells):
        expand = expand_cell(n, n_cells)
        s0, s1 = s1, cell(s0, s1, cell_size, expand, expand_prev, expansion_multiplier, n, n_summ_states)
        expand_prev = expand

    return s1


def get_genotypes(session, n_cells, cell_size, n_summ_states):

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
            print(f"STATE {i+2}:")

            for j in range(i + 2):

                alpha_name = f"{'expand' if _expand else 'normal'}_cell_{cell_index}/mixed_op_{j}_{i + 2}"
                alpha = [v for v in arch_variables if alpha_name in v.name and 'alphas:' in v.name]
                if len(alpha) != 1:
                    if len(alpha) > 1:
                        msg = f"Too many alphas ({alpha_name}) found: {alpha}, should be exactly one."
                    else:
                        msg = f"Alpha ({alpha_name}) not found."
                    raise ValueError(msg)
                alpha = alpha[0]

                alpha_value = session.run(alpha)
                print(f"\t{alpha_name}: {alpha_value}")

                alpha_value_argsort = alpha_value.argsort()

                # TODO: if we keep zero-connections as candidates, we may obtain much smaller graphs,
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
                print(f"\t{j}-th state via {edge_candidate.upper()}"
                      f" (max_index={max_index}, score={edge_score:.2f})")
                edges.append((edge_candidate, j))
                edges_scores.append(edge_score)

            edges_scores = np.array(edges_scores)
            print(f"edges_scores = {edges_scores}")

            edges_scores_argsort = np.argsort(edges_scores)[::-1]  # decreasing order
            edges_sorted = [edges[i] for i in edges_scores_argsort]
            max_edges = edges_sorted[:2]

            print(f"max_edges = {max_edges}\n")
            result.extend(max_edges)

        return result

    genotypes = []

    for n in range(n_cells):
        expand = expand_cell(n, n_cells)
        summ_states = list(range(cell_size + 2 - n_summ_states, cell_size + 2))
        gene = _parse(expand, n)
        if expand:
            genotype = Genotype(
                normal=(), normal_summ=(),
                expand=gene, expand_summ=summ_states
            )
        else:
            genotype = Genotype(
                normal=gene, normal_summ=summ_states,
                expand=(), expand_summ=()
            )
        genotypes.append(genotype)

    return genotypes

