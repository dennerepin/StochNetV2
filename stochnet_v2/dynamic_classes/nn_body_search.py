import numpy as np
import tensorflow as tf

from stochnet_v2.dynamic_classes.op_registry import OP_REGISTRY
from stochnet_v2.dynamic_classes.op_registry import simple_dense as dense
# from stochnet_v2.dynamic_classes.op_registry import rich_dense as dense  # TODO: ?
from stochnet_v2.dynamic_classes.genotypes import Genotype
from stochnet_v2.dynamic_classes.genotypes import PRIMITIVES
from stochnet_v2.dynamic_classes.util import expand_cell
from stochnet_v2.dynamic_classes.util import l1_regularizer
from stochnet_v2.dynamic_classes.util import l2_regularizer

# null_scope = tf.compat.v1.VariableScope("")


def mixed_op(x, index, cell_index, expand, expansion_coeff):

    with tf.compat.v1.variable_scope("architecture_variables"):
        alphas = tf.compat.v1.get_variable(
            name=f"alpha_{cell_index}_{2 if expand else 1}_{index}",
            shape=[len(PRIMITIVES)],
            # initializer=tf.compat.v1.keras.initializers.random_normal,  # TODO:?
            initializer=tf.compat.v1.ones_initializer,
            trainable=True,
        )
        # alphas_regularizer = l2_regularizer(alphas)
        alphas_reg_loss = l1_regularizer(alphas, 0.005)
        tf.compat.v1.add_to_collection('architecture_regularization_losses', alphas_reg_loss)

    outputs = []

    alphas = tf.nn.softmax(alphas)

    for idx, primitive in enumerate(PRIMITIVES):

        out = OP_REGISTRY[primitive](x, expansion_coeff)
        mask = [idx == i for i in range(len(PRIMITIVES))]
        mask = tf.compat.v1.constant(mask, tf.bool)
        alpha = tf.compat.v1.boolean_mask(alphas, mask)

        outputs.append(alpha * out)

    out = tf.compat.v1.add_n(outputs)

    return out


def cell(
        s0,
        s1,
        cell_size,
        expand,
        expand_prev,
        expansion_multiplier,
        cell_index,
):

    with tf.compat.v1.variable_scope(f"{'expand' if expand else 'normal'}_cell_{cell_index}"):

        with tf.variable_scope("state_0"):
            if expand_prev:
                s0 = dense(s0, expansion_multiplier)
            else:
                s0 = dense(s0, 1)

        with tf.variable_scope("state_1"):
            s1 = dense(s1, 1)

        state = [s0, s1]
        offset = 0

        for i in range(cell_size):
            tmp = []
            for j in range(i + 2):
                expansion_coeff = expansion_multiplier if expand and j < 2 else 1
                with tf.compat.v1.variable_scope(f"mixed_op_{2 if expand else 1}_{offset + j}"):
                    mix = mixed_op(state[j], offset + j, cell_index, expand, expansion_coeff)
                tmp.append(mix)

            offset += len(state)
            with tf.variable_scope(f"state_{i+2}"):
                new_state = tf.add_n(tmp)
            state.append(new_state)

        out = state[-1]

    return out


def body(x, n_cells=4, cell_size=4, expansion_multiplier=4):
    out_dim = x.shape.as_list()[-1]
    s0 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    s1 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    expand_prev = False

    for n in range(n_cells):
        expand = expand_cell(n, n_cells)

        s0, s1 = s1, cell(s0, s1, cell_size, expand, expand_prev, expansion_multiplier, n)
        expand_prev = expand

    return s1


def get_genotypes(session, n_cells, cell_size):

    def _parse(_expand, cell_index):

        offset = 0
        result = []
        arch_variables = [
            v
            for v in session.graph.get_collection('variables')
            if 'architecture_variables' in v.name
        ]

        for i in range(cell_size):
            edges = []
            edges_alphas = []

            for j in range(i + 2):

                alpha_name = f"alpha_{cell_index}_{2 if _expand else 1}_{offset + j}:"
                alpha = [v for v in arch_variables if alpha_name in v.name]
                if len(alpha) != 1:
                    if len(alpha) > 1:
                        msg = f"Too many alphas ({alpha_name}) found: {alpha}, should be exactly one."
                    else:
                        msg = f"Alpha ({alpha_name}) not found."
                    raise ValueError(msg)
                alpha = alpha[0]

                alpha_value = session.run(alpha)
                value_sorted = alpha_value.argsort()
                max_index = \
                    value_sorted[-2] \
                    if value_sorted[-1] == PRIMITIVES.index('none') \
                    else value_sorted[-1]

                edges.append((PRIMITIVES[max_index], j))
                edges_alphas.append(alpha_value[max_index])

            edges_alphas = np.array(edges_alphas)
            max_edges = [edges[np.argsort(edges_alphas)[-1]], edges[np.argsort(edges_alphas)[-2]]]
            result.extend(max_edges)
            offset += i + 2

        return result

    genotypes = []

    for n in range(n_cells):
        expand = expand_cell(n, n_cells)
        gene = _parse(expand, n)
        if expand:
            genotype = Genotype(normal=(), expand=gene)
        else:
            genotype = Genotype(normal=gene, expand=())
        genotypes.append(genotype)

    return genotypes

