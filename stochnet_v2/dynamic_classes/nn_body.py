import tensorflow as tf

from stochnet_v2.dynamic_classes.op_registry import OP_REGISTRY
from stochnet_v2.dynamic_classes.op_registry import simple_dense as dense
# from stochnet_v2.dynamic_classes.op_registry import rich_dense as dense  # TODO: ?
from stochnet_v2.dynamic_classes.util import expand_cell


def cell(
        s0,
        s1,
        genotype,
        expand,
        expand_prev,
        expansion_multiplier,
        cell_index,
):
    if expand:
        op_names, indices = zip(*genotype.expand)
    else:
        op_names, indices = zip(*genotype.normal)

    cell_size = len(op_names) // 2

    with tf.compat.v1.variable_scope(f"{'expand' if expand else 'normal'}_cell_{cell_index}"):

        with tf.variable_scope("state_0"):
            if expand_prev:
                s0 = dense(s0, expansion_multiplier)
            else:
                s0 = dense(s0, 1)

        with tf.variable_scope("state_1"):
            s1 = dense(s1, 1)

        state = [s0, s1]

        for i in range(cell_size):
            tmp = []
            for j in range(2):
                output_state_idx = i + 2
                genotype_idx = 2 * i + j
                input_state_idx = indices[genotype_idx]
                expansion_coeff = expansion_multiplier if expand and input_state_idx < 2 else 1
                s = state[input_state_idx]
                with tf.compat.v1.variable_scope(f"mixed_op_{input_state_idx}_{output_state_idx}"):
                    out = OP_REGISTRY[op_names[genotype_idx]](s, expansion_coeff)
                tmp.append(out)

            with tf.variable_scope(f"state_{output_state_idx}"):
                new_state = tf.add_n(tmp)
            state.append(new_state)

        out = state[-1]

    return out


def body(x, genotypes, n_cells=4, expansion_multiplier=4):
    out_dim = x.shape.as_list()[-1]
    s0 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    s1 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    expand_prev = False

    for n in range(n_cells):
        expand = expand_cell(n, n_cells)
        s0, s1 = s1, cell(s0, s1, genotypes[n], expand, expand_prev, expansion_multiplier, n)
        expand_prev = expand

    return s1
