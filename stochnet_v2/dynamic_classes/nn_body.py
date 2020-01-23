import tensorflow as tf

from stochnet_v2.dynamic_classes.op_registry import OP_REGISTRY
from stochnet_v2.dynamic_classes.nn_body_search import expand_op
from stochnet_v2.dynamic_classes.util import cell_is_expanding


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
        states_reduce = genotype.expand_reduce
    else:
        op_names, indices = zip(*genotype.normal)
        states_reduce = genotype.normal_reduce

    cell_size = len(op_names) // 2

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

            for j in range(2):
                genotype_idx = 2 * i + j
                input_state_idx = indices[genotype_idx]
                expansion_coeff = 1

                s = state[input_state_idx]
                if s != -1:
                    op_candidate_name = op_names[genotype_idx]
                    if op_candidate_name != 'none':
                        with tf.compat.v1.variable_scope(f"mixed_op_{input_state_idx}_{output_state_idx}"):
                            out = OP_REGISTRY[op_candidate_name](s, expansion_coeff)
                        tmp.append(out)

            if len(tmp) > 0:
                with tf.variable_scope(f"state_{output_state_idx}"):
                    new_state = tf.add_n(tmp)
                state.append(new_state)
            else:
                state.append(-1)

        # out = next((x for x in state[::-1] if x != -1), s0 + s1)
        reduce_candidates = []
        for idx in states_reduce:
            candidate = state[idx]
            if candidate != -1:
                reduce_candidates.append(candidate)
        if len(reduce_candidates) == 0:
            reduce_candidates = [s0, s1]

        out = tf.compat.v1.reduce_mean(reduce_candidates, 0)

    return out


def body(x, genotypes, expansion_multiplier):
    n_cells = len(genotypes)
    # out_dim = x.shape.as_list()[-1]
    # s0 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    # s1 = tf.compat.v1.layers.Dense(out_dim, activation='relu')(x)
    s0 = tf.compat.v1.identity(x)
    s1 = tf.compat.v1.identity(x)
    expand_prev = False

    for n in range(n_cells):
        expand = cell_is_expanding(n, n_cells)
        s0, s1 = s1, cell(s0, s1, genotypes[n], expand, expand_prev, expansion_multiplier, n)
        expand_prev = expand

    return s1
