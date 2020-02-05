from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_reduce expand expand_reduce')

PRIMITIVES = [
    'simple_dense',
    # 'activated_dense',
    # 'element_wise',
    'gated_linear_unit',
    'skip_connect',
    'relu',
    # 'swish',
    'none',
]
