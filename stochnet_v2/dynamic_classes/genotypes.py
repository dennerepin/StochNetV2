from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_summ expand expand_summ')

PRIMITIVES = [
    # 'simple_dense',
    'rich_dense_0',
    # 'rich_dense_1',
    # 'rich_dense_2',
    # 'element_wise',
    'skip_connect',
    'none',
]
