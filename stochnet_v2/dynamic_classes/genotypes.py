from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_summ expand expand_summ')

PRIMITIVES = [
    'simple_dense',
    'dense_relu',
    # 'bn_dense_relu',
    # 'relu_dense_bn',
    'element_wise',
    'skip_connect',
    'none',
]
