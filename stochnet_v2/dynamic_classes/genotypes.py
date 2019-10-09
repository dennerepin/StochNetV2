from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_summ expand expand_summ')

PRIMITIVES = [
    'simple_dense',  #
    'activated_dense',  #+
    'element_wise',  #+
    'gated_linear_unit',  #
    'skip_connect',  #!
    'none',  #+
    # 'relu',
    # 'swish',
]
