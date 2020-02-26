import os
from setuptools import setup
from setuptools import find_packages


here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='StochNetV2',
    version='0.0.1',
    description='Package implements workflow for approximation dynamics '
                'of reaction networks with muxture density networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/dennerepin/StochNetV2',
    author='Denis Repin',
    author_email='',
    license='GPL',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GPL License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='probabilistic models',
    packages=find_packages(where=here, exclude=[]),
    install_requires=[
        'bidict',
        'gillespy',
        'graphviz',
        'h5py',
        'libsbml',
        'luigi',
        'matplotlib',
        'numpy>=1.16',
        'scikit-learn',
        'tensorflow==1.15.2',
        'tensorflow-probability==0.7.0',
        'tqdm',
    ],
    python_requires='>=3.6',
)
