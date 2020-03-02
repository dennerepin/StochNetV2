import os
from itertools import takewhile
from setuptools import setup
from setuptools import find_packages


_HERE = os.path.abspath(os.path.dirname(__file__))
_REQUIREMENTS_PATH = os.path.join(_HERE, 'requirements.txt')


def _get_requirements():
    with open(_REQUIREMENTS_PATH, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
        lines = [''.join(takewhile(lambda c: c != ' ', line)) for line in lines]
    return lines


setup(
    name='StochNetV2',
    version='1.0.0',
    description='Package implements workflow for approximation dynamics '
                'of reaction networks with muxture density networks',
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
    packages=find_packages(where=_HERE, exclude=[]),
    package_data={
        "stochnet_v2": [
            "logging.conf"
        ]},
    install_requires=_get_requirements(),
    python_requires='>=3.6',
)
