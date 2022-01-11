#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
# from distutils.core import setup
import os
import stat
import shutil
import platform
import sys
import site
import glob


# -- file paths --
long_description="""HIDS: hypothesis testing influence functions for data subsampling"""
setup(
    name='hids',
    version='1.0.0',
    description='A python implementation of HIDS',
    long_description=long_description,
    url='https://github.com/gauenk/hids',
    author='Kent Gauen',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords='hypothesis testing, influence functions, data subsampling',
    install_requires=['numpy','torch','flake8'],
    packages=find_packages(),
)
