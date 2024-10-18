# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-12 15:04:23
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-06 17:57:56
# setup.py

from setuptools import setup, find_packages

setup(
    name='quante',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        "h5py",
        "numba",
        "numpy",
        "scipy",
        "cytoolz",
        "psutil"
    ],
    include_package_data=True,
)