# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-12 15:04:23
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-09 19:08:37
# setup.py

from setuptools import setup, find_packages

setup(
    name='quante',
    version='0.2.0',
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