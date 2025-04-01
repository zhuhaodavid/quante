# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-12 15:04:23
# @Last Modified by:   hzhu
# @Last Modified time: 2025-04-05 01:54:24
# setup.py

from setuptools import setup, find_packages

setup(
    name='quante',
    version='0.3.0',
    packages=find_packages(),
    install_requires=[
        "numba",
        "numpy",
        "scipy",
        "h5py",
        "cytoolz",
        "psutil",
        "objprint",
        "quspin-extensions",
        "tqdm",
        "line_profiler"
    ],
    include_package_data=True,
)
