# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-12 15:04:23
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-13 23:44:39
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
        "tqdm",
        "line_profiler",
        "dowhen",
    ],
    include_package_data=True,
)
