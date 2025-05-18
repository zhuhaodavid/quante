# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2023-10-22 18:27:12
# @Last Modified by:   dzwang
# @Last Modified time: 2025-04-19 14:59:30
"""
==================================
tensor (:mod:`quante.tensor`)
==================================

numpy 张量网络
"""

# Basic building blocks for tensor network operations.
from .linalg import *
from .tensortrain import *

# Higher-level modules for specific tensor network models and algorithms.
from . import networks
from . import algorithms
