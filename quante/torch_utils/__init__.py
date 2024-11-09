# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-09 19:28:03
# @Last Modified by:   hzhu
# @Last Modified time: 2024-11-09 02:53:09

"""
============================================
torch_utils (:mod:`quante.torch_utils`)
============================================

`torch` 工具包，包含了一些常用的 `torch` 工具函数。

包括 `linalg` 模块，包含了一些线性代数相关的函数；`tensor_network` 模块，包含了张量网络相关的函数。

.. currentmodule:: quante.torch_utils

分解操作
-----------

.. autofunction:: quante.torch_utils.linalg.truncate
.. autofunction:: quante.torch_utils.linalg.svd
.. autofunction:: quante.torch_utils.linalg.qr
.. autofunction:: quante.torch_utils.linalg.rq
.. autofunction:: quante.torch_utils.linalg.eigh
.. autofunction:: quante.torch_utils.linalg.tt_decompose


指数作用
-----------
.. autofunction:: quante.torch_utils.linalg.expm_multiply
.. autofunction:: quante.torch_utils.linalg.evolve_engine

稀疏工具
----------
.. autofunction:: quante.torch_utils.sparse.to_csr
.. autofunction:: quante.torch_utils.sparse.eye
.. autofunction:: quante.torch_utils.sparse.trace
.. autofunction:: quante.torch_utils.sparse.norm

梯度工具
----------
.. autoclass:: quante.torch_utils.utils.AdaptiveLRScheduler

.. autofunction:: quante.torch_utils.utils.open_grad
.. autofunction:: quante.torch_utils.utils.close_grad
.. autofunction:: quante.torch_utils.utils.to_numpy_array
.. autofunction:: quante.torch_utils.utils.convert_to_torch
.. autofunction:: quante.torch_utils.utils.clone_list


张量网络工具
--------------
.. autoclass:: quante.torch_utils.tensor_network.MPS
.. autoclass:: quante.torch_utils.tensor_network.MPO

.. autofunction:: quante.torch_utils.tensor_network.mpo_eye
.. autofunction:: quante.torch_utils.tensor_network.full_contract
.. autofunction:: quante.torch_utils.tensor_network.tn_inner
.. autofunction:: quante.torch_utils.tensor_network.tn_norm
.. autofunction:: quante.torch_utils.tensor_network.canonicalize
.. autofunction:: quante.torch_utils.tensor_network.orthogonalize
.. autofunction:: quante.torch_utils.tensor_network.add
.. autofunction:: quante.torch_utils.tensor_network.canonicalize_infinite
.. autofunction:: quante.torch_utils.tensor_network.periodic_trace

"""
# 使用： import quante.torch_utils as qtc

from . import sparse
from . import linalg
from . import tensor_network

# 这两个为了方便使用
from .utils import *
from .tensor_network import *

