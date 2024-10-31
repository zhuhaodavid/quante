# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-09 19:28:03
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 20:46:12

"""
============================================
torch_utils (:mod:`quante.torch_utils`)
============================================

`torch` 工具包，包含了一些常用的 `torch` 工具函数。

包括 `linalg` 模块，包含了一些线性代数相关的函数；`tensor_network` 模块，包含了张量网络相关的函数。

分解操作
-----------
.. autosummary::
   :toctree: _autosummary

   linalg.truncate
   linalg.svd
   linalg.qr
   linalg.rq
   linalg.eigh
   linalg.tt_decompose

指数作用
-----------
.. autosummary::
   :toctree: _autosummary
   
   linalg.expm_multiply
   linalg.evolve_engine

稀疏工具
----------
.. autosummary:: 
   :toctree: _autosummary
   
   sparse.to_csr
   sparse.eye
   sparse.trace
   sparse.norm

梯度工具
----------
.. autosummary::
   :toctree: _autosummary

    AdaptiveLRScheduler
    open_grad
    close_grad
    to_numpy_array
    convert_to_torch
    clone_list
    save_h5
    load_h5

张量网络工具
--------------
.. autosummary::
   :toctree: _autosummary

    MPS
    MPO
    mpo_eye
    full_contract
    tn_inner
    tn_norm
    canonicalize
    orthogonalize
    add
    canonicalize_infinite
    periodic_trace
"""
# 使用： import quante.torch_utils as qtc

from . import sparse
from . import linalg
from . import tensor_network

# 这两个为了方便使用
from .utils import *
from .tensor_network import *

