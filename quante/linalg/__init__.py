# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 18:27:12
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 03:35:02
"""
==================================
linalg (:mod:`quante.linalg`)
==================================

功能：

* 本征, 奇异值分解: `eig`, `eigvals`
* 矩阵运算、矩阵元素操作: `expm`, `entropy`, `kron`, `partial_trace`
* 演化工具: `expm_multiple`，`evolve_engine_spexpm`，`evolve_engine_eig`

本征值分解
------------
.. autosummary::
   :toctree: _autosummary

   eig
   eigh
   eigvals
   eigvalsh
   eighbetween

奇异值分解
--------------
.. autosummary::
   :toctree: _autosummary

   svd
   truncate
   svd_truncate
   TruncationError

其它线性代数操作
----------------------
.. autosummary::
   :toctree: _autosummary

   norm
   expm
   sqrtm
   logm
   kron
   ikron
   partial_trace

矩阵元操作
----------
.. autosummary::
   :toctree: _autosummary

   uptrig
   uptrig_inv
   uptrigindex
   uptrigindex_inv
   observe_states

拟合与插值
------------
.. autosummary::
   :toctree: _autosummary

   log_Gauss
   find_boundary
   interp
   fit

演化
-----
.. autosummary::
   :toctree: _autosummary

   evolve_engine_spexpm
   get_time_evolution_states_ED
   expm_multiply

krylov
-------
.. autosummary::
   :toctree: _autosummary

   lanczos_ground_state
   lanczos_evolve_state
   lanczos_arpack
"""

from .eig_modified import *
from .svd_robust import *
from .operations import *
from .evolve import *
from .krylov import *

