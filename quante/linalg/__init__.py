# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 18:27:12
# @Last Modified by:   hzhu
# @Last Modified time: 2024-11-22 02:03:17
"""
==================================
linalg (:mod:`quante.linalg`)
==================================

功能：

* 本征, 奇异值分解: `eig`, `eigvals`
* 矩阵运算、矩阵元素操作: `expm`, `entropy`, `kron`, `partial_trace`
* 演化工具: `expm_multiple`，`evolve_engine_spexpm`，`evolve_engine_eig`

.. currentmodule:: quante.linalg

本征值分解
------------
.. automodule:: quante.linalg.eig_modified
.. autofunction:: quante.linalg.perturbation.eigh_perturbation

奇异值分解
--------------
.. automodule:: quante.linalg.svd_robust

其它线性代数操作
----------------------

.. autofunction:: quante.linalg.operations.norm
.. autofunction:: quante.linalg.operations.expm
.. autofunction:: quante.linalg.operations.sqrtm
.. autofunction:: quante.linalg.operations.logm
.. autofunction:: quante.linalg.operations.kron
.. autofunction:: quante.linalg.operations.ikron
.. autofunction:: quante.linalg.operations.partial_trace


矩阵元操作
----------
.. autofunction:: quante.linalg.operations.uptrig
.. autofunction:: quante.linalg.operations.uptrig_inv
.. autofunction:: quante.linalg.operations.uptrigindex
.. autofunction:: quante.linalg.operations.uptrigindex_inv
.. autofunction:: quante.linalg.operations.observe_states


拟合与插值
------------
.. autofunction:: quante.linalg.operations.log_Gauss
.. autofunction:: quante.linalg.operations.find_boundary
.. autofunction:: quante.linalg.operations.interp
.. autofunction:: quante.linalg.operations.fit

演化
-----
.. automodule:: quante.linalg.evolve
   :members: evolve_engine_spexpm, get_time_evolution_states_ED, expm_multiply

krylov
-------
.. automodule:: quante.linalg.krylov
   :members: lanczos_ground_state, lanczos_evolve_state, lanczos_arpack
   
"""

from .eig_modified import *
from .svd_robust import *
from .operations import *
from .evolve import *
from .krylov import *
from .perturbation import *

