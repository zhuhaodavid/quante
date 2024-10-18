# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 18:27:12
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-07 00:10:03
"""
功能：
- 本征, 奇异值分解: `eig`, `eigvals`
- 矩阵运算、矩阵元素操作: `expm`, `entropy`, `kron`, `partial_trace`
- 演化工具: `expm_multiple`，`evolve_engine_spexpm`，`evolve_engine_eig`
"""

from .eig_modified import *
from .svd_robust import *
from .operations import *
from .evolve import *
from .krylov import *

