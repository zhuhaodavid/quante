# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 17:55:56
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 01:29:26
"""
==================================
genetate (:mod:`quante.generate`)
==================================

功能：
- 生成哈密顿量, `Oper`, `heisenberg_operator`
- 生成基 `spin_basis`, `show_spin_basis`
- 生成矩阵 `pauli_matrix` `heisenberg_matrix` `random_matrix`
- 生成态 `state_random` `state_bloch`
"""

from . import operas
from . import basis
from . import matrix
from . import state