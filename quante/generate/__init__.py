# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 17:55:56
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 20:40:51
"""
==================================
genetate (:mod:`quante.generate`)
==================================

功能：
- 生成哈密顿量, `Oper`, `heisenberg_operator`
- 生成基 `spin_basis`, `show_spin_basis`
- 生成矩阵 `pauli_matrix` `heisenberg_matrix` `random_matrix`
- 生成态 `state_random` `state_bloch`

生成算符
-----------

内置的算符包括：`I`, `p`, `m`, `x`, `y`, `z`, `n`, `nn`, `zz`, `mp`, `pm`, `xx`, `yy`, `xy`, `yx`

.. autosummary::
   :toctree: _autosummary

   operas.Oper
   operas.heisenberg_operator
   operas.sum

生成基矢
------------
.. autosummary::
   :toctree: _autosummary

   basis.spin_basis
   basis.show_spin_basis

生成矩阵
------------
.. autosummary::
   :toctree: _autosummary

    matrix.pauli_matrix
    matrix.random_matrix
    matrix.random_orthorgonal_matrix_close_I
    matrix.random_unitary_matrix_close_I
    matrix.random_sparse_matrix
    matrix.hadamard_gate
    matrix.phase_gate
    matrix.rotation_gate
    matrix.heisenberg_matrix
    matrix.random_phase_model
    matrix.local_hamiltonian_spin_1D


生成量子态
------------
.. autosummary::
   :toctree: _autosummary

    state.spin_up
    state.spin_down
    state.basis_state
    state.xplus
    state.xminus
    state.plus
    state.minus
    state.yplus
    state.yminus
    state.bloch
    state.bell
    state.singlet
    state.thermal
    state.product_state
    state.neel
    state.singlet_pairs
    state.werner
    state.ghz
    state.w
    state.antisymmetric
    state.random
    state.random_sparse

"""

from . import operas
from . import basis
from . import matrix
from . import state