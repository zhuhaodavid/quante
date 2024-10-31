# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-09 18:57:14
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 20:45:00
"""

solvable_models (:mod:`quante.solvable_models`)
==================================================

一些可解模型

包括 free fermion 和 Anderson model

.. currentmodule:: quante.solvable_models

自由费米子
-----------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    free_fermion.XY_gdenergy
    free_fermion.XY_energies
    free_fermion.XY_free_energy
    free_fermion.XY_internal_energy
    free_fermion.XY_specific_heat
    free_fermion.XY_magnetization
    free_fermion.XXX_gdenergy_pbc_approx
    free_fermion.XXZ_gdenergy_inf
    
Anderson 模型
----------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   
    anderson_model.anderson_matrix
    anderson_model.anderson_kmat
    anderson_model.anderson_eigstate
    anderson_model.anderson_energies
    anderson_model.plot_anderson_band

一些可解模型

包括 free fermion 和 Anderson model

"""

from . import free_fermion
from . import anderson_model

__all__ = ['free_fermion', 'anderson_model']