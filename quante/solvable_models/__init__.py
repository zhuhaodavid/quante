# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-09 18:57:14
# @Last Modified by:   hzhu
# @Last Modified time: 2024-11-09 02:45:53
"""

solvable_models (:mod:`quante.solvable_models`)
==================================================

一些可解模型

包括 free fermion 和 Anderson model

.. currentmodule:: quante.solvable_models

自由费米子
-----------

.. autofunction:: quante.solvable_models.free_fermion.XY_gdenergy
.. autofunction:: quante.solvable_models.free_fermion.XY_energies
.. autofunction:: quante.solvable_models.free_fermion.XY_free_energy
.. autofunction:: quante.solvable_models.free_fermion.XY_internal_energy
.. autofunction:: quante.solvable_models.free_fermion.XY_specific_heat
.. autofunction:: quante.solvable_models.free_fermion.XY_magnetization
.. autofunction:: quante.solvable_models.free_fermion.XXX_gdenergy_pbc_approx
.. autofunction:: quante.solvable_models.free_fermion.XXZ_gdenergy_inf
    
Anderson 模型
----------------

.. autofunction:: quante.solvable_models.anderson_model.anderson_matrix
.. autofunction:: quante.solvable_models.anderson_model.anderson_kmat
.. autofunction:: quante.solvable_models.anderson_model.anderson_eigstate
.. autofunction:: quante.solvable_models.anderson_model.anderson_energies
.. autofunction:: quante.solvable_models.anderson_model.plot_anderson_band


"""

from . import free_fermion
from . import anderson_model

__all__ = ['free_fermion', 'anderson_model']