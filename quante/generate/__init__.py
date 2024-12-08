# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 17:55:56
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-08 16:00:07
"""
.. currentmodule:: quante.generate

generate (:mod:`quante.generate`)
=======================================

operas
-------------------------------------
.. automodule:: quante.generate.operas
    :exclude-members: I, p, m, x, y, z, n, nn, zz, mp, pm, xx, yy, xy, yx

basis
-------------------------------------
.. automodule:: quante.generate.basis

matrix
-------------------------------------
.. automodule:: quante.generate.matrix


state
-------------------------------------
.. automodule:: quante.generate.state
"""

from . import operas
from . import basis
from . import matrix
from . import state

from .basis import spin_basis
from .matrix import pauli_matrix, PAULI_MAT