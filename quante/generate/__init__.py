# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 17:55:56
# @Last Modified by:   hzhu
# @Last Modified time: 2025-02-02 14:55:58
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

from . import basis
from . import operas

from . import matrix
from . import state

from .basis import spin_basis
from .matrix import pauli_matrix, PAULI_MAT