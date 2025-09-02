# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 20:34:11
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-02 18:31:09

import numpy as _np
from .pauli import pauli_matrix

def hadamard_gate(dtype=complex):
    """The Hadamard gate."""
    return _np.array([[1., 1.],[1., -1.]], dtype=dtype) / 2**0.5

def phase_gate(phi=0.0, dtype=complex):
    """The generalized qubit phase-gate."""
    return _np.array([[1., 0.],[0., _np.exp(1.0j * phi)]], dtype=dtype)

def rotation_gate(phi, xyz='Z', dtype=complex):
    """The single qubit rotation gate."""
    R = _np.cos(phi / 2) * pauli_matrix('I') - 1.0j * _np.sin(phi / 2) * pauli_matrix(xyz)
    return _np.array(R, dtype=dtype)

