# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 20:34:58
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-30 17:35:46

from ..bitsoperation import operateon, flip
from ......linalg.usenumba.numba_settings import config, numba_cache_dir, pnjit, prange
import numpy as np


config.CACHE_DIR = numba_cache_dir
@pnjit('f8[:,:](i8,f8,f8,b1)')
def heisenberg_matrix_element(L, jxy, jz, cyclic):
    r"""
    \sum_{i=1}^L jxy * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1}) + hz * s^z_i s^z_{i+1}
    
    Examples
    >>> L = 6
    >>> mat = heisenberg_chain_noblock(L, jxy=1., jz=1.)
    """
    H = np.zeros((1<<L, 1<<L), dtype=float)
    last_site = L if cyclic else L - 1
    for a in prange(1<<L):
        for i in range(last_site):
            j = (i+1) % L
            if ((a >> i) & 1) == ((a >> j) & 1):
                H[a,a] += 0.25 * jz
            else:
                H[a,a] -= 0.25 * jz
                b = flip(a, i, j)
                if b != -1:
                    H[b,a] += 0.5 * jxy
    return H

config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element(opnm, posn, coef, L, M, row_init, col_init, ME_init):
    ME_init.fill(0.0)
    
    for a in prange(M):
        
        opco, t = operateon(opnm, posn, a, L)
        if t != -1:
            row_init[a] = t
            col_init[a] = a
            ME_init[a] = opco * coef

    mask = np.logical_not(np.abs(ME_init)==0.0)
    
    return row_init[mask], col_init[mask], ME_init[mask]

config.CACHE_DIR = numba_cache_dir
@pnjit
def diag_matrix_element(opnm, posn, coef, L, M, dtype):
    diag = np.empty(M, dtype=dtype)
    for a in prange(M):
        opco, _ = operateon(opnm, posn, a, L)
        diag[a] = opco * coef
    return diag
