# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 20:34:58
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-06 12:37:34

from ..bitsoperation import operateon
from ......basicfun.utils_numba import config, numba_cache_dir, pnjit, prange
import numpy as np

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
        opco, t = operateon(opnm, posn, a, L)
        if t < 0:
            diag[a] = 0.0
        else:
            diag[a] = opco * coef
    return diag
