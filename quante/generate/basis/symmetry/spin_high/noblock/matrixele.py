# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 20:34:58
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-30 12:39:15

from ......linalg.usenumba.numba_settings import njit, config, numba_cache_dir, pnjit, prange
from ..basis_operations import operateon
from .defbasis import index2state, state2index
import numpy as np

config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element(opnm, posn, coef, L, S, M, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    local_dim = np.int64(2 * S) + 1
    for a in prange(M):
        sa = index2state(a, L, local_dim)
        opco = operateon(opnm, posn, S, sa)
        if ~np.isnan(opco):
            b = state2index(sa, L, local_dim)
            
            row_init[a] = b
            col_init[a] = a
            ME_init[a] = opco * coef

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

