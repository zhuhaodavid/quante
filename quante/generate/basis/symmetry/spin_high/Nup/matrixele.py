# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-12 22:05:55
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-15 20:31:44



from ......basicfun.utils_numba import pnjit, prange, config, numba_cache_dir
import numpy as np
from ..basis_operations import operateon, findstate


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element(opnm, posn, coef, L, S, M, s_list, row_init, col_init, ME_init):
    ME_init.fill(0.0)
    for a in prange(M):
        sa = s_list[a]
        sb = sa.copy()
        opco = operateon(opnm, posn, S, sb)
        if not np.isnan(opco):
            b = findstate(s_list, sb)
            if b != -1:
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

