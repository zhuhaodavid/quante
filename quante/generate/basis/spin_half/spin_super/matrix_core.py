# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-01 18:06:39
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-01 18:45:42

import numpy as np

from .....basicfun.utils_numba import njit, config, numba_cache_dir, pnjit, prange
from ..bitsoperation import operateon, findstate, perm_operation
from .basis_core import representative_Z21


###########################################
# Z21
##############################################
config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_full_sym(opnm, posn, coef, L, perm, M_sym, M, s_list, row_init, col_init, ME_init):
    ME_init.fill(0.0)
    for a in prange(M_sym):
        sa = s_list[a]
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            sb, l = representative_Z21(s, perm)
            b = findstate(s_list[:M_sym], sb)
            if b >= 0:
                Na = 4 if perm_operation(sa, perm) == sa else 2
                Nb = 4 if perm_operation(sb, perm) == sb else 2
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (Nb/Na) ** 0.5
    for a in prange(M_sym, M):
        sa = s_list[a]
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            sb, l = representative_Z21(s, perm)
            b = findstate(s_list[M_sym:], sb)
            if b >= 0:
                Na = 4 if perm_operation(sa, perm) == sa else 2
                Nb = 4 if perm_operation(sb, perm) == sb else 2
                row_init[a] = M_sym+b
                col_init[a] = a
                ME_init[a] = opco * coef * (Nb/Na) ** 0.5 * (-1)**l

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_full_asym(opnm, posn, coef, L, perm, M_sym, M, s_list, row_init, col_init, ME_init):
    ME_init.fill(0.0)
    for a in prange(M_sym):
        sa = s_list[a]
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            sb, l = representative_Z21(s, perm)
            b = findstate(s_list[M_sym:], sb)
            if b >= 0:
                Na = 4 if perm_operation(sa, perm) == sa else 2
                Nb = 4 if perm_operation(sb, perm) == sb else 2
                row_init[a] = M_sym+b
                col_init[a] = a
                ME_init[a] = opco * coef * (Nb/Na) ** 0.5 * (-1)**l
    for a in prange(M_sym, M):
        sa = s_list[a]
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            sb, l = representative_Z21(s, perm)
            b = findstate(s_list[:M_sym], sb)
            if b >= 0:
                Na = 4 if perm_operation(sa, perm) == sa else 2
                Nb = 4 if perm_operation(sb, perm) == sb else 2
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (Nb/Na) ** 0.5

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]


