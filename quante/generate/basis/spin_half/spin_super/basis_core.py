# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-01 15:43:21
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-01 18:42:34

import numpy as np
from .....basicfun.utils_numba import njit, config, numba_cache_dir
from ..bitsoperation import perm_operation, count_tot_down, next_combination


###########################################
# Z21
##############################################
config.CACHE_DIR = numba_cache_dir
@njit
def construct_full_basis(L, anci_perm):
    s_list_sym = []
    s_list_asym = []
    for s in range(1<<L):
        s_prime = perm_operation(s, anci_perm)
        if s_prime < s:
            continue
        s_list_sym.append(s)
        if s_prime != s:
            s_list_asym.append(s)

    s_list = s_list_sym + s_list_asym
    return np.array(s_list), len(s_list_sym), len(s_list_asym)


config.CACHE_DIR = numba_cache_dir
@njit
def representative_Z21(s, perm):
    t = perm_operation(s, perm)
    if t < s:
        return t, 1
    else:
        return s, 0


