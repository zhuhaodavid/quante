# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 20:39:15
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-11 20:55:19

import numpy as np
from .....basicfun.utils_numba import njit, numba_cache_dir, config

config.CACHE_DIR = numba_cache_dir
@njit
def _Hi_model(J:float, h:np.ndarray, L:int):
    mat = np.zeros((2 ** L,), dtype=float)
    for i in range(1<<L):

        zsum = 0.
        mask = 1 << (L-1)
        for j in range(L):
            if i & mask:
                zsum += - h[j]
            else:
                zsum += h[j]
            mask >>= 1

        zzsum = 0
        mask11 = 0b11 << (L-2)
        for j in range(L-1):
            if mask11 & i == 0 or mask11 & i == mask11:
                zzsum += 1
            mask11 >>= 1
        if (i & 1) == (i >> (L-1)):
            zzsum += 1

        mat[i] = J * (2 * zzsum - L) + zsum
    return mat