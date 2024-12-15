# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 22:04:59
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-12 14:57:08

import numpy as np
from ......linalg.usenumba.numba_settings import njit, config, numba_cache_dir
from ..bitsoperation import invertbits, reflectbits


config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8)")
def construct_pzblock_basis(L:int, pz:int) -> tuple[int, np.ndarray]:
    """
    - pz: -1, 1
    """
    s_list = []
    for s in range(1<<L):
        if reflectbits(invertbits(s, L), L) < s:
            continue
        if pz==-1 and reflectbits(invertbits(s, L), L) == s:
            continue
        s_list.append(s)
    return len(s_list), np.array(s_list)


config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8))(i8,i8)")
def representative(s:int, L:int) -> tuple[int, int]:
    t = reflectbits(invertbits(s, L), L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit
def recover(state, L, pz, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = reflectbits(invertbits(t,L), L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += pz * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += pz * coef / np.sqrt(2)
    return res
