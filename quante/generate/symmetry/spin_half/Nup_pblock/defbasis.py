# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 22:16:34
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-12 14:54:43

import numpy as np
from .....linalg.usenumba.numba_settings import njit, config, numba_cache_dir
from ..bitsoperation import reflectbits, next_combination

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8,i8)")
def construct_Nup_pblock_basis(L:int, Nup:int, p:int) -> tuple[int, np.ndarray]:
    """
    - Nup: 0,1,...,L
    - p: -1, 1
    """
    s_list = []
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << (L - Nup)) - 1
    
    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        if reflectbits(s, L) < s:
            if s == 0: break
            s = next_combination(s)
            continue
        if p==-1 and reflectbits(s, L) == s:
            if s == 0: break
            s = next_combination(s)
            continue
        s_list.append(s)
        if s == 0: break
        s = next_combination(s)
    return len(s_list), np.array(s_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8))(i8,i8)")
def representative(s, L):
    t = reflectbits(s, L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit
def recover(state, L, p, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = reflectbits(t,L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += p * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += p * coef / np.sqrt(2)
    return res
