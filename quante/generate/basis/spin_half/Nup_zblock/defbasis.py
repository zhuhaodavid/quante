# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 22:31:07
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-12 14:56:05

import numpy as np
from .....basicfun.utils_numba import njit, config, numba_cache_dir
from ..bitsoperation import invertbits, next_combination

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8,i8)")
def construct_Nup_zblock_basis(L:int, Nup:int, z:int) -> tuple[int, np.ndarray]:
    """
    - Nup: 0,1,...,L//2
    - z: -1, 1
    """
    # 初始值：前Nup个位为1，其他位为0
    s_list = []
    lis = [(1 << Nup) - 1] if L == 2*Nup else [(1 << (L - Nup)) - 1, (1 << Nup) - 1]
    for s in lis:
        # 在 N 个位置内生成具有 Nup 个1的所有可能组合
        while s < (1 << L):
            if invertbits(s, L) < s:
                if s == 0: break
                s = next_combination(s)
                continue
            if z==-1 and invertbits(s, L) == s:
                if s == 0: break
                s = next_combination(s)
                continue
            s_list.append(s)
            if s == 0: break
            s = next_combination(s)
    s_list = np.array(s_list, dtype=np.int64)
    return len(s_list), np.sort(s_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8))(i8,i8)")
def representative(s:int, L:int) -> tuple[int, int]:
    t = invertbits(s, L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit
def recover(state, L, z, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = invertbits(t,L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += z * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += z * coef / np.sqrt(2)
    return res
