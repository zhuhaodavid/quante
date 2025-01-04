# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 21:51:46
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-12 14:56:39

import numpy as np
from ......linalg.usenumba.numba_settings import njit, config, numba_cache_dir
from ..bitsoperation import reflectbits

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8)")
def construct_pblock_basis(L:int, p:int) -> tuple[int, np.ndarray]:
    """
    - p: -1, 1
    """
    s_list = []
    for s in range(1<<L):
        if reflectbits(s, L) < s:
            continue
        if p==-1 and reflectbits(s, L) == s:
            continue
        s_list.append(s)
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
@njit('f8[:](i8,i8,i8)')
def pblock_state(r:int, L:int, p:int) -> np.ndarray:
    vec = np.zeros(1<<L, dtype=np.float64)
    t = r
    for _ in range(L):
        vec[t] += p
        t = reflectbits(t,L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8)')
def projective(s_list:list, M:int, L:int, p:int) -> np.ndarray:
    proj = np.zeros((1<<L, M), dtype=np.float64)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[t,j] += p
            t = reflectbits(t,L)
            if t == r:
                break
    # 归一化每一列
    for j in range(M):
        norm = np.linalg.norm(proj[:, j])
        if norm != 0:
            proj[:, j] /= norm
    return proj

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
