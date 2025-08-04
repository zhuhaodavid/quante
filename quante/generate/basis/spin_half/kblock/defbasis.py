# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 21:02:07
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 20:13:32

import numpy as np
from .....basicfun.utils_numba import njit, config, numba_cache_dir
from ..bitsoperation import cyclebits

config.CACHE_DIR = numba_cache_dir

@njit("i8(i8,i8,i8)")
def checkstate(s:int, L:int, k:int) -> int:
    t = s
    for i in range(1,L+1):
        t = cyclebits(t,1,L)
        if t < s:
            return -1
        elif t == s:
            if k % (L//i) != 0:
                return -1
            return i
    return -1

config.CACHE_DIR = numba_cache_dir

@njit("Tuple((i8,i8[:],i8[:]))(i8,i8)")
def construct_kblock_basis(L:int, k:int) -> tuple[int, np.ndarray, np.ndarray]:
    """
    - k: 0,1,...,N-1
    """
    a = 0
    s_list = []
    R_list = []
    for s in range(1<<L):
        R = checkstate(s, L, k)
        if R >= 0:
            s_list.append(s)
            R_list.append(R)
            a = a + 1
    return a, np.array(s_list), np.array(R_list)

config.CACHE_DIR = numba_cache_dir

@njit('Tuple((i8,i8))(i8,i8)')
def representative(s:int, L:int) -> tuple[int,int]:
    """
    Finding the representative r of a state-integer s
    """
    r = s; t = s; l = 0
    for i in range(1,L):
        t = cyclebits(t,1,L)
        if t < r:
            r = t; l = i
    return r, l

config.CACHE_DIR = numba_cache_dir
@njit("c16[:,:](c16[:,:],i8,i8,i8[:],i8[:])")
def recover(state, L, k, s_list, R_list):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=np.complex128)
    for j in range(N):
        for a in range(M):
            coef = state[a,j]
            if coef == 0:
                continue
            sqrtRa = np.sqrt(R_list[a])
            r = s_list[a]
            t = r
            for i in range(L):
                res[t, j] += coef * np.exp( - 2j * np.pi * k * i / L) / sqrtRa
                t = cyclebits(t,1,L)
                if t == r:
                    break
    return res

