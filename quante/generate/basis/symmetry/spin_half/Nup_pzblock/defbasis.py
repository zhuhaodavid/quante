# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 22:31:07
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-12 14:55:28

import numpy as np
from ......linalg.usenumba.numba_settings import njit, config, numba_cache_dir
from ..bitsoperation import invertbits, reflectbits, findstate, next_combination

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8,i8)")
def construct_Nup_pzblock_basis(L:int, Nup:int, pz:int) -> tuple[int, np.ndarray]:
    """
    - Nup: 0,1,...,L//2
    - pz: -1, 1
    """
    # 初始值：前Nup个位为1，其他位为0
    s_list = []
    lis = [(1 << Nup) - 1] if L == 2*Nup else [(1 << (L - Nup)) - 1, (1 << Nup) - 1]
    for s in lis:
        # 在 N 个位置内生成具有 Nup 个1的所有可能组合
        while s < (1 << L):
            if reflectbits(invertbits(s, L), L) < s:
                if s == 0: break
                s = next_combination(s)
                continue
            if pz==-1 and reflectbits(invertbits(s, L), L) == s:
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
    t = reflectbits(invertbits(s, L), L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit('f8[:](i8,i8,i8)')
def Nup_pzblock_state(r:int, L:int, pz:int) -> np.ndarray:
    vec = np.zeros(1<<L, dtype=np.float64)
    t = r
    for _ in range(L):
        vec[t] += pz
        t = reflectbits(invertbits(t,L),L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('f8[:](i8,i8,i8,i8,i8[:])')
def Nup_pzblock_state_in_Nup(r, L, pz, M, s_list):
    vec = np.zeros(M, dtype=np.float64)
    t = r
    for i in range(L):
        vec[findstate(s_list, t)] += pz
        t = reflectbits(invertbits(t,L),L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8)')
def projective(s_list:list, M:int, L:int, pz:int) -> np.ndarray:
    proj = np.zeros((1<<L, M), dtype=np.float64)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[t,j] += pz
            t = reflectbits(invertbits(t,L),L)
            if t == r:
                break
    # 归一化每一列
    for j in range(M):
        norm = np.linalg.norm(proj[:, j])
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8,i8,i8[:])')
def projective_in_Nup(s_list:list, M:int, L:int, pz:int, M_Nup:int, s_list_Nup:list) -> np.ndarray:
    proj = np.zeros((M_Nup, M), dtype=np.float64)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[findstate(s_list_Nup, t),j] += pz
            t = reflectbits(invertbits(t,L),L)
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
def recover(state, L, pz, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = reflectbits(invertbits(t,L),L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += pz * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += pz * coef / np.sqrt(2)
    return res
