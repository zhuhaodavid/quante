# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 22:25:55
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 20:14:04

import numpy as np
from .....basicfun.utils_numba import njit, config, numba_cache_dir
from ..bitsoperation import cyclebits, findstate, next_combination

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
@njit("Tuple((i8,i8[:],i8[:]))(i8,i8,i8)")
def construct_Nup_kblock_basis(L:int, Nup:int, k:int) -> tuple[int, np.ndarray, np.ndarray]:
    """
    - Nup: 0,1,...,N
    - k: 0,1,...,N-1
    """
    a = 0
    s_list = []
    R_list = []
    
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << (L - Nup)) - 1
    
    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        R = checkstate(s, L, k)
        if R >= 0:
            s_list.append(s)
            R_list.append(R)
            a = a + 1
        if s == 0:
            break
        s = next_combination(s)
        
    return a, np.array(s_list), np.array(R_list)

config.CACHE_DIR = numba_cache_dir
@njit('Tuple((i8,i8))(i8,i8)')
def representative(s:int, N:int) -> tuple[int,int]:
    """
    Finding the representative r of a state-integer s
    """
    r = s; t = s; l = 0
    for i in range(1,N):
        t = cyclebits(t,1,N)
        if t < r:
            r = t; l = i
    return r, l

config.CACHE_DIR = numba_cache_dir
@njit('c16[:](i8,i8,i8)')
def Nup_kblock_state(r:int, L:int, k:int) -> np.ndarray:
    vec = np.zeros(1<<L, dtype=np.complex128)
    t = r
    for i in range(L):
        vec[t] += np.exp(-2j*np.pi*k*i/L)
        t = cyclebits(t,1,L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('c16[:](i8,i8,i8,i8,i8[:])')
def Nup_kblock_state_in_Nup(r, L, k, M, s_list):
    vec = np.zeros(M, dtype=np.complex128)
    t = r
    for i in range(L):
        vec[findstate(s_list, t)] += np.exp(-2j*np.pi*k*i/L)
        t = cyclebits(t,1,L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('c16[:,:](i8[:],i8,i8,i8)')
def projective(s_list:list, M:int, L:int, k:int) -> np.ndarray:
    proj = np.zeros((1<<L, M), dtype=np.complex128)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[t,j] += np.exp(-2j*np.pi*k*i/L)
            t = cyclebits(t,1,L)
            if t == r:
                break
    # 归一化每一列
    for j in range(M):
        norm = np.sqrt(np.sum(np.abs(proj[:, j])**2))
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit('c16[:,:](i8[:],i8,i8,i8,i8,i8[:])')
def projective_in_Nup(s_list:list, M:int, L:int, k:int, M_Nup:int, s_list_Nup:list) -> np.ndarray:
    proj = np.zeros((M_Nup, M), dtype=np.complex128)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[findstate(s_list_Nup, t),j] += np.exp(-2j*np.pi*k*i/L)
            t = cyclebits(t,1,L)
            if t == r:
                break
    # 归一化每一列
    for j in range(M):
        norm = np.sqrt(np.sum(np.abs(proj[:, j])**2))
        if norm != 0:
            proj[:, j] /= norm
    return proj


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
