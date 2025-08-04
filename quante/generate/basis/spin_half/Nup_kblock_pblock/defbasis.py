# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 23:27:53
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 20:14:09

import numpy as np
from .....basicfun.utils_numba import njit, config, numba_cache_dir
from ..bitsoperation import cyclebits, reflectbits, next_combination, findstate

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8))(i8,i8,i8)")
def checkstate(s:int, L:int, k:int) -> int:
    R = -1
    m = -1
    
    t = s
    for i in range(1,L+1):
        t = cyclebits(t, 1, L)
        if t < s:
            return R, m
        elif t == s:
            if k % (L//i) != 0:
                return R, m
            R = i
            break
    
    t = reflectbits(s, L)
    for i in range(R):
        if t < s:
            return -1, m
        elif t == s:
            m = i
            return R, m
        t = cyclebits(t, 1, L)
    return R, m

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:], i8[:], i8[:]))(i8,i8,i8,i8)")
def construct_Nup_kblock_pblock_basis(L:int, Nup:int, k:int, p:int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    - Nup: 0,1,...,N
    - k: 0,1,...,N//2
    - p: -1, 1
    """
    s_list = []
    R_list = []
    m_list = []
    sigma_list = [1] if k == 0 or 2*k == L else [-1, 1]
    
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << (L - Nup)) - 1
    
    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        R, m = checkstate(s, L, k)
        for sigma in sigma_list:
            if m != -1:
                if np.abs(1 + sigma * p * np.cos(k*m*2*np.pi/L)) < 1e-6:
                    continue
                if sigma == -1 and np.abs(1 - sigma * p * np.cos(k*m*2*np.pi/L)) > 1e-6:
                    continue
            if R > 0:
                s_list.append(s)
                R_list.append(sigma * R)
                m_list.append(m)
        if s == 0:
            break
        s = next_combination(s)
        
    return len(s_list), np.array(s_list), np.array(R_list), np.array(m_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8, i8))(i8,i8)")
def representative(s:int, N:int) -> tuple[int, int, int]:
    r = s; t = s; l = 0
    
    for i in range(1,N):
        t = cyclebits(t, -1, N)
        if t < r:
            r = t
            l = i
    
    t = reflectbits(s, N)
    q = 0
    for i in range(N):
        if t < r:
            r = t
            l = i
            q = 1
        t = cyclebits(t, -1, N)
    return r, -l, q

config.CACHE_DIR = numba_cache_dir
@njit("f8[:](i8,i8,i8,i8,i8)")
def Nup_kblock_pblock_state(s:int, L:int, k:int, p:int, sigma:int) -> np.ndarray:
    vec = np.zeros(1<<L, dtype=np.float64)
    for r in range(L):
        i = cyclebits(s, r, L)
        vec[i] += np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L)
        j = reflectbits(cyclebits(s, r, L), L)
        vec[j] += p * (np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L))
    vec = vec / np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit("f8[:](i8,i8,i8,i8,i8,i8,i8[:])")
def Nup_kblock_pblock_state_in_Nup(s:int, L:int, k:int, p:int, sigma:int, M:int, s_list:list) -> np.ndarray:
    vec = np.zeros(M, dtype=np.float64)
    for r in range(L):
        i = cyclebits(s, r, L)
        vec[findstate(s_list, i)] += np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L)
        j = reflectbits(cyclebits(s, r, L), L)
        vec[findstate(s_list, j)] += p * (np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L))
    vec = vec / np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8,i8,i8[:])')
def projective(s_list:list, M:int, L:int, k:int, p:int, R_list:list) -> np.ndarray:
    proj = np.zeros((1<<L, M), dtype=np.float64)
    for t in range(M):
        s = s_list[t]
        sigma = np.sign(R_list[t])
        for r in range(L):
            i = cyclebits(s, r, L)
            proj[i, t] += np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L)
            j = reflectbits(cyclebits(s, r, L), L)
            proj[j, t] += p * (np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L))
    # 归一化每一列
    for j in range(M):
        norm = np.linalg.norm(proj[:, j])
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8,i8,i8[:],i8,i8[:])')
def projective_in_Nup(s_list:list, M:int, L:int, k:int, p:int, R_list:list, M_Nup:int, s_list_Nup:list) -> np.ndarray:
    proj = np.zeros((M_Nup, M), dtype=np.float64)
    for t in range(M):
        s = s_list[t]
        sigma = np.sign(R_list[t])
        for r in range(L):
            i = cyclebits(s, r, L)
            proj[findstate(s_list_Nup, i), t] += np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L)
            j = reflectbits(cyclebits(s, r, L), L)
            proj[findstate(s_list_Nup, j), t] += p * (np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L))
    # 归一化每一列
    for j in range(M):
        norm = np.linalg.norm(proj[:, j])
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit
def recover(state, L, k, p, s_list, R_list, m_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for ii in range(N):
        for t in range(M):
            coef = state[t,ii]
            if coef == 0:
                continue
                
            s = s_list[t]
            mu_a = m_list[t]
            TPa = mu_a != -1
            
            sigma = np.sign(R_list[t])
            R = np.abs(R_list[t])
            norm = L**2 / R * (1 + sigma * p * np.cos(2*np.pi*k*mu_a/L)) if TPa else L**2 / R
            if k == 0 or 2*k == L:
                norm *= 2
            if sigma == -1:
                for r in range(L):
                    tmp = np.sin(k*r * 2*np.pi/L) * coef
                    i = cyclebits(s, r, L)
                    res[i, ii] += tmp / (np.sqrt(norm))
                    i = reflectbits(cyclebits(s, r, L), L)
                    res[i, ii] += p * tmp  / (np.sqrt(norm))
            else:
                for r in range(L):
                    tmp = np.cos(k*r * 2*np.pi/L) * coef
                    i = cyclebits(s, r, L)
                    res[i, ii] += tmp / (np.sqrt(norm))
                    i = reflectbits(cyclebits(s, r, L), L)
                    res[i, ii] += p * tmp  / (np.sqrt(norm))
            
    return res
