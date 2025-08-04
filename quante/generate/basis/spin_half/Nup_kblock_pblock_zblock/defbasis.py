# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 23:37:56
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 20:14:17

from ..bitsoperation import cyclebits, reflectbits, invertbits, next_combination
from .....basicfun.utils_numba import njit, config, numba_cache_dir
import numpy as np

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8, i8, i8))(i8,i8,i8)")
def checkstate(s:int, L:int, k:int) -> tuple[int, int, int, int]:
    R = -1  # 循环周期
    m = -1  # 反射或者反转后的循环
    n = -1  # 反射+反转后的循环
    c = 1   # 代表性质，1，2，3，4，5
    
    t = s
    for i in range(1,L+1):
        t = cyclebits(t, 1, L)
        if t < s:  # not a representative
            return R, m, n, c
        elif t == s:  # may be representative
            if k % (L//i) != 0:  # not representative for this k block
                return R, m, n, c
            R = i  # representative for k block
            break  # not enough, need to check case after reflection
    
    t = reflectbits(s, L)
    for i in range(R):  # only need to check up to R-1
        if t < s:  # not a representative, cause there is a smaller one
            return -1 ,m, n, c
        elif t == s:  # may be representative
            m = i
            c = 2
            break  # not enough, need to check case after inversion
        t = cyclebits(t, 1, L)
    
    t = invertbits(s, L)
    for i in range(R):
        if t < s:  # not a representative, cause there is a smaller one
            return -1 ,m, n, c
        elif t == s:  # may be representative
            if m == -1:
                m = i
                c = 3
            else:
                n = i
                c = 5
            break  # not enough, need to check case after inversion and reflection
        t = cyclebits(t, 1, L)
    
    t = reflectbits(invertbits(s, L), L)
    for i in range(R):
        if t < s:
            return -1 ,m, c, n
        elif t == s:
            if m == -1:
                m = i
                c = 4
            else:
                if n == -1:
                    raise Exception("Should not happen")
            break
        t = cyclebits(t, 1, L)
    
    return R, m, n, c

config.CACHE_DIR = numba_cache_dir
@njit("f8(i8,i8,i8,i8,i8,i8,i8,i8)")
def get_nmc(L, k, p, z, sigma, m, n, c):
    kpi = k * 2 *np.pi / L
    if c == 2:
        nmc = 1 + sigma * p * np.cos(m*kpi)
    elif c == 3:
        nmc = 1 + z * np.cos(m*kpi)
    elif c == 4:
        nmc = 1 + sigma * p * z * np.cos(m*kpi)
    elif c == 5:
        nmc = (1 + sigma * p * np.cos(m*kpi))*(1 + z * np.cos(n*kpi))
    else:
        nmc = 1.
    return nmc

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8[:], i8[:], i8[:], i8[:]))(i8,i8,i8,i8)")
def construct_Nup_kblock_pblock_zblock_basis(L:int, k:int, p:int, z:int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    - N: 偶数, 只能考虑半满空间：Nup = N/2
    - k: 0,1,...,N//2
    - p: -1, 1
    - z: -1, 1
    """
    assert L % 2 == 0, "N must be even"
    s_list = []
    R_list = []
    m_list = []
    c_list = []
    sigma_list = [1] if k == 0 or 2*k == L else [-1, 1]
    
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << L//2) - 1
    
    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        R, m, n, c = checkstate(s, L, k)
        for sigma in sigma_list:
            nmc = 1.
            if m != -1:
                nmc = get_nmc(L, k, p, z, sigma, m, n, c)
                if nmc == 0.:
                    continue
                if c == 2:
                    if sigma == -1 and np.abs(1 - sigma * p * np.cos(k*m*2*np.pi/L)) > 1e-6:
                        continue
                elif c == 4:
                    if sigma == -1 and np.abs(1 - sigma * p * z * np.cos(k*m*2*np.pi/L)) > 1e-6:
                        continue
                elif c == 5:
                    if sigma == -1 and np.abs((1 - sigma * p * np.cos(k*m*2*np.pi/L))*(1 + z * np.cos(k*n*2*np.pi/L))) > 1e-6:
                        continue
            if R > 0:
                s_list.append(s)
                R_list.append(sigma * R)
                m_list.append(m)
                c_list.append(c+n+1)
        if s == 0: break
        s = next_combination(s)
        
    return len(s_list), np.array(s_list, dtype=np.int64), np.array(R_list, dtype=np.int64), np.array(m_list, dtype=np.int64), np.array(c_list, dtype=np.int64)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8, i8, i8))(i8,i8)")
def representative(s:int, L:int) -> tuple[int, int, int, int]:
    r = s; t = s
    
    l = 0
    for i in range(1,L):
        t = cyclebits(t, -1, L)
        if t < r:
            r = t
            l = i
    
    t = reflectbits(s, L)
    q = 0
    for i in range(L):
        if t < r:
            r = t
            l = i
            q = 1
        t = cyclebits(t, -1, L)
    
    t = invertbits(s, L)
    g = 0
    for i in range(L):
        if t < r:
            r = t
            l = i
            g = 1
            q = 0
        t = cyclebits(t, -1, L)
    
    t = invertbits(reflectbits(s, L), L)
    for i in range(L):
        if t < r:
            r = t
            l = i
            g = 1
            q = 1
        t = cyclebits(t, -1, L)
    
    return r, -l, q, g

config.CACHE_DIR = numba_cache_dir
@njit
def recover(state, L, k, p, z, s_list, R_list, m_list, c_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for ii in range(N):
        for t in range(M):
            coef = state[t,ii]
            if coef == 0:
                continue

            s = s_list[t]
            sigma = np.sign(R_list[t]) 
            mu = m_list[t]
            
            c_p = c_list[t]
            if c_p > 4:
                c = 5
                n = c_p - 6
            else:
                c = c_p
                n = -1

            sigma = np.sign(R_list[t])
            R = np.abs(R_list[t])
            norm = 2 * L**2 / R * get_nmc(L, k, p, z, sigma, mu, n, c)
            
            if k == 0 or 2*k == L:
                norm *= 2
            
            if sigma == -1:
                for r in range(L):
                    tmp = np.sin(k*r * 2*np.pi/L) / np.sqrt(norm)
                    i = cyclebits(s, r, L)
                    res[i, t] += tmp
                    i = reflectbits(cyclebits(s, r, L), L)
                    res[i, t] += p * tmp
                    i = invertbits(cyclebits(s, r, L), L)
                    res[i, t] += z * tmp
                    i = reflectbits(invertbits(cyclebits(s, r, L), L), L)
                    res[i, t] += p*z* tmp
            else:
                for r in range(L):
                    tmp = np.cos(k*r * 2*np.pi/L) / np.sqrt(norm)
                    i = cyclebits(s, r, L)
                    res[i, t] += tmp
                    i = reflectbits(cyclebits(s, r, L), L)
                    res[i, t] += p * tmp
                    i = invertbits(cyclebits(s, r, L), L)
                    res[i, t] += z * tmp
                    i = reflectbits(invertbits(cyclebits(s, r, L), L), L)
                    res[i, t] += p*z* tmp
    return res
