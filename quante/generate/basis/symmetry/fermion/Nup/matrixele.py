# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-03 14:19:42
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-06 13:30:07

from ..bitsoperation import findstate, flip, operateon
from ......linalg.usenumba.numba_settings import njit, config, numba_cache_dir, pnjit, prange
import numpy as np

config.CACHE_DIR = numba_cache_dir
@pnjit("f8[:,:](i8,i8,i8[:],f8,f8,b1)")
def heisenberg_matrix_element(L:int, M:int, s_list:np.ndarray, jxy:float, jz:float, cyclic:bool) -> np.ndarray:
    r"""
    \sum_{i=1}^L jxy * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1}) + jz * s^z_i s^z_{i+1}

    Examples
    >>> L = 6
    >>> Nup = 3
    >>> M, s_list = construct_Nup_basis(L, Nup)
    >>> mat = heisenberg_chain_in_Nup(L,M,s_list,jxy=1.,jz=1.,cyclic=True)
    """
    H = np.zeros((M, M), dtype=float)
    last_site = L if cyclic else L - 1
    for a in prange(M):
        sa = s_list[a]
        for i in range(last_site):
            j = (i+1) % L
            if ((sa >> i) & 1) == ((sa >> j) & 1):
                H[a,a] += 0.25 * jz
            else:
                H[a,a] -= 0.25 * jz
                s = flip(sa, i ,j)
                if s != -1:
                    b = findstate(s_list, s)
                    if b != -1:
                        H[b,a] += 0.5 * jxy
    return H


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element(opnm, posn, coef, L, M, s_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    for a in prange(M):
        sa = s_list[a]
        
        opco, sb = operateon(opnm, posn, sa, L)
        if sb != -1:
            b = findstate(s_list, sb)
            if b != -1:
                
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]


config.CACHE_DIR = numba_cache_dir
@pnjit
def diag_matrix_element(opnm, posn, coef, L, M, s_list, dtype):
    diag = np.empty(M, dtype=dtype)
    for a in prange(M):
        sa = s_list[a]
        opco, t = operateon(opnm, posn, sa, L)
        if t < 0:
            diag[a] = 0.0
        else:
            diag[a] = opco * coef
    return diag


config.CACHE_DIR = numba_cache_dir
@njit
def project(state, Ns, s_list):
    M, N = state.shape
    res = np.zeros((Ns, N), dtype=np.complex128)
    for i in range(N):
        for a in range(M):
            stateai = state[a,i]
            if stateai == 0:
                continue
            b = findstate(s_list, a)
            if b >= 0:
                res[b,i] += stateai
    return res
