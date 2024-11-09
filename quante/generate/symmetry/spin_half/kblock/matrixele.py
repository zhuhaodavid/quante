# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 21:04:47
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-19 12:32:14


import numpy as np
from quante.generate.symmetry.spin_half.bitsoperation import findstate, flip
from .....linalg.usenumba.numba_settings import njit, config, numba_cache_dir, pnjit, prange
from ..bitsoperation import operateon, findstate
from .defbasis import representative

config.CACHE_DIR = numba_cache_dir
@pnjit('c16[:,:](i8,i8,i8,f8,f8,i8[:],i8[:])')
def heisenberg_matrix_element(L:int, M:int, k:int, jxy:float, jz:float, s_list:np.ndarray, R_list:np.ndarray) -> np.ndarray:
    r"""
    \sum_{i=1}^L jxy * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1}) + jz * s^z_i s^z_{i+1}

    Examples
    >>> L = 10; k = 1
    >>> M, s_list, R_list = construct_kblock_basis(L, k)
    >>> mat = heisenberg_chain_in_kblock(L, M, k, jxy=1., jz=1., s_list=s_list, R_list=R_list)
    """
    H = np.zeros((M, M), dtype=np.complex128)
    tmpval = np.exp(-1j*2*np.pi*k/L)
    for a in prange(M):
        for i in range(L):
            j = (i+1) % L
            if ((s_list[a] >> i) & 1) == ((s_list[a] >> j) & 1):
                H[a,a] += 0.25 * jz
            else:
                H[a,a] -= 0.25 * jz
                s = flip(s_list[a], i, j)
                r, l = representative(s,L)
                b = findstate(s_list, r)
                if b >= 0:
                    H[b,a] += 0.5 * (R_list[a]/R_list[b])**0.5 * tmpval**l * jxy
    return H


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element(opnm, posn, coef, L, k, M, s_list, R_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    tmpval = np.exp(-1j * 2 * np.pi * k / L)
    for a in prange(M):
        sa = s_list[a]
        
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            r, l = representative(s, L)
            b = findstate(s_list, r)
            if b >= 0:
                
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (R_list[a] / R_list[b])**0.5 * tmpval ** l

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

@njit
def project(state, L, k, Ns, s_list, R_list):
    M, N = state.shape
    res = np.zeros((Ns, N), dtype=np.complex128)
    for i in range(N):
        for a in range(M):
            stateai = state[a,i]
            if stateai == 0:
                continue
            r, l = representative(a, L)
            if r == -1:
                continue
            b = findstate(s_list, r)
            if b >= 0:
                res[b,i] += stateai * (R_list[b])**0.5/L * np.exp(-1j * 2 * np.pi * k * l / L)
    return res
