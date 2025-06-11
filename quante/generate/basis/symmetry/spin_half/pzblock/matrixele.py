# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 22:06:13
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-19 12:31:19


import numpy as np
from ......basicfun.utils_numba import config, numba_cache_dir, pnjit, prange
from ..bitsoperation import flip, operateon, findstate, invertbits, reflectbits
from .defbasis import representative

config.CACHE_DIR = numba_cache_dir
@pnjit("f8[:,:](i8,i8,i8,f8,f8,i8[:],b1)")
def heisenberg_matrix_element(L:int, M:int, pz:int, jxy:float, jz:float, s_list:np.ndarray, cyclic:bool) -> np.ndarray:
    r"""
    \sum_{i=1}^L jxy * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1}) + jz * s^z_i s^z_{i+1}

    Examples
    >>> L = 10; pz = 1
    >>> M, s_list = construct_pzblock_basis(L, pz)
    >>> mat = heisenberg_chain_in_pzblock(L, pz, jxy=1., jz=1., s_list=s_list)
    """
    H = np.zeros((M,M), dtype=np.float64)

    for a in prange(M):
        sa = s_list[a]
        last_site = L if cyclic else L - 1
        for i in range(last_site):
            j = (i+1) % L
            if ((sa >> i) & 1) == ((sa >> j) & 1):
                H[a,a] += 0.25 * jz
            else:
                H[a,a] -= 0.25 * jz

            flipsa = flip(sa, i, j)
            if flipsa == -1:
                continue

            sb, l = representative(flipsa, L)
            b = findstate(s_list, sb)
            if b >= 0:
                Na = 4 if reflectbits(invertbits(sa, L), L) == sa else 2
                Nb = 4 if reflectbits(invertbits(sb, L), L) == sb else 2
                H[b,a] += jxy * 0.5 * (Nb/Na) ** 0.5 * pz**l
    return H


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element(opnm, posn, coef, L, pz, M, s_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    for a in prange(M):
        sa = s_list[a]
        
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            sb, l = representative(s, L)
            b = findstate(s_list, sb)
            if b != -1:
                Na = 4 if reflectbits(invertbits(sa, L), L) == sa else 2
                Nb = 4 if reflectbits(invertbits(sb, L), L) == sb else 2
                
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (Nb/Na) ** 0.5 * pz**l

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]
