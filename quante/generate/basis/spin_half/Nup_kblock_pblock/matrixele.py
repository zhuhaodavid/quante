# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 23:29:24
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 20:14:13

import numpy as np

from .....basicfun.utils_numba import njit, config, numba_cache_dir, pnjit, prange
from ..bitsoperation import flip, operateon, findstate
from .defbasis import representative

config.CACHE_DIR = numba_cache_dir
@njit("f8[:,:](i8,i8,i8,i8,f8,f8,i8[:],i8[:],i8[:])")
def heisenberg_matrix_element(L:int, M:int, k:int, p:int, jxy:float, jz:float, s_list:np.ndarray, R_list:np.ndarray, m_list:np.ndarray) -> np.ndarray:
    r"""
    \sum_{i=1}^N jxy * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1}) + jz * s^z_i s^z_{i+1}

    Examples
    >>> L = 6
    >>> Nup = 3
    >>> k = 1
    >>> p = 1
    >>> M, s_list, R_list, m_list = construct_Nup_kblock_pblock_basis(L=L, Nup=Nup, k=k, p=p)
    >>> mat = heisenberg_chain_in_Nup_kblock_plock(L, M, k, p, jxy=1., jz=1., s_list=s_list, R_list=R_list, m_list=m_list)
    """
    M = len(s_list)
    H = np.zeros((M,M), dtype=np.float64)

    for a in range(M):
        sa = s_list[a]
        if a > 0 and sa == s_list[a-1]:
            continue
        elif a < M - 1 and sa == s_list[a+1]:
            n = 2
        else:
            n = 1  # n is the number of copies of the representative

        for i in range(L):
            for ai in range(a, a+n):
                j = (i+1) % L
                if ((sa >> i) & 1) == ((sa >> j) & 1):
                    H[ai,ai] += 0.25 * jz
                else:
                    H[ai,ai] -= 0.25 * jz

            flipsa = flip(sa, i, j)
            if flipsa == -1:
                continue
            sb, l, q = representative(flipsa, L)
            b = findstate(s_list, sb)
            if b >= 0:

                if b > 1 and sb == s_list[b-1]:
                    m = 2
                    b = b - 1
                elif b + 1 < M and sb == s_list[b+1]:
                    m = 2
                else:
                    m = 1

                for jj in range(b, b+m):
                    for ii in range(a, a+n):

                        sigma_a = np.sign(R_list[ii])
                        mu_a = m_list[ii]
                        sigma_b = np.sign(R_list[jj])
                        mu_b = m_list[jj]

                        helement = jxy * 0.5 * (sigma_a * p) ** q * (np.abs(R_list[ii]/R_list[jj]))**0.5

                        TPa = mu_a != -1  # cyclebits(reflectbits(sa,N), mu_a, N) == sa
                        TPb = mu_b != -1  # cyclebits(reflectbits(sb,N), mu_b, N) == sb

                        if TPa:
                            helement /= (1 + sigma_a * p * np.cos(k*mu_a*2*np.pi/L)) ** 0.5

                        if TPb:
                            helement *= (1 + sigma_b * p * np.cos(k*mu_b*2*np.pi/L)) ** 0.5

                        if sigma_a == sigma_b:
                            if not TPb:
                                helement *= np.cos(k*l*2*np.pi/L)
                            else:
                                helement *= (np.cos(k*l*2*np.pi/L) + sigma_a * p * np.cos(k*(l - mu_b)*2*np.pi/L)) / (1 + sigma_a * p * np.cos(k*mu_b*2*np.pi/L))
                        else:
                            if not TPb:
                                helement *= sigma_b * np.sin(k*l*2*np.pi/L)
                            else:
                                helement *= (sigma_b * np.sin(k*l*2*np.pi/L) + p * np.sin(k*(l - mu_b)*2*np.pi/L)) / (1 + sigma_b * p * np.cos(k*mu_b*2*np.pi/L))

                        H[jj,ii] = H[jj,ii] + helement
    return H


config.CACHE_DIR = numba_cache_dir
@njit
def helement(coef, sigma_a, Ri_Rj, p, mu_a, q, k, L, mu_b, sigma_b, l):
    helement = coef * (sigma_a * p) ** q * Ri_Rj**0.5
    
    TPa = mu_a != -1  # cyclebits(reflectbits(sa,N), mu_a, N) == sa
    TPb = mu_b != -1  # cyclebits(reflectbits(sb,N), mu_b, N) == sb
    
    if TPa:
        helement /= (1 + sigma_a * p * np.cos(k*mu_a*2*np.pi/L)) ** 0.5
    
    if TPb:
        helement *= (1 + sigma_b * p * np.cos(k*mu_b*2*np.pi/L)) ** 0.5
    
    if sigma_a == sigma_b:
        if not TPb:
            helement *= np.cos(k*l*2*np.pi/L)
        else:
            helement *= (np.cos(k*l*2*np.pi/L) + sigma_a * p * np.cos(k*(l - mu_b)*2*np.pi/L)) / (1 + sigma_a * p * np.cos(k*mu_b*2*np.pi/L))
    else:
        if not TPb:
            helement *= sigma_b * np.sin(k*l*2*np.pi/L)  # todo 为什么这里有负号
        else:
            helement *= (sigma_b * np.sin(k*l*2*np.pi/L) + p * np.sin(k*(l - mu_b)*2*np.pi/L)) / (1 + sigma_b * p * np.cos(k*mu_b*2*np.pi/L))
    return helement 


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element(opnm, posn, coef, L, k, p, M, s_list, R_list, m_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    for a in prange(M):
        sa = s_list[a]
        if a > 0 and sa == s_list[a-1]:
            continue
        elif a < M - 1 and sa == s_list[a+1]:
            n = 2
        else:
            n = 1  # n is the number of copies of the representative
        
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            sb, l, q = representative(s, L)
            b = findstate(s_list, sb)
            if b >= 0: #  and sb == 447
                if b > 1 and sb == s_list[b-1]:
                    m = 2
                    b = b - 1
                elif b + 1 < M and sb == s_list[b+1]:
                    m = 2
                else:
                    m = 1
                for jj in range(b, b+m):
                    for ii in range(a, a+n):
                        sigma_a = np.sign(R_list[ii]) 
                        mu_a = m_list[ii]
                        sigma_b = np.sign(R_list[jj])
                        mu_b = m_list[jj]
                        Ri_Rj = np.abs(R_list[ii]/R_list[jj])
                        
                        row_init[4*a+(ii-a)+2*(jj-b)] = jj
                        col_init[4*a+(ii-a)+2*(jj-b)] = ii
                        ME_init[4*a+(ii-a)+2*(jj-b)] = opco * helement(coef, sigma_a, Ri_Rj, p, mu_a, q, k, L, mu_b, sigma_b, l)

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

