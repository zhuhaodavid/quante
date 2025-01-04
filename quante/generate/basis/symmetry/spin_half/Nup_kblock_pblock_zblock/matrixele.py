# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 23:39:48
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-19 12:31:36



import numpy as np
import scipy.sparse as sp

from ......linalg.usenumba.numba_settings import njit, config, numba_cache_dir, pnjit, prange
from ..bitsoperation import flip, operateon, findstate
from .defbasis import representative, get_nmc


config.CACHE_DIR = numba_cache_dir
@njit("f8[:,:](i8,i8,i8,i8,i8,f8,f8,i8[:],i8[:],i8[:],i8[:])")
def heisenberg_matrix_element(L:int, M:int, k:int, p:int, z:int, jxy:float, jz:float, s_list:np.ndarray, R_list:np.ndarray, m_list:np.ndarray, c_list:np.ndarray) -> np.ndarray:
    r"""
    \sum_{i=1}^N jxy * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1}) + jz * s^z_i s^z_{i+1}

    Examples
    >>> L = 6
    >>> Nup = 3
    >>> k = 1
    >>> p = 1
    >>> z = 1
    >>> M, s_list, R_list, m_list, c_list = construct_Nup_kblock_pblock_zblock_basis(L=L, k=k, p=p, z=z)
    >>> mat = heisenberg_chain_in_Nup_kblock_plock_zblock(L, M, k, p, z, jxy=1., jz=1., s_list=s_list, R_list=R_list, m_list=m_list, c_list=c_list)
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
            sb, l, q, g = representative(flipsa, L)
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

                        c_a_p = c_list[ii]
                        if c_a_p > 4:
                            c_a = 5
                            n_a = c_a_p - 6
                        else:
                            c_a = c_a_p
                            n_a = -1

                        c_b_p = c_list[jj]
                        if c_b_p > 4:
                            c_b = 5
                            n_b = c_b_p - 6
                        else:
                            c_b = c_b_p
                            n_b = -1

                        helement = jxy * 0.5 * (sigma_a * p) ** q * z**g
                        nmc_a = get_nmc(L, k, p, z, sigma_a, mu_a, n_a, c_a)
                        nmc_b = get_nmc(L, k, p, z, sigma_b, mu_b, n_b, c_b)
                        helement *= (np.abs(R_list[ii]/R_list[jj]) * nmc_b/nmc_a )**0.5
                        # print(f"sa = {sa}, sb = {sb}, sigma_a={sigma_a}, q={q}, c_b={c_b}")
                        # print(f"helement = {helement}")

                        if sigma_a == sigma_b:
                            if c_b in [1,3]:
                                helement *= np.cos(k*l*2*np.pi/L)
                            elif c_b in [2,5]:
                                helement *= np.cos(k*l*2*np.pi/L) + sigma_b * p * np.cos(k*(l - mu_b)*2*np.pi/L)
                                helement /= 1 + sigma_b * p * np.cos(k*mu_b*2*np.pi/L)
                            else:
                                helement *= np.cos(k*l*2*np.pi/L) + sigma_b * p * z * np.cos(k*(l - mu_b)*2*np.pi/L)
                                helement /= 1 + sigma_b * p * z * np.cos(k*mu_b*2*np.pi/L)

                        else:
                            if c_b in [1,3]:
                                helement *= sigma_b * np.sin(k*l*2*np.pi/L)
                            elif c_b in [2,5]:
                                helement *= sigma_b * np.sin(k*l*2*np.pi/L) + p * np.sin(k*(l - mu_b)*2*np.pi/L)
                                helement /= 1 + sigma_b * p * np.cos(k*mu_b*2*np.pi/L)
                                # print(f"helement = {helement}, l = {l}, {np.sin(k*l*2*np.pi/N)}")
                            else:
                                helement *= sigma_b * np.sin(k*l*2*np.pi/L) + p * z * np.sin(k*(l - mu_b)*2*np.pi/L)
                                helement /= 1 + sigma_b * p * z * np.cos(k*mu_b*2*np.pi/L)

                        H[jj,ii] = H[jj,ii] + helement
    return H


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element(opnm, posn, coef, L, k, p, z, M, s_list, R_list, m_list, c_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    tmpval = np.exp(-1j * 2 * np.pi * k / L)
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
            sb, l, q, g = representative(s, L)
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
                        
                        c_a_p = c_list[ii]
                        if c_a_p > 4:
                            c_a = 5
                            n_a = c_a_p - 6
                        else:
                            c_a = c_a_p
                            n_a = -1
                            
                        c_b_p = c_list[jj]
                        if c_b_p > 4:
                            c_b = 5
                            n_b = c_b_p - 6
                        else:
                            c_b = c_b_p
                            n_b = -1
                        
                        helement = coef * (sigma_a * p) ** q * z**g
                        nmc_a = get_nmc(L, k, p, z, sigma_a, mu_a, n_a, c_a)
                        nmc_b = get_nmc(L, k, p, z, sigma_b, mu_b, n_b, c_b)
                        helement *= (np.abs(R_list[ii]/R_list[jj]) * nmc_b/nmc_a )**0.5
                        # print(f"sa = {sa}, sb = {sb}, sigma_a={sigma_a}, q={q}, c_b={c_b}")
                        # print(f"helement = {helement}")
                        
                        if sigma_a == sigma_b:
                            if c_b in [1,3]:
                                helement *= np.cos(k*l*2*np.pi/L)
                            elif c_b in [2,5]:
                                helement *= np.cos(k*l*2*np.pi/L) + sigma_b * p * np.cos(k*(l - mu_b)*2*np.pi/L)
                                helement /= 1 + sigma_b * p * np.cos(k*mu_b*2*np.pi/L)
                            else:
                                helement *= np.cos(k*l*2*np.pi/L) + sigma_b * p * z * np.cos(k*(l - mu_b)*2*np.pi/L)
                                helement /= 1 + sigma_b * p * z * np.cos(k*mu_b*2*np.pi/L)
                        else:
                            if c_b in [1,3]:
                                helement *= sigma_b * np.sin(k*l*2*np.pi/L)
                            elif c_b in [2,5]:
                                helement *= sigma_b * np.sin(k*l*2*np.pi/L) + p * np.sin(k*(l - mu_b)*2*np.pi/L)
                                helement /= 1 + sigma_b * p * np.cos(k*mu_b*2*np.pi/L)
                                # print(f"helement = {helement}, l = {l}, {np.sin(k*l*2*np.pi/N)}")
                            else:
                                helement *= sigma_b * np.sin(k*l*2*np.pi/L) + p * z * np.sin(k*(l - mu_b)*2*np.pi/L)
                                helement /= 1 + sigma_b * p * z * np.cos(k*mu_b*2*np.pi/L)
                        
                        row_init[4*a+(ii-a)+2*(jj-b)] = jj
                        col_init[4*a+(ii-a)+2*(jj-b)] = ii
                        ME_init[4*a+(ii-a)+2*(jj-b)] = opco * helement

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]
