# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-28 13:01:08
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-28 14:26:34


import numpy as np
from ..bitsoperation import findstate, flip
from .....basicfun.utils_numba import njit, config, numba_cache_dir, pnjit, prange
from ..bitsoperation import operateon, findstate, reflectbits, invertbits
from .basis_core import (
    representative_kblock, 
    representative_kblock_pblock,
    representative_Nup_kblock_pblock,
    representative_Nup_kblock_pblock_zblock,
    get_nmc,
    representative_pblock,
    representative_pzblock,
    representative_zblock
)

#############################################
# noblock
#############################################
config.CACHE_DIR = numba_cache_dir
@pnjit('f8[:,:](i8,f8,f8,b1)')
def heisenberg_matrix_element_noblock(L, jxy, jz, cyclic):
    r"""
    \sum_{i=1}^L jxy * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1}) + hz * s^z_i s^z_{i+1}
    
    Examples
    >>> L = 6
    >>> mat = heisenberg_chain_noblock(L, jxy=1., jz=1.)
    """
    H = np.zeros((1<<L, 1<<L), dtype=float)
    last_site = L if cyclic else L - 1
    for a in prange(1<<L):
        for i in range(last_site):
            j = (i+1) % L
            if ((a >> i) & 1) == ((a >> j) & 1):
                H[a,a] += 0.25 * jz
            else:
                H[a,a] -= 0.25 * jz
                b = flip(a, i, j)
                if b != -1:
                    H[b,a] += 0.5 * jxy
    return H

config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_noblock(opnm, posn, coef, L, M, row_init, col_init, ME_init):
    ME_init.fill(0.0)
    
    for a in prange(M):
        
        opco, t = operateon(opnm, posn, a, L)
        if t != -1:
            row_init[a] = t
            col_init[a] = a
            ME_init[a] = opco * coef

    mask = np.logical_not(np.abs(ME_init)==0.0)
    
    return row_init[mask], col_init[mask], ME_init[mask]

config.CACHE_DIR = numba_cache_dir
@pnjit
def diag_matrix_element_noblock(opnm, posn, coef, L, M, dtype):
    diag = np.empty(M, dtype=dtype)
    for a in prange(M):
        opco, t = operateon(opnm, posn, a, L)
        if t < 0:
            diag[a] = 0.0
        else:
            diag[a] = opco * coef
    return diag

############################################
# Nup
#############################################
config.CACHE_DIR = numba_cache_dir
@pnjit("f8[:,:](i8,i8,i8[:],f8,f8,b1)")
def heisenberg_matrix_element_Nup(L:int, M:int, s_list:np.ndarray, jxy:float, jz:float, cyclic:bool) -> np.ndarray:
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
def single_sparse_matrix_element_Nup(opnm, posn, coef, L, M, s_list, row_init, col_init, ME_init):
    
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
def diag_matrix_element_Nup(opnm, posn, coef, L, M, s_list, dtype):
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
def project_Nup(state, Ns, s_list):
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


############################################
# pblock
############################################
config.CACHE_DIR = numba_cache_dir
@pnjit("f8[:,:](i8,i8,i8,f8,f8,i8[:],b1)")
def heisenberg_matrix_element_pblock(L:int, M:int, p:int, jxy:float, jz:float, s_list:np.ndarray, cyclic:bool) -> np.ndarray:
    r"""
    \sum_{i=1}^L jxy * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1}) + jz * s^z_i s^z_{i+1}

    Examples
    >>> L = 10; p = 2
    >>> M, s_list = construct_pblock_basis(L, p)
    >>> mat = heisenberg_chain_in_pblock(L, p, jxy=1., jz=1., s_list=s_list)
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

            sb, l = representative_pblock(flipsa, L)
            b = findstate(s_list, sb)
            if b >= 0:
                Na = 4 if reflectbits(sa, L) == sa else 2
                Nb = 4 if reflectbits(sb, L) == sb else 2
                H[b,a] += jxy * 0.5 * (Nb/Na) ** 0.5 * p**l
    return H


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_pblock(opnm, posn, coef, L, p, M, s_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    for a in prange(M):
        sa = s_list[a]
        
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            sb, l = representative_pblock(s, L)
            b = findstate(s_list, sb)
            if b >= 0:
                Na = 4 if reflectbits(sa, L) == sa else 2
                Nb = 4 if reflectbits(sb, L) == sb else 2
                
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (Nb/Na) ** 0.5 * p**l

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

###########################################
# zblock
##########################################
config.CACHE_DIR = numba_cache_dir
@pnjit("f8[:,:](i8,i8,i8,f8,f8,i8[:],b1)")
def heisenberg_matrix_element_zblock(L:int, M:int, z:int, jxy:float, jz:float, s_list:np.ndarray, cyclic:bool) -> np.ndarray:
    r"""
    \sum_{i=1}^L jxy * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1}) + jz * s^z_i s^z_{i+1}

    Examples
    >>> L = 10; z = 1
    >>> M, s_list = construct_zblock_basis(L, z)
    >>> mat = heisenberg_chain_in_zblock(L, z, jxy=1., jz=1., s_list=s_list)
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

            sb, l = representative_zblock(flipsa, L)
            b = findstate(s_list, sb)
            if b >= 0:
                H[b,a] += jxy * 0.5 * z**l
    return H


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_zblock(opnm, posn, coef, L, z, M, s_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    for a in prange(M):
        sa = s_list[a]
        
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            sb, l = representative_zblock(s, L)
            b = findstate(s_list, sb)
            if b != -1:
                
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * z**l

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

############################################
# pzblock
############################################
config.CACHE_DIR = numba_cache_dir
@pnjit("f8[:,:](i8,i8,i8,f8,f8,i8[:],b1)")
def heisenberg_matrix_element_pzblock(L:int, M:int, pz:int, jxy:float, jz:float, s_list:np.ndarray, cyclic:bool) -> np.ndarray:
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

            sb, l = representative_pzblock(flipsa, L)
            b = findstate(s_list, sb)
            if b >= 0:
                Na = 4 if reflectbits(invertbits(sa, L), L) == sa else 2
                Nb = 4 if reflectbits(invertbits(sb, L), L) == sb else 2
                H[b,a] += jxy * 0.5 * (Nb/Na) ** 0.5 * pz**l
    return H


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_pzblock(opnm, posn, coef, L, pz, M, s_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    for a in prange(M):
        sa = s_list[a]
        
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            sb, l = representative_pzblock(s, L)
            b = findstate(s_list, sb)
            if b != -1:
                Na = 4 if reflectbits(invertbits(sa, L), L) == sa else 2
                Nb = 4 if reflectbits(invertbits(sb, L), L) == sb else 2
                
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (Nb/Na) ** 0.5 * pz**l

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]


############################################
# kblock
############################################
config.CACHE_DIR = numba_cache_dir
@pnjit('c16[:,:](i8,i8,i8,f8,f8,i8[:],i8[:])')
def heisenberg_matrix_element_kblock(L:int, M:int, k:int, jxy:float, jz:float, s_list:np.ndarray, R_list:np.ndarray) -> np.ndarray:
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
                r, l = representative_kblock(s,L)
                b = findstate(s_list, r)
                if b >= 0:
                    H[b,a] += 0.5 * (R_list[a]/R_list[b])**0.5 * tmpval**l * jxy
    return H


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_kblock(opnm, posn, coef, L, k, M, s_list, R_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    tmpval = np.exp(-1j * 2 * np.pi * k / L)
    for a in prange(M):
        sa = s_list[a]
        
        opco, s = operateon(opnm, posn, sa, L)
        if s != -1:
            r, l = representative_kblock(s, L)
            b = findstate(s_list, r)
            if b >= 0:
                
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (R_list[a] / R_list[b])**0.5 * tmpval ** l

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

@njit
def project_kblock(state, L, k, Ns, s_list, R_list):
    M, N = state.shape
    res = np.zeros((Ns, N), dtype=np.complex128)
    for i in range(N):
        for a in range(M):
            stateai = state[a,i]
            if stateai == 0:
                continue
            r, l = representative_kblock(a, L)
            if r == -1:
                continue
            b = findstate(s_list, r)
            if b >= 0:
                res[b,i] += stateai * (R_list[b])**0.5/L * np.exp(-1j * 2 * np.pi * k * l / L)
    return res

#############################################
# kblock pblock
#############################################
config.CACHE_DIR = numba_cache_dir
@njit("f8[:,:](i8,i8,i8,i8,f8,f8,i8[:],i8[:],i8[:])")
def heisenberg_matrix_element_kblock_pblock(L:int, M:int, k:int, p:int, jxy:float, jz:float, s_list:np.ndarray, R_list:np.ndarray, m_list:np.ndarray) -> np.ndarray:
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
            sb, l, q = representative_kblock_pblock(flipsa, L)
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
def helement_kblock_pblock(coef, sigma_a, Ri_Rj, p, mu_a, q, k, L, mu_b, sigma_b, l):
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
def single_sparse_matrix_element_kblock_pblock(opnm, posn, coef, L, k, p, M, s_list, R_list, m_list, row_init, col_init, ME_init):
    
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
            sb, l, q = representative_kblock_pblock(s, L)
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
                        ME_init[4*a+(ii-a)+2*(jj-b)] = opco * helement_kblock_pblock(coef, sigma_a, Ri_Rj, p, mu_a, q, k, L, mu_b, sigma_b, l)

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]


############################################
# Nup kblock pblock
############################################
config.CACHE_DIR = numba_cache_dir
@njit("f8[:,:](i8,i8,i8,i8,f8,f8,i8[:],i8[:],i8[:])")
def heisenberg_matrix_element_Nup_kblock_pblock(L:int, M:int, k:int, p:int, jxy:float, jz:float, s_list:np.ndarray, R_list:np.ndarray, m_list:np.ndarray) -> np.ndarray:
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
            sb, l, q = representative_Nup_kblock_pblock(flipsa, L)
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
def helement_Nup_kblock_pblock(coef, sigma_a, Ri_Rj, p, mu_a, q, k, L, mu_b, sigma_b, l):
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
def single_sparse_matrix_element_Nup_kblock_pblock(opnm, posn, coef, L, k, p, M, s_list, R_list, m_list, row_init, col_init, ME_init):
    
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
            sb, l, q = representative_Nup_kblock_pblock(s, L)
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
                        ME_init[4*a+(ii-a)+2*(jj-b)] = opco * helement_Nup_kblock_pblock(coef, sigma_a, Ri_Rj, p, mu_a, q, k, L, mu_b, sigma_b, l)

    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

############################################
# Nup kblock pblock zblock
############################################
config.CACHE_DIR = numba_cache_dir
@njit("f8[:,:](i8,i8,i8,i8,i8,f8,f8,i8[:],i8[:],i8[:],i8[:])")
def heisenberg_matrix_element_Nup_kblock_pblock_zblock(L:int, M:int, k:int, p:int, z:int, jxy:float, jz:float, s_list:np.ndarray, R_list:np.ndarray, m_list:np.ndarray, c_list:np.ndarray) -> np.ndarray:
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
            sb, l, q, g = representative_Nup_kblock_pblock_zblock(flipsa, L)
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
def single_sparse_matrix_element_Nup_kblock_pblock_zblock(opnm, posn, coef, L, k, p, z, M, s_list, R_list, m_list, c_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    # tmpval = np.exp(-1j * 2 * np.pi * k / L)
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
            sb, l, q, g = representative_Nup_kblock_pblock_zblock(s, L)
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

