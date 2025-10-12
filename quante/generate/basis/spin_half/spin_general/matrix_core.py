# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-28 15:16:13
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-12 16:22:51

import numpy as np

from .....basicfun.utils_numba import njit, config, numba_cache_dir, pnjit, prange
from ..bitsoperation import operateon, findstate, perm_operation
from .basis_core import _sign, _coef


###########################################
# Z2N
##############################################

config.CACHE_DIR = numba_cache_dir
@njit
def representative_Z2N(s, ps):
    l = len(ps)
    oslis = np.zeros(1<<l, dtype=np.int64)
    oslis[0] = s
    cur_pos = 1
    for i in range(l):
        p = ps[i]
        for j in range(cur_pos):
            t = perm_operation(oslis[j], p)
            oslis[cur_pos+j] = t
        cur_pos *= 2
    res = np.argmin(oslis)
    return oslis[res], res


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_Z2N(opnm, posn, coef, L, perm, block, M, s_list, R_list, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)
    
    for a in prange(M):
        sa = s_list[a]
        
        opco, s = operateon(opnm, posn, sa)
        if s != -1:
            sb, ls = representative_Z2N(s, perm)
            b = findstate(s_list, sb)
            if b >= 0:
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * _sign(ls, block)


    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]


###########################################
# ZNN
##############################################

config.CACHE_DIR = numba_cache_dir
@njit
def representative_ZNN(s, ps, ns):
    l = len(ps)
    dim = np.prod(ns)
    oslis = np.zeros(dim, dtype=np.int64)
    record_op = np.zeros((dim, l), dtype=np.int64)
    oslis[0] = s
    cur_pos = 1

    for p_ind in range(l):
        # for each symmetry operation
        p = ps[p_ind]
        n = ns[p_ind]
        ct = 0
        for j in range(cur_pos):
            # for each basis
            t = oslis[j]
            for k in range(1,n):
                # apply n times
                t = perm_operation(t, p)
                if t == s:
                    break
                oslis[cur_pos+ct] = t
                record_op[cur_pos+ct] = record_op[j]
                record_op[cur_pos+ct, p_ind] = n-k
                ct += 1
        cur_pos += ct
    res_indx = np.argmin(oslis[:cur_pos])
    return oslis[res_indx], record_op[res_indx]


config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_ZNN(opnm, posn, coef, L, perm, block, M, s_list, R_list, ns, row_init, col_init, ME_init):
    
    ME_init.fill(0.0)

    permlen = len(perm)
    tmpval = np.zeros(permlen, dtype=np.complex128)
    for i in range(permlen):
        tmpval[i] = np.exp(-1j * 2 * np.pi * block[i] / ns[i])
    
    for a in prange(M):
        sa = s_list[a]
        
        opco, s = operateon(opnm, posn, sa)
        if s != -1:
            sb, ls = representative_ZNN(s, perm, ns)
            b = findstate(s_list, sb)
            if b >= 0:
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * _coef(ls, tmpval)


    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]
 

# from .....basicfun.utils_numba import njit, config, numba_cache_dir, pnjit, prange
# from ..bitsoperation import operateon, findstate, perm_operation
# from .basis_core import representative_Z21, representative_Z22, representative_Z23, representative_Z2N, _sign


# ###########################################
# # Z21
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_Z21(opnm, posn, coef, L, perm, p, M, s_list, row_init, col_init, ME_init):
    
#     ME_init.fill(0.0)
    
#     for a in prange(M):
#         sa = s_list[a]
        
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l = representative_Z21(s, perm)
#             b = findstate(s_list, sb)
#             if b >= 0:
#                 Na = 4 if perm_operation(sa, perm) == sa else 2
#                 Nb = 4 if perm_operation(sb, perm) == sb else 2

                
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (Nb/Na) ** 0.5 * (-1)**(p*l)

#                 # print(opnm, posn, coef, a, ME_init[a])

#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]


# ############################################
# # Ndiff
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_Nup(opnm, posn, coef, L, M, s_list, row_init, col_init, ME_init):
    
#     ME_init.fill(0.0)
    
#     for a in prange(M):
#         sa = s_list[a]
        
#         opco, sb = operateon(opnm, posn, sa)
#         if sb != -1:
#             b = findstate(s_list, sb)
#             if b != -1:
                
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef

#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]

# ###########################################
# # Z22
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_Z22(opnm, posn, coef, L, perm0, perm1, p0, p1, M, s_list, R_list, row_init, col_init, ME_init):
    
#     ME_init.fill(0.0)
    
#     for a in prange(M):
#         sa = s_list[a]
        
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1 = representative_Z22(s, perm0, perm1)
#             b = findstate(s_list, sb)
#             if b >= 0:
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0 + p1*l1)

#                 # print(opnm, posn, coef, a, ME_init[a])

#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]


# ###########################################
# # Z23
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_Z23(opnm, posn, coef, L, perm0, perm1, perm2, p0, p1, p2, M, s_list, R_list, row_init, col_init, ME_init):
    
#     ME_init.fill(0.0)
    
#     for a in prange(M):
#         sa = s_list[a]
        
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1, l2 = representative_Z23(s, perm0, perm1, perm2)
#             b = findstate(s_list, sb)
#             if b >= 0:
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0 + p1*l1 + p2*l2)

#                 # print(opnm, posn, coef, a, ME_init[a])

#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]


