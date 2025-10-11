# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-01 18:06:39
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-11 19:49:31

import numpy as np

from .....basicfun.utils_numba import config, numba_cache_dir, pnjit, prange
from ..bitsoperation import operateon, findstate
from .basis_core import representative_Z2N, _sign


###########################################
# Z2N
##############################################

config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_Z2N_sym(opnm, posn, coef, L, ps, bs, anci_perm, R_list, N_sym, N, s_list, row_init, col_init, ME_init):
    ME_init.fill(0.0)
    l = len(ps)
    for a in prange(N_sym):
        sa = s_list[a]
        opco, s = operateon(opnm, posn, sa)
        if s != -1:
            sb, ls = representative_Z2N(s, ps, anci_perm)
            b = findstate(s_list[:N_sym], sb)
            if b >= 0:
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * _sign(ls, bs, False)
    for a in prange(N_sym, N):
        sa = s_list[a]
        opco, s = operateon(opnm, posn, sa)
        if s != -1:
            sb, ls = representative_Z2N(s, ps, anci_perm)
            b = findstate(s_list[N_sym:], sb)
            if b >= 0:
                b = b + N_sym
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * _sign(ls, bs, True)
    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

config.CACHE_DIR = numba_cache_dir
@pnjit
def single_sparse_matrix_element_Z2N_asym(opnm, posn, coef, L, ps, bs, anci_perm, R_list, N_sym, N, s_list, row_init, col_init, ME_init):
    ME_init.fill(0.0)
    for a in prange(N_sym):
        sa = s_list[a]
        opco, s = operateon(opnm, posn, sa)
        if s != -1:
            sb, ls = representative_Z2N(s, ps, anci_perm)
            b = findstate(s_list[N_sym:], sb)
            if b >= 0:
                b = b + N_sym
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = -opco * coef * (R_list[a]/R_list[b])**0.5 * _sign(ls, bs, True)
    for a in prange(N_sym, N):
        sa = s_list[a]
        opco, s = operateon(opnm, posn, sa)
        if s != -1:
            sb, ls = representative_Z2N(s, ps, anci_perm)
            b = findstate(s_list[:N_sym], sb)
            if b >= 0:
                row_init[a] = b
                col_init[a] = a
                ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * _sign(ls, bs, False)
    mask = np.logical_not(np.abs(ME_init)==0.0)
    return row_init[mask], col_init[mask], ME_init[mask]

# from .....basicfun.utils_numba import njit, config, numba_cache_dir, pnjit, prange
# from ..bitsoperation import operateon, findstate, perm_operation
# from .basis_core import representative_Z21, representative_Z22, representative_Z23, representative_Z2N, _sign

# ###########################################
# # full
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_full_sym(opnm, posn, coef, L, perm, N_sym, N, s_list, row_init, col_init, ME_init):
#     ME_init.fill(0.0)
#     for a in prange(N_sym):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l = representative_Z21(s, perm)
#             b = findstate(s_list[:N_sym], sb)
#             if b >= 0:
#                 Na = 4 if perm_operation(sa, perm) == sa else 2
#                 Nb = 4 if perm_operation(sb, perm) == sb else 2
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (Nb/Na) ** 0.5
#     for a in prange(N_sym, N):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l = representative_Z21(s, perm)
#             b = findstate(s_list[N_sym:], sb)
#             if b >= 0:
#                 Na = 4 if perm_operation(sa, perm) == sa else 2
#                 Nb = 4 if perm_operation(sb, perm) == sb else 2
#                 row_init[a] = b+N_sym
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (Nb/Na) ** 0.5 * (-1)**l
#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]




# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_full_asym(opnm, posn, coef, L, perm, M_sym, M, s_list, row_init, col_init, ME_init):
#     ME_init.fill(0.0)
#     for a in prange(M_sym):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l = representative_Z21(s, perm)
#             b = findstate(s_list[M_sym:], sb)
#             if b >= 0:
#                 Na = 4 if perm_operation(sa, perm) == sa else 2
#                 Nb = 4 if perm_operation(sb, perm) == sb else 2
#                 row_init[a] = M_sym+b
#                 col_init[a] = a
#                 ME_init[a] = -opco * coef * (Nb/Na) ** 0.5 * (-1)**l
#     for a in prange(M_sym, M):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l = representative_Z21(s, perm)
#             b = findstate(s_list[:M_sym], sb)
#             if b >= 0:
#                 Na = 4 if perm_operation(sa, perm) == sa else 2
#                 Nb = 4 if perm_operation(sb, perm) == sb else 2
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (Nb/Na) ** 0.5

#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]


# ###########################################
# # Z21
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_Z21_sym(opnm, posn, coef, L, perm0, p0, anci_perm, R_list, N_sym, N, s_list, row_init, col_init, ME_init):
#     ME_init.fill(0.0)
#     for a in prange(N_sym):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1 = representative_Z22(s, perm0, anci_perm)
#             b = findstate(s_list[:N_sym], sb)
#             if b >= 0:
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0)
#     for a in prange(N_sym, N):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1 = representative_Z22(s, perm0, anci_perm)
#             b = findstate(s_list[N_sym:], sb)
#             if b >= 0:
#                 b = b + N_sym
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0 + l1)
#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]

# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_Z21_asym(opnm, posn, coef, L, perm0, p0, anci_perm, R_list, N_sym, N, s_list, row_init, col_init, ME_init):
#     ME_init.fill(0.0)
#     for a in prange(N_sym):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1 = representative_Z22(s, perm0, anci_perm) 
#             b = findstate(s_list[N_sym:], sb)
#             if b >= 0:
#                 b = b + N_sym
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = -opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0 + l1)
#     for a in prange(N_sym, N):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1 = representative_Z22(s, perm0, anci_perm)
#             b = findstate(s_list[:N_sym], sb)
#             if b >= 0:
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0)
#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]


# ###########################################
# # Z22
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_Z22_sym(opnm, posn, coef, L, perm0, p0, perm1, p1, anci_perm, R_list, N_sym, N, s_list, row_init, col_init, ME_init):
#     ME_init.fill(0.0)
#     for a in prange(N_sym):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1, l2 = representative_Z23(s, perm0, perm1, anci_perm)
#             b = findstate(s_list[:N_sym], sb)
#             if b >= 0:
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0 + p1*l1)
#     for a in prange(N_sym, N):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1, l2 = representative_Z23(s, perm0, perm1, anci_perm)
#             b = findstate(s_list[N_sym:], sb)
#             if b >= 0:
#                 b = b + N_sym
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0 + p1*l1 + l2)
#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]

# config.CACHE_DIR = numba_cache_dir
# @pnjit
# def single_sparse_matrix_element_Z22_asym(opnm, posn, coef, L, perm0, p0, perm1, p1, anci_perm, R_list, N_sym, N, s_list, row_init, col_init, ME_init):
#     ME_init.fill(0.0)
#     for a in prange(N_sym):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1, l2 = representative_Z23(s, perm0, perm1, anci_perm) 
#             b = findstate(s_list[N_sym:], sb)
#             if b >= 0:
#                 b = b + N_sym
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = -opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0 + p1*l1 + l2)
#     for a in prange(N_sym, N):
#         sa = s_list[a]
#         opco, s = operateon(opnm, posn, sa)
#         if s != -1:
#             sb, l0, l1, l2 = representative_Z23(s, perm0, perm1, anci_perm)
#             b = findstate(s_list[:N_sym], sb)
#             if b >= 0:
#                 row_init[a] = b
#                 col_init[a] = a
#                 ME_init[a] = opco * coef * (R_list[a]/R_list[b])**0.5 * (-1)**(p0*l0 + p1*l1)
#     mask = np.logical_not(np.abs(ME_init)==0.0)
#     return row_init[mask], col_init[mask], ME_init[mask]



