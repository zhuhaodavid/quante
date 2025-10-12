# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-01 15:43:21
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-12 17:22:06

import numpy as np
from .....basicfun.utils_numba import njit, config, numba_cache_dir
from ..bitsoperation import perm_operation, count_tot_down, next_combination

config.CACHE_DIR = numba_cache_dir
@njit(inline='always')
def _sign(ls, bs, sym):
    l = len(bs)
    sign = 1
    for i in range(l):
        ii = (ls >> i) & 1
        bi = bs[i]
        sign *= (-1)**(bi*ii)
    if sym:
        sign *= (-1)**((ls>>l) & 1)
    return sign

config.CACHE_DIR = numba_cache_dir
@njit
def count_nonequal(lis):
    """equal to len(set(lis)), but implemented in numba"""
    tmplist = []
    count = 0
    for i in lis:
        isnew = True
        for j in tmplist:
            if i == j:
                isnew = False
                break
        if isnew:
            count += 1
            tmplist.append(i)
    return count

###########################################
# Z2N
###########################################
config.CACHE_DIR = numba_cache_dir
@njit
def is_repr_super_Z2N(s, ps, bs, anci_perm):
    l = len(ps)

    oslis = np.zeros(1<<(l+1), dtype=np.int64)
    bslis = np.zeros(1<<(l+1), dtype=np.int64)
    oslis[0] = s
    bslis[0] = 0
    cur_pos = 1
    for i in range(l):
        p = ps[i]
        b = bs[i]
        for j in range(cur_pos):
            t = perm_operation(oslis[j], p)
            if t < s:
                return False, False, -1
            elif t == s:
                if (bslis[j] + b) % 2 == 1:
                    return False, False, -1
            oslis[cur_pos+j] = t
            bslis[cur_pos+j] = bslis[j] + b
        cur_pos *= 2
    
    sym, asym = True, True
    for j in range(cur_pos):
        t = perm_operation(oslis[j], anci_perm)
        if t < s:
            return False, False, -1
        elif t == s:
            if bslis[j] % 2 == 1:
                sym = False
            else:
                asym = False
        oslis[cur_pos+j] = t
    return sym, asym, count_nonequal(oslis)

config.CACHE_DIR = numba_cache_dir
@njit
def construct_basis_Z2N(L, ps, bs, anci_perm):
    s_list_sym = []
    s_list_asym = []
    R_list_sym = []
    R_list_asym = []
    for s in range(1<<L):
        issym, isasym, r = is_repr_super_Z2N(s, ps, bs, anci_perm)
        if issym:
            s_list_sym.append(s)
            R_list_sym.append(r)
        if isasym:
            s_list_asym.append(s)
            R_list_asym.append(r)
    dim_sym = len(s_list_sym)
    dim = dim_sym + len(s_list_asym)
    s_list = np.zeros(dim, dtype=np.int64)
    R_list = np.zeros(dim, dtype=np.int64)
    s_list_sym = np.array(s_list_sym, dtype=np.int64)
    s_list_asym = np.array(s_list_asym, dtype=np.int64)
    R_list_sym = np.array(R_list_sym, dtype=np.int64)
    R_list_asym = np.array(R_list_asym, dtype=np.int64)
    indx_sym = np.argsort(s_list_sym)
    indx_asym = np.argsort(s_list_asym)
    s_list[:dim_sym] = s_list_sym[indx_sym]
    s_list[dim_sym:] = s_list_asym[indx_asym]
    R_list[:dim_sym] = R_list_sym[indx_sym]
    R_list[dim_sym:] = R_list_asym[indx_asym]
    return s_list, R_list, dim_sym, dim - dim_sym


config.CACHE_DIR = numba_cache_dir
@njit
def representative_Z2N(s, ps, anci_perm):
    l = len(ps)
    oslis = np.zeros(1<<(l+1), dtype=np.int64)
    oslis[0] = s
    cur_pos = 1
    for i in range(l):
        p = ps[i]
        for j in range(cur_pos):
            t = perm_operation(oslis[j], p)
            oslis[cur_pos+j] = t
        cur_pos *= 2
    for j in range(cur_pos):
        oslis[cur_pos+j] = perm_operation(oslis[j], anci_perm)
    res = np.argmin(oslis)
    return oslis[res], res


config.CACHE_DIR = numba_cache_dir
@njit
def construct_Ndiff_basis_Z2N(L: int, Ndiff: np.ndarray, ps, bs, anci_perm, flipmask) -> tuple[int, np.ndarray]:
    s_list_sym = []
    s_list_asym = []
    R_list_sym = []
    R_list_asym = []
    l = count_tot_down(flipmask)
    for ndiff in Ndiff:
        Nup = l + ndiff
        if 2*L == Nup:
            t = 0 ^ flipmask
            issym, isasym, r = is_repr_super_Z2N(t, ps, bs, anci_perm)
            if issym:
                s_list_sym.append(t)
                R_list_sym.append(r)
            if isasym:
                s_list_asym.append(t)
                R_list_asym.append(r)
            continue
        s = (1 << (2*L - Nup)) - 1
        while s < (1 << (2*L)):
            t = s ^ flipmask
            issym, isasym, r = is_repr_super_Z2N(t, ps, bs, anci_perm)
            if issym:
                s_list_sym.append(t)
                R_list_sym.append(r)
            if isasym:
                s_list_asym.append(t)
                R_list_asym.append(r)
            s = next_combination(s)

    dim_sym = len(s_list_sym)
    dim = dim_sym + len(s_list_asym)
    s_list = np.zeros(dim, dtype=np.int64)
    R_list = np.zeros(dim, dtype=np.int64)
    s_list_sym = np.array(s_list_sym, dtype=np.int64)
    s_list_asym = np.array(s_list_asym, dtype=np.int64)
    R_list_sym = np.array(R_list_sym, dtype=np.int64)
    R_list_asym = np.array(R_list_asym, dtype=np.int64)
    indx_sym = np.argsort(s_list_sym)
    indx_asym = np.argsort(s_list_asym)
    s_list[:dim_sym] = s_list_sym[indx_sym]
    s_list[dim_sym:] = s_list_asym[indx_asym]
    R_list[:dim_sym] = R_list_sym[indx_sym]
    R_list[dim_sym:] = R_list_asym[indx_asym]
    return s_list, R_list, dim_sym, dim - dim_sym


config.CACHE_DIR = numba_cache_dir
@njit
def construct_Nup2_basis_Z2N(L: int, Nup2: np.ndarray, ps, bs, anci_perm) -> tuple[int, np.ndarray]:
    s_list_sym = []
    s_list_asym = []
    R_list_sym = []
    R_list_asym = []
    for n1, n2 in Nup2:
        if L == n1:
            s1 = 0
            if L == n2:
                s2 = 0
                s = (s1 << L) | s2
                issym, isasym, r = is_repr_super_Z2N(s, ps, bs, anci_perm)
                if issym:
                    s_list_sym.append(s)
                    R_list_sym.append(r)
                if isasym:
                    s_list_asym.append(s)
                    R_list_asym.append(r)
            else:
                s2 = (1 << (L - n2)) - 1
                while s2 < (1 << L):
                    s = (s1 << L) | s2
                    issym, isasym, r = is_repr_super_Z2N(s, ps, bs, anci_perm)
                    if issym:
                        s_list_sym.append(s)
                        R_list_sym.append(r)
                    if isasym:
                        s_list_asym.append(s)
                        R_list_asym.append(r)
                    s2 = next_combination(s2)
            continue
        s1 = (1 << (L - n1)) - 1
        while s1 < (1 << L):
            if L == n2:
                s2 = 0
                s = (s1 << L) | s2
                issym, isasym, r = is_repr_super_Z2N(s, ps, bs, anci_perm)
                if issym:
                    s_list_sym.append(s)
                    R_list_sym.append(r)
                if isasym:
                    s_list_asym.append(s)
                    R_list_asym.append(r)
            else:
                s2 = (1 << (L-n2)) - 1
                while s2 < (1 << L):
                    s = (s1 << L) | s2
                    issym, isasym, r = is_repr_super_Z2N(s, ps, bs, anci_perm)
                    if issym:
                        s_list_sym.append(s)
                        R_list_sym.append(r)
                    if isasym:
                        s_list_asym.append(s)
                        R_list_asym.append(r)
                    s2 = next_combination(s2)
            s1 = next_combination(s1)

    dim_sym = len(s_list_sym)
    dim = dim_sym + len(s_list_asym)
    s_list = np.zeros(dim, dtype=np.int64)
    R_list = np.zeros(dim, dtype=np.int64)
    s_list_sym = np.array(s_list_sym, dtype=np.int64)
    s_list_asym = np.array(s_list_asym, dtype=np.int64)
    R_list_sym = np.array(R_list_sym, dtype=np.int64)
    R_list_asym = np.array(R_list_asym, dtype=np.int64)
    indx_sym = np.argsort(s_list_sym)
    indx_asym = np.argsort(s_list_asym)
    s_list[:dim_sym] = s_list_sym[indx_sym]
    s_list[dim_sym:] = s_list_asym[indx_asym]
    R_list[:dim_sym] = R_list_sym[indx_sym]
    R_list[dim_sym:] = R_list_asym[indx_asym]
    return s_list, R_list, dim_sym, dim - dim_sym

config.CACHE_DIR = numba_cache_dir
@njit
def proj_coef_Z2N(s, ps, anci_perm):
    l = len(ps)
    oslis = np.zeros(1<<(l+1), dtype=np.int64)
    oslis[0] = s
    cur_pos = 1
    for i in range(l):
        p = ps[i]
        for j in range(cur_pos):
            t = perm_operation(oslis[j], p)
            oslis[cur_pos+j] = t
        cur_pos *= 2
    for j in range(cur_pos):
        oslis[cur_pos+j] = perm_operation(oslis[j], anci_perm)

    res = []    
    ls = []
    for i in range(1<<(l+1)):
        o = oslis[i]
        if o not in res:
            res.append(o)
            ls.append(i)
    return res, ls


config.CACHE_DIR = numba_cache_dir
@njit
def projmat_Z2N(L, s_list, N_sym, Ns, anci_perm, ps, bs):
    dim = (2**(len(ps)+2))*Ns
    row = np.zeros(dim, dtype=np.int64)
    col = np.zeros(dim, dtype=np.int64)
    ele = np.zeros(dim, dtype=np.complex128)
    ct = 0
    for i in range(N_sym):
        t = s_list[i]
        news, ops = proj_coef_Z2N(t, ps, anci_perm)
        l = len(news)
        for j in range(l):
            row[ct], col[ct] = news[j], i
            ele[ct] = l**(-0.5) * _sign(ops[j], bs, False)
            ct += 1
    for i in range(N_sym, Ns):
        t = s_list[i]
        news, ops = proj_coef_Z2N(t, ps, anci_perm)
        l = len(news)
        for j in range(l):
            row[ct], col[ct] = news[j], i
            ele[ct] = 1j * l**(-0.5) * _sign(ops[j], bs, True)
            ct += 1
    return row[:ct], col[:ct], ele[:ct]


config.CACHE_DIR = numba_cache_dir
@njit
def project_Z2N(state, L, s_list, N_sym, Ns, anci_perm, ps, bs):
    M, N = state.shape
    res = np.zeros((Ns, N), dtype=np.complex128)
    for j in range(N):
        for i in range(N_sym):
            t = s_list[i]
            news, ops = proj_coef_Z2N(t, ps, anci_perm)
            l = len(news)
            for k in range(l):
                res[i, j] += l**(-0.5) * _sign(ops[k], bs, False) * state[news[k], j]
        for i in range(N_sym, Ns):
            t = s_list[i]
            news, ops = proj_coef_Z2N(t, ps, anci_perm)
            l = len(news)
            for k in range(l):
                res[i, j] += -1j * l**(-0.5) * _sign(ops[k], bs, True) * state[news[k], j]
    return res



config.CACHE_DIR = numba_cache_dir
@njit
def recover_Z2N(state, L, s_list, N_sym, Ns, anci_perm, ps, bs):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=np.complex128)
    for j in range(N):
        for i in range(N_sym):
            t = s_list[i]
            news, ops = proj_coef_Z2N(t, ps, anci_perm)
            l = len(news)
            for k in range(l):
                res[news[k], j] += l**(-0.5) * _sign(ops[k], bs, False) * state[i, j]
        for i in range(N_sym, Ns):
            t = s_list[i]
            news, ops = proj_coef_Z2N(t, ps, anci_perm)
            l = len(news)
            for k in range(l):
                res[news[k], j] += 1j * l**(-0.5) * _sign(ops[k], bs, True) * state[i, j]
    return res


# import numpy as np
# from .....basicfun.utils_numba import njit, config, numba_cache_dir
# from ..bitsoperation import perm_operation, count_tot_down, next_combination, findstate, comb, no_equal
# ###########################################
# # full
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_full_basis(L, anci_perm):
#     dim = 1 << (2*L)
#     s_list = np.zeros(dim, dtype=np.int64)
#     ct = 0
#     ct1 = (dim + (1<<L))//2
#     for s in range(dim):
#         s_prime = perm_operation(s, anci_perm)
#         if s_prime == s:
#             s_list[ct] = s
#             ct += 1
#         elif s < s_prime:
#             s_list[ct] = s
#             s_list[ct1] = s
#             ct += 1
#             ct1 += 1
#     return s_list, ct, dim-ct


# config.CACHE_DIR = numba_cache_dir
# @njit
# def representative_Z21(s, perm):
#     t = perm_operation(s, perm)
#     if t < s:
#         return t, 1
#     else:
#         return s, 0

# config.CACHE_DIR = numba_cache_dir
# @njit
# def projmat_full(L, s_list, N_sym, Ns, anci_perm):
#     dim = 3*Ns - 2*N_sym
#     row = np.empty(dim, dtype=np.int64)
#     col = np.empty(dim, dtype=np.int64)
#     ele = np.empty(dim, dtype=np.complex128)
#     ct = 0
#     for i in range(N_sym):
#         t = s_list[i]
#         tp = perm_operation(t, anci_perm)
#         if tp == t:
#             row[ct], col[ct], ele[ct] = t, i, 1.
#             ct += 1
#         else:
#             row[ct], col[ct], ele[ct] = t, i, 1/np.sqrt(2)
#             ct += 1
#             row[ct], col[ct], ele[ct] = tp, i, 1/np.sqrt(2)
#             ct += 1
#     for i in range(N_sym, Ns):
#         t = s_list[i]
#         tp = perm_operation(t, anci_perm)
#         row[ct], col[ct], ele[ct] = t, i, 1j/np.sqrt(2)
#         ct += 1
#         row[ct], col[ct], ele[ct] = tp, i, -1j/np.sqrt(2)
#         ct += 1
#     if ct != dim:
#         raise ValueError("Internal error in projmat_full")
#     return row, col, ele


# config.CACHE_DIR = numba_cache_dir
# @njit
# def project_full(state, L, s_list, N_sym, Ns, anci_perm, dtype):
#     M, N = state.shape
#     res = np.zeros((Ns, N), dtype=np.complex128)
#     for j in range(N):
#         for i in range(N_sym):
#             t = s_list[i]
#             tp = perm_operation(t, anci_perm)
#             if tp == t:
#                 res[i, j] += state[t, j]
#             else:
#                 res[i, j] += 1/np.sqrt(2) * (state[t, j] + state[tp, j])
#         for i in range(N_sym, Ns):
#             t = s_list[i]
#             tp = perm_operation(t, anci_perm)
#             res[i, j] += -1j/np.sqrt(2) * (state[t, j] - state[tp, j])
#     return res

# config.CACHE_DIR = numba_cache_dir
# @njit
# def recover_full(state, L, s_list, N_sym, Ns, anci_perm):
#     M, N = state.shape
#     res = np.zeros((1<<L, N), dtype=np.complex128)
#     for j in range(N):
#         for i in range(N_sym):
#             t = s_list[i]
#             tp = perm_operation(t, anci_perm)
#             if tp == t:
#                 res[t, j] += state[i, j]
#             else:
#                 res[t, j] += 1/np.sqrt(2) * state[i, j]
#                 res[tp, j] += 1/np.sqrt(2) * state[i, j]
#         for i in range(N_sym, Ns):
#             t = s_list[i]
#             tp = perm_operation(t, anci_perm)
#             res[t, j] += 1j/np.sqrt(2) * state[i, j]
#             res[tp, j] += -1j/np.sqrt(2) * state[i, j]
#     return res

# config.CACHE_DIR = numba_cache_dir
# @njit
# def _update_s_list(t ,tp, s_list, ct, ct1):
#     if tp == t:
#         s_list[ct] = t
#         ct += 1
#     elif t < tp:
#         s_list[ct] = t
#         s_list[ct1] = t
#         ct += 1
#         ct1 -= 1
#     return ct, ct1

# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Ndiff_basis(L: int, Ndiff: np.ndarray, anci_perm, flipmask) -> tuple[int, np.ndarray]:
#     dim = 0
#     l = count_tot_down(flipmask)
#     for ndiff in Ndiff:
#         dim += comb(2*L, l+ndiff)
#     s_list = np.zeros(dim, dtype=np.int64)
#     ct = 0
#     ct1 = dim-1
#     for ndiff in Ndiff:
#         Nup = l + ndiff
#         if 2*L == Nup:
#             t = 0 ^ flipmask
#             tp = perm_operation(t, anci_perm)
#             ct, ct1 = _update_s_list(t, tp, s_list, ct, ct1)
#             continue
#         s = (1 << (2*L - Nup)) - 1
#         while s < (1 << (2*L)):
#             t = s ^ flipmask
#             tp = perm_operation(t, anci_perm)
#             ct, ct1 = _update_s_list(t, tp, s_list, ct, ct1)
#             s = next_combination(s)
#     dim_sym = ct
#     if (ct != ct1 + 1):
#         raise ValueError("Internal error in construct_Ndiff_basis")
#     s_list[:dim_sym] = np.sort(s_list[:dim_sym]) 
#     s_list[dim_sym:] = np.sort(s_list[dim_sym:])
#     return s_list, dim_sym, dim-dim_sym


# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Nup2_basis(L: int, Nup2: np.ndarray, anci_perm) -> tuple[int, np.ndarray]:
#     dim = 0
#     for n1, n2 in Nup2:
#         dim += comb(L, n1) * comb(L, n2)
#     s_list = np.zeros(dim, dtype=np.int64)
#     ct = 0
#     ct1 = dim-1
#     sp_list = []
#     for n1, n2 in Nup2:
#         if L == n1:
#             s1 = 0
#             if L == n2:
#                 s2 = 0
#                 s = (s1 << L) | s2
#                 sp_list.append(s)
#                 tp = perm_operation(s, anci_perm)
#                 ct, ct1 = _update_s_list(s, tp, s_list, ct, ct1)
#             else:
#                 s2 = (1 << (L - n2)) - 1
#                 while s2 < (1 << L):
#                     s = (s1 << L) | s2
#                     sp_list.append(s)
#                     tp = perm_operation(s, anci_perm)
#                     ct, ct1 = _update_s_list(s, tp, s_list, ct, ct1)
#                     s2 = next_combination(s2)
#             continue
#         s1 = (1 << (L - n1)) - 1
#         while s1 < (1 << L):
#             if L == n2:
#                 s2 = 0
#                 s = (s1 << L) | s2
#                 sp_list.append(s)
#                 tp = perm_operation(s, anci_perm)
#                 ct, ct1 = _update_s_list(s, tp, s_list, ct, ct1)
#             else:
#                 s2 = (1 << (L-n2)) - 1
#                 while s2 < (1 << L):
#                     s = (s1 << L) | s2
#                     sp_list.append(s)
#                     tp = perm_operation(s, anci_perm)
#                     ct, ct1 = _update_s_list(s, tp, s_list, ct, ct1)
#                     s2 = next_combination(s2)
#             s1 = next_combination(s1)
#     dim_sym = ct
#     if (ct != ct1 + 1):
#         raise ValueError("Internal error in construct_Ndiff_basis")
#     s_list[:dim_sym] = np.sort(s_list[:dim_sym]) 
#     s_list[dim_sym:] = np.sort(s_list[dim_sym:])
#     sp_list = np.array(sp_list, dtype=np.int64)
#     sp_list = np.sort(sp_list)
#     return sp_list, s_list, dim_sym, dim-dim_sym

# config.CACHE_DIR = numba_cache_dir
# @njit
# def project_full_Nup2(state, L, s_list, s2_list, N_sym, Ns, anci_perm, dtype):
#     M, N = state.shape
#     res = np.zeros((Ns, N), dtype=np.complex128)
#     for j in range(N):
#         for i in range(N_sym):
#             x = s_list[i]
#             t = findstate(s2_list, x)
#             assert t >= 0
#             xp = perm_operation(x, anci_perm)
#             tp = findstate(s2_list, xp)
#             assert tp >= 0
#             if xp == x:
#                 res[i, j] += state[t, j]
#             else:
#                 res[i, j] += 1/np.sqrt(2) * (state[t, j] + state[tp, j])
#         for i in range(N_sym, Ns):
#             x = s_list[i]
#             t = findstate(s2_list, x)
#             assert t >= 0
#             xp = perm_operation(x, anci_perm)
#             tp = findstate(s2_list, xp)
#             assert tp >= 0
#             res[i, j] += -1j/np.sqrt(2) * (state[t, j] - state[tp, j])
#     return res



# ###########################################
# # Z21
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def is_repr_super_Z21(s, perm0, block0, anci_perm):
#     s_prime0 = perm_operation(s, perm0)
#     sym, asym = True, True
#     if (s_prime0 < s) or (block0==1 and s_prime0 == s):
#         return False, False, -1
#     s_prime1 = perm_operation(s, anci_perm)
#     if s_prime1 < s:
#         return False, False, -1
#     elif s_prime1 == s:
#         asym = False
#     s_prime01 = perm_operation(s_prime1, perm0)
#     if s_prime01 < s:
#         return False, False, -1
#     elif s_prime01 == s:
#         if block0 == 0:
#             asym = False
#         else:
#             sym = False

#     if s == s_prime0 == s_prime1 == s_prime01:
#         return sym, asym, 1  # 4 states combined into 1
#     elif no_equal([s, s_prime0, s_prime1, s_prime01]):
#         return sym, asym, 4  # 4 states combined into 4
#     else:
#         return sym, asym, 2  # 4 states combined into 2


# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_basis_Z21(L, perm0, block0, anci_perm):
#     s_list_sym = []
#     s_list_asym = []
#     R_list_sym = []
#     R_list_asym = []
#     for s in range(1<<L):
#         issym, isasym, r = is_repr_super_Z21(s, perm0, block0, anci_perm)
#         if issym:
#             s_list_sym.append(s)
#             R_list_sym.append(r)
#         if isasym:
#             s_list_asym.append(s)
#             R_list_asym.append(r)
#     dim_sym = len(s_list_sym)
#     dim = dim_sym + len(s_list_asym)
#     s_list = np.zeros(dim, dtype=np.int64)
#     R_list = np.zeros(dim, dtype=np.int64)
#     s_list_sym = np.array(s_list_sym, dtype=np.int64)
#     s_list_asym = np.array(s_list_asym, dtype=np.int64)
#     R_list_sym = np.array(R_list_sym, dtype=np.int64)
#     R_list_asym = np.array(R_list_asym, dtype=np.int64)
#     indx_sym = np.argsort(s_list_sym)
#     indx_asym = np.argsort(s_list_asym)
#     s_list[:dim_sym] = s_list_sym[indx_sym]
#     s_list[dim_sym:] = s_list_asym[indx_asym]
#     R_list[:dim_sym] = R_list_sym[indx_sym]
#     R_list[dim_sym:] = R_list_asym[indx_asym]
#     return s_list, R_list, dim_sym, dim - dim_sym

# config.CACHE_DIR = numba_cache_dir
# @njit
# def representative_Z22(s, perm0, perm1):
#     t0 = perm_operation(s, perm0)
#     t1 = perm_operation(s, perm1)
#     t01 = perm_operation(t1, perm0)
    
#     mint = min([s, t0, t1, t01]) 
#     if mint == s:
#         c0, c1 = 0, 0
#     elif mint == t0:
#         c0, c1 = 1, 0
#     elif mint == t1:
#         c0, c1 = 0, 1
#     else:  # mint == t01
#         c0, c1 = 1, 1
#     return mint, c0, c1

# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Ndiff_basis_Z21(L: int, Ndiff: np.ndarray, perm0, block0, anci_perm, flipmask) -> tuple[int, np.ndarray]:
#     s_list_sym = []
#     s_list_asym = []
#     R_list_sym = []
#     R_list_asym = []
#     l = count_tot_down(flipmask)
#     for ndiff in Ndiff:
#         Nup = l + ndiff
#         if 2*L == Nup:
#             t = 0 ^ flipmask
#             issym, isasym, r = is_repr_super_Z21(t, perm0, block0, anci_perm)
#             if issym:
#                 s_list_sym.append(t)
#                 R_list_sym.append(r)
#             if isasym:
#                 s_list_asym.append(t)
#                 R_list_asym.append(r)
#             continue
#         s = (1 << (2*L - Nup)) - 1
#         while s < (1 << (2*L)):
#             t = s ^ flipmask
#             issym, isasym, r = is_repr_super_Z21(t, perm0, block0, anci_perm)
#             if issym:
#                 s_list_sym.append(t)
#                 R_list_sym.append(r)
#             if isasym:
#                 s_list_asym.append(t)
#                 R_list_asym.append(r)
#             s = next_combination(s)

#     dim_sym = len(s_list_sym)
#     dim = dim_sym + len(s_list_asym)
#     s_list = np.zeros(dim, dtype=np.int64)
#     R_list = np.zeros(dim, dtype=np.int64)
#     s_list_sym = np.array(s_list_sym, dtype=np.int64)
#     s_list_asym = np.array(s_list_asym, dtype=np.int64)
#     R_list_sym = np.array(R_list_sym, dtype=np.int64)
#     R_list_asym = np.array(R_list_asym, dtype=np.int64)
#     indx_sym = np.argsort(s_list_sym)
#     indx_asym = np.argsort(s_list_asym)
#     s_list[:dim_sym] = s_list_sym[indx_sym]
#     s_list[dim_sym:] = s_list_asym[indx_asym]
#     R_list[:dim_sym] = R_list_sym[indx_sym]
#     R_list[dim_sym:] = R_list_asym[indx_asym]
#     return s_list, R_list, dim_sym, dim - dim_sym

# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Nup2_basis_Z21(L: int, Nup2: np.ndarray, perm0, block0, anci_perm) -> tuple[int, np.ndarray]:
#     s_list_sym = []
#     s_list_asym = []
#     R_list_sym = []
#     R_list_asym = []
#     for n1, n2 in Nup2:
#         if L == n1:
#             s1 = 0
#             if L == n2:
#                 s2 = 0
#                 s = (s1 << L) | s2
#                 issym, isasym, r = is_repr_super_Z21(s, perm0, block0, anci_perm)
#                 if issym:
#                     s_list_sym.append(s)
#                     R_list_sym.append(r)
#                 if isasym:
#                     s_list_asym.append(s)
#                     R_list_asym.append(r)
#             else:
#                 s2 = (1 << (L - n2)) - 1
#                 while s2 < (1 << L):
#                     s = (s1 << L) | s2
#                     issym, isasym, r = is_repr_super_Z21(s, perm0, block0, anci_perm)
#                     if issym:
#                         s_list_sym.append(s)
#                         R_list_sym.append(r)
#                     if isasym:
#                         s_list_asym.append(s)
#                         R_list_asym.append(r)
#                     s2 = next_combination(s2)
#             continue
#         s1 = (1 << (L - n1)) - 1
#         while s1 < (1 << L):
#             if L == n2:
#                 s2 = 0
#                 s = (s1 << L) | s2
#                 issym, isasym, r = is_repr_super_Z21(s, perm0, block0, anci_perm)
#                 if issym:
#                     s_list_sym.append(s)
#                     R_list_sym.append(r)
#                 if isasym:
#                     s_list_asym.append(s)
#                     R_list_asym.append(r)
#             else:
#                 s2 = (1 << (L-n2)) - 1
#                 while s2 < (1 << L):
#                     s = (s1 << L) | s2
#                     issym, isasym, r = is_repr_super_Z21(s, perm0, block0, anci_perm)
#                     if issym:
#                         s_list_sym.append(s)
#                         R_list_sym.append(r)
#                     if isasym:
#                         s_list_asym.append(s)
#                         R_list_asym.append(r)
#                     s2 = next_combination(s2)
#             s1 = next_combination(s1)

#     dim_sym = len(s_list_sym)
#     dim = dim_sym + len(s_list_asym)
#     s_list = np.zeros(dim, dtype=np.int64)
#     R_list = np.zeros(dim, dtype=np.int64)
#     s_list_sym = np.array(s_list_sym, dtype=np.int64)
#     s_list_asym = np.array(s_list_asym, dtype=np.int64)
#     R_list_sym = np.array(R_list_sym, dtype=np.int64)
#     R_list_asym = np.array(R_list_asym, dtype=np.int64)
#     indx_sym = np.argsort(s_list_sym)
#     indx_asym = np.argsort(s_list_asym)
#     s_list[:dim_sym] = s_list_sym[indx_sym]
#     s_list[dim_sym:] = s_list_asym[indx_asym]
#     R_list[:dim_sym] = R_list_sym[indx_sym]
#     R_list[dim_sym:] = R_list_asym[indx_asym]
#     return s_list, R_list, dim_sym, dim - dim_sym

# config.CACHE_DIR = numba_cache_dir
# @njit
# def projmat_Z21(L, s_list, N_sym, Ns, anci_perm, perm0, block0):
#     dim = 4*Ns
#     row = np.empty(dim, dtype=np.int64)
#     col = np.empty(dim, dtype=np.int64)
#     ele = np.empty(dim, dtype=np.complex128)
#     ct = 0
#     for i in range(N_sym):
#         t = s_list[i]
#         t1 = perm_operation(t, anci_perm)
#         t2 = perm_operation(t, perm0)
#         t3 = perm_operation(t1, perm0)
#         if t == t1 == t2 == t3:
#             row[ct], col[ct], ele[ct] = t, i, 1.
#             ct += 1
#         elif no_equal([t, t1, t2, t3]):
#             row[ct], col[ct], ele[ct] = t, i, 1/2
#             ct += 1
#             row[ct], col[ct], ele[ct] = t1, i, 1/2
#             ct += 1
#             row[ct], col[ct], ele[ct] = t2, i, 1/2 * (-1)**block0
#             ct += 1
#             row[ct], col[ct], ele[ct] = t3, i, 1/2 * (-1)**block0
#             ct += 1
#         elif t == t1:
#             row[ct], col[ct], ele[ct] = t, i, 1/np.sqrt(2)
#             ct += 1
#             row[ct], col[ct], ele[ct] = t2, i, 1/np.sqrt(2) * (-1)**block0
#             ct += 1
#         else:
#             row[ct], col[ct], ele[ct] = t, i, 1/np.sqrt(2)
#             ct += 1
#             row[ct], col[ct], ele[ct] = t1, i, 1/np.sqrt(2)
#             ct += 1
#     for i in range(N_sym, Ns):
#         t = s_list[i]
#         t1 = perm_operation(t, anci_perm)
#         t2 = perm_operation(t, perm0)
#         t3 = perm_operation(t1, perm0)
#         if t == t1 == t2 == t3:
#             row[ct], col[ct], ele[ct] = t, i, 1j
#             ct += 1
#         elif no_equal([t, t1, t2, t3]):
#             row[ct], col[ct], ele[ct] = t, i, 1j/2
#             ct += 1
#             row[ct], col[ct], ele[ct] = t1, i, -1j/2
#             ct += 1
#             row[ct], col[ct], ele[ct] = t2, i, 1j/2 * (-1)**block0
#             ct += 1
#             row[ct], col[ct], ele[ct] = t3, i, -1j/2 * (-1)**block0
#             ct += 1
#         else:
#             row[ct], col[ct], ele[ct] = t, i, 1j/np.sqrt(2)
#             ct += 1
#             row[ct], col[ct], ele[ct] = t1, i, -1j/np.sqrt(2)
#             ct += 1
#     return row[:ct], col[:ct], ele[:ct]


# config.CACHE_DIR = numba_cache_dir
# @njit
# def project_Z21(state, L, s_list, N_sym, Ns, anci_perm, perm0, block0):
#     M, N = state.shape
#     res = np.zeros((Ns, N), dtype=np.complex128)
#     for j in range(N):
#         for i in range(N_sym):
#             t = s_list[i]
#             t1 = perm_operation(t, anci_perm)
#             t2 = perm_operation(t, perm0)
#             t3 = perm_operation(t1, perm0)
#             if t == t1 == t2 == t3:
#                 res[i, j] += state[t, j]
#             elif no_equal([t, t1, t2, t3]):
#                 res[i, j] += (state[t, j] + state[t1, j] + (-1)**block0 * state[t2, j] + (-1)**block0 * state[t3, j]) / 2
#             elif t == t1:
#                 res[i, j] += (state[t, j] + (-1)**block0 * state[t2, j]) / np.sqrt(2)
#             else:
#                 res[i, j] += (state[t, j] + state[t1, j]) / np.sqrt(2)
#         for i in range(N_sym, Ns):
#             t = s_list[i]
#             t1 = perm_operation(t, anci_perm)
#             t2 = perm_operation(t, perm0)
#             t3 = perm_operation(t1, perm0)
#             if t == t1 == t2 == t3:
#                 res[i, j] += - 1j * state[t, j]
#             elif no_equal([t, t1, t2, t3]):
#                 res[i, j] += -1j/2 * (
#                     state[t, j] 
#                     - state[t1, j] 
#                     + (-1)**block0 * state[t2, j] 
#                     - (-1)**block0 * state[t3, j]
#                 )
#             else:
#                 res[i, j] += -1j/np.sqrt(2) * (
#                     state[t, j] - state[t1, j]
#                 )
#     return res


# config.CACHE_DIR = numba_cache_dir
# @njit
# def recover_Z21(state, L, s_list, N_sym, Ns, anci_perm, perm0, block0):
#     M, N = state.shape
#     res = np.zeros((1<<L, N), dtype=np.complex128)
#     for j in range(N):
#         for i in range(N_sym):
#             t = s_list[i]
#             t1 = perm_operation(t, anci_perm)
#             t2 = perm_operation(t, perm0)
#             t3 = perm_operation(t1, perm0)
#             if t == t1 == t2 == t3:
#                 res[t, j] += state[i, j]
#             elif no_equal([t, t1, t2, t3]):
#                 res[t, j] += state[i, j] / 2
#                 res[t1, j] += state[i, j] / 2
#                 res[t2, j] += state[i, j] / 2 * (-1)**block0
#                 res[t3, j] += state[i, j] / 2 * (-1)**block0
#             elif t == t1:
#                 res[t, j] += state[i, j] / np.sqrt(2)
#                 res[t2, j] += state[i, j] / np.sqrt(2) * (-1)**block0
#             else:
#                 res[t, j] += state[i, j] / np.sqrt(2)
#                 res[t1, j] += state[i, j] / np.sqrt(2)
#         for i in range(N_sym, Ns):
#             t = s_list[i]
#             t1 = perm_operation(t, anci_perm)
#             t2 = perm_operation(t, perm0)
#             t3 = perm_operation(t1, perm0)
#             if t == t1 == t2 == t3:
#                 res[t, j] += 1j * state[i, j]
#             elif no_equal([t, t1, t2, t3]):
#                 res[t, j] += 1j/2 * state[i, j]
#                 res[t1, j] += -1j/2 * state[i, j]
#                 res[t2, j] += 1j/2 * (-1)**block0 * state[i, j]
#                 res[t3, j] += -1j/2 * (-1)**block0 * state[i, j]
#             else:
#                 res[t, j] += 1j/np.sqrt(2) * state[i, j]
#                 res[t1, j] += -1j/np.sqrt(2) * state[i, j]
#     return res

# ###########################################
# # Z22
# ##############################################
   

# config.CACHE_DIR = numba_cache_dir
# @njit
# def is_repr_super_Z22(s, perm0, block0, perm1, block1, anci_perm):
#     s_prime0 = perm_operation(s, perm0)
#     sym, asym = True, True
#     if (s_prime0 < s) or (block0==1 and s_prime0 == s):
#         return False, False, -1
    
#     s_prime1 = perm_operation(s, perm1)
#     sym, asym = True, True
#     if (s_prime1 < s) or (block1==1 and s_prime1 == s):
#         return False, False, -1

#     s_prime2 = perm_operation(s, anci_perm)
#     if s_prime2 < s:
#         return False, False, -1
#     elif s_prime2 == s:
#         asym = False
    
#     s_prime10 = perm_operation(s_prime0, perm1)
#     if (s_prime10 < s) or ((s_prime10 == s) and (block0 + block1) % 2 == 1):
#         return False, False, -1
    
#     s_prime20 = perm_operation(s_prime0, anci_perm)
#     if s_prime20 < s:
#         return False, False, -1
#     elif s_prime20 == s:
#         if block0 == 0:
#             asym = False
#         else:
#             sym = False

#     s_prime21 = perm_operation(s_prime1, anci_perm)
#     if s_prime21 < s:
#         return False, False, -1
#     elif s_prime21 == s:
#         if block1 == 0:
#             asym = False
#         else:
#             sym = False
    
#     s_prime120 = perm_operation(s_prime20, perm1)
#     if s_prime120 < s:
#         return False, False, -1
#     elif s_prime120 == s:
#         if (block0 + block1) % 2 == 0:
#             asym = False
#         else:
#             sym = False

#     return sym, asym, count_nonequal([s, s_prime0, s_prime1, s_prime2, s_prime10, s_prime20, s_prime21, s_prime120])


# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_basis_Z22(L, perm0, block0, perm1, block1, anci_perm):
#     s_list_sym = []
#     s_list_asym = []
#     R_list_sym = []
#     R_list_asym = []
#     for s in range(1<<L):
#         issym, isasym, r = is_repr_super_Z22(s, perm0, block0, perm1, block1, anci_perm)
#         if issym:
#             s_list_sym.append(s)
#             R_list_sym.append(r)
#         if isasym:
#             s_list_asym.append(s)
#             R_list_asym.append(r)
#     dim_sym = len(s_list_sym)
#     dim = dim_sym + len(s_list_asym)
#     s_list = np.zeros(dim, dtype=np.int64)
#     R_list = np.zeros(dim, dtype=np.int64)
#     s_list_sym = np.array(s_list_sym, dtype=np.int64)
#     s_list_asym = np.array(s_list_asym, dtype=np.int64)
#     R_list_sym = np.array(R_list_sym, dtype=np.int64)
#     R_list_asym = np.array(R_list_asym, dtype=np.int64)
#     indx_sym = np.argsort(s_list_sym)
#     indx_asym = np.argsort(s_list_asym)
#     s_list[:dim_sym] = s_list_sym[indx_sym]
#     s_list[dim_sym:] = s_list_asym[indx_asym]
#     R_list[:dim_sym] = R_list_sym[indx_sym]
#     R_list[dim_sym:] = R_list_asym[indx_asym]
#     return s_list, R_list, dim_sym, dim - dim_sym

# config.CACHE_DIR = numba_cache_dir
# @njit
# def representative_Z23(s, perm0, perm1, perm2):
#     t0 = perm_operation(s, perm0)
#     t1 = perm_operation(s, perm1)
#     t2 = perm_operation(s, perm2)
#     t01 = perm_operation(t1, perm0)
#     t02 = perm_operation(t2, perm0)
#     t12 = perm_operation(t2, perm1)
#     t012 = perm_operation(t12, perm0)
    
#     mint = min([s, t0, t1, t2, t01, t02, t12, t012]) 
#     if mint == s:
#         c0, c1, c2 = 0, 0, 0
#     elif mint == t0:
#         c0, c1, c2 = 1, 0, 0
#     elif mint == t1:
#         c0, c1, c2 = 0, 1, 0
#     elif mint == t2:
#         c0, c1, c2 = 0, 0, 1
#     elif mint == t01:
#         c0, c1, c2 = 1, 1, 0
#     elif mint == t02:
#         c0, c1, c2 = 1, 0, 1
#     elif mint == t12:
#         c0, c1, c2 = 0, 1, 1
#     else:
#         c0, c1, c2 = 1, 1, 1
#     return mint, c0, c1, c2


# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Ndiff_basis_Z22(L: int, Ndiff: np.ndarray, perm0, block0, perm1, block1, anci_perm, flipmask) -> tuple[int, np.ndarray]:
#     s_list_sym = []
#     s_list_asym = []
#     R_list_sym = []
#     R_list_asym = []
#     l = count_tot_down(flipmask)
#     for ndiff in Ndiff:
#         Nup = l + ndiff
#         if 2*L == Nup:
#             t = 0 ^ flipmask
#             issym, isasym, r = is_repr_super_Z22(t, perm0, block0, perm1, block1, anci_perm)
#             if issym:
#                 s_list_sym.append(t)
#                 R_list_sym.append(r)
#             if isasym:
#                 s_list_asym.append(t)
#                 R_list_asym.append(r)
#             continue
#         s = (1 << (2*L - Nup)) - 1
#         while s < (1 << (2*L)):
#             t = s ^ flipmask
#             issym, isasym, r = is_repr_super_Z22(t, perm0, block0, perm1, block1, anci_perm)
#             if issym:
#                 s_list_sym.append(t)
#                 R_list_sym.append(r)
#             if isasym:
#                 s_list_asym.append(t)
#                 R_list_asym.append(r)
#             s = next_combination(s)

#     dim_sym = len(s_list_sym)
#     dim = dim_sym + len(s_list_asym)
#     s_list = np.zeros(dim, dtype=np.int64)
#     R_list = np.zeros(dim, dtype=np.int64)
#     s_list_sym = np.array(s_list_sym, dtype=np.int64)
#     s_list_asym = np.array(s_list_asym, dtype=np.int64)
#     R_list_sym = np.array(R_list_sym, dtype=np.int64)
#     R_list_asym = np.array(R_list_asym, dtype=np.int64)
#     indx_sym = np.argsort(s_list_sym)
#     indx_asym = np.argsort(s_list_asym)
#     s_list[:dim_sym] = s_list_sym[indx_sym]
#     s_list[dim_sym:] = s_list_asym[indx_asym]
#     R_list[:dim_sym] = R_list_sym[indx_sym]
#     R_list[dim_sym:] = R_list_asym[indx_asym]
#     return s_list, R_list, dim_sym, dim - dim_sym

# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Nup2_basis_Z22(L: int, Nup2: np.ndarray, perm0, block0, perm1, block1, anci_perm) -> tuple[int, np.ndarray]:
#     s_list_sym = []
#     s_list_asym = []
#     R_list_sym = []
#     R_list_asym = []
#     for n1, n2 in Nup2:
#         if L == n1:
#             s1 = 0
#             if L == n2:
#                 s2 = 0
#                 s = (s1 << L) | s2
#                 issym, isasym, r = is_repr_super_Z22(s, perm0, block0, perm1, block1, anci_perm)
#                 if issym:
#                     s_list_sym.append(s)
#                     R_list_sym.append(r)
#                 if isasym:
#                     s_list_asym.append(s)
#                     R_list_asym.append(r)
#             else:
#                 s2 = (1 << (L - n2)) - 1
#                 while s2 < (1 << L):
#                     s = (s1 << L) | s2
#                     issym, isasym, r = is_repr_super_Z22(s, perm0, block0, perm1, block1, anci_perm)
#                     if issym:
#                         s_list_sym.append(s)
#                         R_list_sym.append(r)
#                     if isasym:
#                         s_list_asym.append(s)
#                         R_list_asym.append(r)
#                     s2 = next_combination(s2)
#             continue
#         s1 = (1 << (L - n1)) - 1
#         while s1 < (1 << L):
#             if L == n2:
#                 s2 = 0
#                 s = (s1 << L) | s2
#                 issym, isasym, r = is_repr_super_Z22(s, perm0, block0, perm1, block1, anci_perm)
#                 if issym:
#                     s_list_sym.append(s)
#                     R_list_sym.append(r)
#                 if isasym:
#                     s_list_asym.append(s)
#                     R_list_asym.append(r)
#             else:
#                 s2 = (1 << (L-n2)) - 1
#                 while s2 < (1 << L):
#                     s = (s1 << L) | s2
#                     issym, isasym, r = is_repr_super_Z22(s, perm0, block0, perm1, block1, anci_perm)
#                     if issym:
#                         s_list_sym.append(s)
#                         R_list_sym.append(r)
#                     if isasym:
#                         s_list_asym.append(s)
#                         R_list_asym.append(r)
#                     s2 = next_combination(s2)
#             s1 = next_combination(s1)

#     dim_sym = len(s_list_sym)
#     dim = dim_sym + len(s_list_asym)
#     s_list = np.zeros(dim, dtype=np.int64)
#     R_list = np.zeros(dim, dtype=np.int64)
#     s_list_sym = np.array(s_list_sym, dtype=np.int64)
#     s_list_asym = np.array(s_list_asym, dtype=np.int64)
#     R_list_sym = np.array(R_list_sym, dtype=np.int64)
#     R_list_asym = np.array(R_list_asym, dtype=np.int64)
#     indx_sym = np.argsort(s_list_sym)
#     indx_asym = np.argsort(s_list_asym)
#     s_list[:dim_sym] = s_list_sym[indx_sym]
#     s_list[dim_sym:] = s_list_asym[indx_asym]
#     R_list[:dim_sym] = R_list_sym[indx_sym]
#     R_list[dim_sym:] = R_list_asym[indx_asym]
#     return s_list, R_list, dim_sym, dim - dim_sym

# config.CACHE_DIR = numba_cache_dir
# @njit
# def proj_coef_Z22(s, perm0, perm1, perm2):
#     t0 = perm_operation(s, perm0)
#     t1 = perm_operation(s, perm1)
#     t2 = perm_operation(s, perm2)
#     t01 = perm_operation(t1, perm0)
#     t02 = perm_operation(t2, perm0)
#     t12 = perm_operation(t2, perm1)
#     t012 = perm_operation(t12, perm0)
    

#     ss = [s, t0, t1, t2, t01, t02, t12, t012]
#     whichop = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
#     # ss = [s, t2, t1, t2, t01, t02, t12, t012]
#     # whichop = [[0,0,0], [0,0,1], [0,1,0], [1,0,0], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
#     res = []
#     ls = []
#     for i in range(8):
#         isnew = True
#         for j in range(len(res)):
#             if ss[i] == res[j]:
#                 isnew = False
#                 break
#         if isnew:
#             res.append(ss[i])
#             ls.append(whichop[i])
#     return res, ls


# config.CACHE_DIR = numba_cache_dir
# @njit
# def projmat_Z22(L, s_list, N_sym, Ns, anci_perm, perm0, block0, perm1, block1):
#     dim = 8*Ns
#     row = np.zeros(dim, dtype=np.int64)
#     col = np.zeros(dim, dtype=np.int64)
#     ele = np.zeros(dim, dtype=np.complex128)
#     ct = 0
#     for i in range(N_sym):
#         t = s_list[i]
#         news, ops = proj_coef_Z22(t, perm0, perm1, anci_perm)
#         l = len(news)
#         for j in range(l):
#             row[ct], col[ct] = news[j], i
#             ele[ct] = l**(-0.5) * (-1)**(ops[j][0]*block0 + ops[j][1]*block1)
#             ct += 1
#     for i in range(N_sym, Ns):
#         t = s_list[i]
#         news, ops = proj_coef_Z22(t, perm0, perm1, anci_perm)
#         l = len(news)
#         for j in range(l):
#             row[ct], col[ct] = news[j], i
#             ele[ct] = 1j * l**(-0.5) * (-1)**(ops[j][0]*block0 + ops[j][1]*block1 + ops[j][2])
#             ct += 1
#     return row[:ct], col[:ct], ele[:ct]


# config.CACHE_DIR = numba_cache_dir
# @njit
# def project_Z22(state, L, s_list, N_sym, Ns, anci_perm, perm0, block0, perm1, block1):
#     M, N = state.shape
#     res = np.zeros((Ns, N), dtype=np.complex128)
#     for j in range(N):
#         for i in range(N_sym):
#             t = s_list[i]
#             news, ops = proj_coef_Z22(t, perm0, perm1, anci_perm)
#             l = len(news)
#             for k in range(l):
#                 res[i, j] += l**(-0.5) * (-1)**(ops[k][0]*block0 + ops[k][1]*block1) * state[news[k], j]
#         for i in range(N_sym, Ns):
#             t = s_list[i]
#             news, ops = proj_coef_Z22(t, perm0, perm1, anci_perm)
#             l = len(news)
#             for k in range(l):
#                 res[i, j] += -1j * l**(-0.5) * (-1)**(ops[k][0]*block0 + ops[k][1]*block1 + ops[k][2]) * state[news[k], j]
#     return res


# config.CACHE_DIR = numba_cache_dir
# @njit
# def recover_Z22(state, L, s_list, N_sym, Ns, anci_perm, perm0, block0, perm1, block1):
#     M, N = state.shape
#     res = np.zeros((1<<L, N), dtype=np.complex128)
#     for j in range(N):
#         for i in range(N_sym):
#             t = s_list[i]
#             news, ops = proj_coef_Z22(t, perm0, perm1, anci_perm)
#             l = len(news)
#             for k in range(l):
#                 res[news[k], j] += l**(-0.5) * (-1)**(ops[k][0]*block0 + ops[k][1]*block1) * state[i, j]
#         for i in range(N_sym, Ns):
#             t = s_list[i]
#             news, ops = proj_coef_Z22(t, perm0, perm1, anci_perm)
#             l = len(news)
#             for k in range(l):
#                 res[news[k], j] += 1j * l**(-0.5) * (-1)**(ops[k][0]*block0 + ops[k][1]*block1 + ops[k][2]) * state[i, j]
#     return res
