# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-28 14:30:52
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-11 22:40:41


import numpy as np
from .....basicfun.utils_numba import njit, config, numba_cache_dir
from ..bitsoperation import perm_operation, count_tot_down, next_combination

config.CACHE_DIR = numba_cache_dir
@njit
def make_state(flipmask, s1, s2, l1, l2):
    state = 0
    p1 = 0
    p2 = 0
    for i in range(l1+l2):
        if (flipmask >> i) & 1:
            if (s2 >> p2) & 1:
                state |= (1 << i)
            p2 += 1
        else:
            if (s1 >> p1) & 1:
                state |= (1 << i)
            p1 += 1
    return state

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
##############################################
config.CACHE_DIR = numba_cache_dir
@njit
def is_repr_Z2N(s, ps, bs):
    l = len(ps)
    oslis = np.zeros(1<<l, dtype=np.int64)
    bslis = np.zeros(1<<l, dtype=np.int64)
    oslis[0] = s
    bslis[0] = 0
    cur_pos = 1
    for i in range(l):
        p = ps[i]
        b = bs[i]
        for j in range(cur_pos):
            t = perm_operation(oslis[j], p)
            if t < s:
                return False, -1
            elif t == s:
                if (bslis[j] + b) % 2 == 1:
                    return False, -1
            oslis[cur_pos+j] = t
            bslis[cur_pos+j] = bslis[j] + b
        cur_pos *= 2
    return True, count_nonequal(oslis)

@njit
def construct_Z2N_basis(L, ps, bs):
    s_list = []
    R_list = []
    for s in range(1<<L):
        is_repr_s, r = is_repr_Z2N(s, ps, bs)
        if is_repr_s:
            s_list.append(s)
            R_list.append(r)
    return len(s_list), np.array(s_list), np.array(R_list)

@njit
def construct_Ndiff_Z2N_basis(L, flipmask, Ndiff, ps, bs):
    s_list = []
    R_list = []
    l = count_tot_down(flipmask)

    for ndiff in Ndiff:
        Nup = l + ndiff

        if L == Nup:
            real_s = 0 ^ flipmask

            is_repr_real_s, r = is_repr_Z2N(real_s, ps, bs)
            if is_repr_real_s:
                s_list.append(real_s)
                R_list.append(r)
            continue
        
        s = (1 << (L - Nup)) - 1

        # 在 N 个位置内生成具有 Nup 个1的所有可能组合
        while s < (1 << L):
            real_s = s ^ flipmask
            s = next_combination(s)

            is_repr_real_s, r = is_repr_Z2N(real_s, ps, bs)
            if is_repr_real_s:
                s_list.append(real_s)
                R_list.append(r)

    s_list = np.array(s_list) 
    indx = np.argsort(s_list)
    s_list = s_list[indx]
    R_list = np.array(R_list)[indx]
    return len(s_list), s_list, R_list


config.CACHE_DIR = numba_cache_dir
@njit
def construct_Nup2_Z2N_basis(L: int, flipmask: int, Nup2: np.ndarray, ps, bs) -> tuple[int, np.ndarray]:
    s_list = []
    R_list = []
    l2 = count_tot_down(flipmask)
    l1 = L - l2
    for n1, n2 in Nup2:
        if l1 == n1:
            s1 = 0
            if l2 == n2:
                s2 = 0
                s = make_state(flipmask, s1, s2, l1, l2)
                is_repr_s, r = is_repr_Z2N(s, ps, bs)
                if is_repr_s:
                    s_list.append(s)
                    R_list.append(r)
            else:
                s2 = (1 << (l2 - n2)) - 1
                while s2 < (1 << l2):
                    s = make_state(flipmask, s1, s2, l1, l2)
                    is_repr_s, r = is_repr_Z2N(s, ps, bs)
                    if is_repr_s:
                        s_list.append(s)
                        R_list.append(r)
                    s2 = next_combination(s2)
            continue
        s1 = (1 << (l1 - n1)) - 1
        while s1 < (1 << l1):
            if l2 == n2:
                s2 = 0
                s = make_state(flipmask, s1, s2, l1, l2)
                is_repr_s, r = is_repr_Z2N(s, ps, bs)
                if is_repr_s:
                    s_list.append(s)
                    R_list.append(r)
            else:
                s2 = (1 << (l2 - n2)) - 1
                while s2 < (1 << l2):
                    s = make_state(flipmask, s1, s2, l1, l2)
                    is_repr_s, r = is_repr_Z2N(s, ps, bs)
                    if is_repr_s:
                        s_list.append(s)
                        R_list.append(r)
                    s2 = next_combination(s2)
            s1 = next_combination(s1)

    s_list = np.array(s_list) 
    indx = np.argsort(s_list)
    s_list = s_list[indx]
    R_list = np.array(R_list)[indx]
    return len(s_list), s_list, R_list

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
@njit(inline='always')
def _sign(ls, bs):
    l = len(bs)
    sign = 1
    for i in range(l):
        ii = (ls >> i) & 1
        bi = bs[i]
        sign *= (-1)**(bi*ii)
    return sign

# ###########################################
# # ZNN
# ##############################################
# # config.CACHE_DIR = numba_cache_dir
# # @njit
# def is_repr_ZNN(s, ps, bs, ns):
#     l = len(ps)
#     dim = np.prod(ns)
#     oslis = np.zeros(dim, dtype=np.int64)
#     bslis = np.ones(dim, dtype=np.complex128)
#     oslis[0] = s
#     bslis[0] = 1
#     cur_pos = 1
#     for i in range(l):
#         p = ps[i]
#         b = bs[i]
#         n = ns[i]
#         ct = 0
#         for j in range(cur_pos):
#             t = oslis[j]
#             bj = bslis[j]
#             for k in range(1,n):
#                 t = perm_operation(t, p)

                


#         cur_pos *= n
#     return True, count_nonequal(oslis)

# # config.CACHE_DIR = numba_cache_dir
# # @njit
# def construct_ZNN_basis(L, ps, bs, ns):
#     s_list = []
#     R_list = []
#     for s in range(1<<L):
#         is_repr_s, r = is_repr_ZNN(s, ps, bs, ns)
#         if is_repr_s:
#             s_list.append(s)
#             R_list.append(r)
#     return len(s_list), np.array(s_list), np.array(R_list)

# import numpy as np
# from .....basicfun.utils_numba import njit, config, numba_cache_dir
# from ..bitsoperation import perm_operation, count_tot_down, next_combination, no_equal

# ###########################################
# # Z21
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Z21_basis(L, perm, block):
#     s_list = []
#     for s in range(1<<L):
#         s_prime = perm_operation(s, perm)
#         if s_prime < s:
#             continue
#         if block==1 and s_prime == s:
#             continue
#         s_list.append(s)
#     return len(s_list), np.array(s_list)


# config.CACHE_DIR = numba_cache_dir
# @njit
# def representative_Z21(s, perm):
#     t = perm_operation(s, perm)
#     if t < s:
#         return t, 1
#     else:
#         return s, 0


# ############################################
# # Ndiff
# ##############################################
# # config.CACHE_DIR = numba_cache_dir
# # @njit
# # def construct_Ndiff_basis(L: int, flipmask: int, Ndiff: np.ndarray) -> tuple[int, np.ndarray]:
# #     s_list = []
# #     tmp = L-2*count_tot_down(flipmask)
# #     for s in range(1<<L):
# #         cd_total = count_tot_down(s)
# #         cd_flip_part = count_tot_down(s & flipmask)
# #         # 总共 0 的个数 减去 翻转部分 0 的个数 
# #         # (L-cd_total)-(flip_number - cd_flip_part), (flip_number - cd_flip_part)
# #         ndiff = tmp+2*cd_flip_part-cd_total
# #         if ndiff in Ndiff:
# #             s_list.append(s)
# #     return len(s_list), s_list


# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Ndiff_basis(L: int, flipmask: int, Ndiff: np.ndarray) -> tuple[int, np.ndarray]:
#     s_list = []
#     l = count_tot_down(flipmask)

#     for ndiff in Ndiff:
#         Nup = l + ndiff

#         if L == Nup:
#             s_list.append(0 ^ flipmask)
#             continue
        
#         s = (1 << (L - Nup)) - 1

#         # 在 N 个位置内生成具有 Nup 个1的所有可能组合
#         while s < (1 << L):
#             s_list.append(s ^ flipmask)
#             s = next_combination(s)
        
#     s_list = np.array(np.sort(s_list))
#     return len(s_list), s_list


# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Nup2_basis(L: int, flipmask: int, Nup2: np.ndarray) -> tuple[int, np.ndarray]:
#     s_list = []
#     l2 = count_tot_down(flipmask)
#     l1 = L - l2
#     for n1, n2 in Nup2:
#         if l1 == n1:
#             s1 = 0
#             if l2 == n2:
#                 s2 = 0
#                 s = make_state(flipmask, s1, s2, l1, l2)
#                 s_list.append(s)
#             else:
#                 s2 = (1 << (l2 - n2)) - 1
#                 while s2 < (1 << l2):
#                     s = make_state(flipmask, s1, s2, l1, l2)
#                     s_list.append(s)
#                     s2 = next_combination(s2)
#             continue
#         s1 = (1 << (l1 - n1)) - 1
#         while s1 < (1 << l1):
#             if l2 == n2:
#                 s2 = 0
#                 s = make_state(flipmask, s1, s2, l1, l2)
#                 s_list.append(s)
#             else:
#                 s2 = (1 << (l2 - n2)) - 1
#                 while s2 < (1 << l2):
#                     s = make_state(flipmask, s1, s2, l1, l2)
#                     s_list.append(s)
#                     s2 = next_combination(s2)
#             s1 = next_combination(s1)
#     s_list = np.array(np.sort(s_list))
#     return len(s_list), s_list



# ############################################
# # Ndiff Z21
# #############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def is_repr_Z21(s, perm, block):
#     s_prime = perm_operation(s, perm)
#     if (s_prime < s) or (block==1 and s_prime == s):
#         return False
#     return True



# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Ndiff_Z21_basis(L: int, flipmask: int, Ndiff: np.ndarray, perm, block) -> tuple[int, np.ndarray]:
#     s_list = []
#     l = count_tot_down(flipmask)

#     for ndiff in Ndiff:
#         Nup = l + ndiff

#         if L == Nup:
#             real_s = 0 ^ flipmask
#             if is_repr_Z21(real_s, perm, block):
#                 s_list.append(real_s)
       
#         s = (1 << (L - Nup)) - 1
#         while s < (1 << L):
#             real_s = s ^ flipmask
#             s = next_combination(s)
            
#             if is_repr_Z21(real_s, perm, block):
#                 s_list.append(real_s)
        
#     s_list = np.array(np.sort(s_list))
#     return len(s_list), s_list



# ############################################
# # Nup2 Z21
# #############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Nup2_Z21_basis(L: int, flipmask: int, Nup2: np.ndarray, perm, block) -> tuple[int, np.ndarray]:
#     s_list = []
#     l2 = count_tot_down(flipmask)
#     l1 = L - l2
#     for n1, n2 in Nup2:
#         if l1 == n1:
#             s1 = 0
#             if l2 == n2:
#                 s2 = 0
#                 s = make_state(flipmask, s1, s2, l1, l2)
#                 if is_repr_Z21(s, perm, block):
#                     s_list.append(s)
#             else:
#                 s2 = (1 << (l2 - n2)) - 1
#                 while s2 < (1 << l2):
#                     s = make_state(flipmask, s1, s2, l1, l2)
#                     if is_repr_Z21(s, perm, block):
#                         s_list.append(s)
#                     s2 = next_combination(s2)
#             continue
#         s1 = (1 << (l1 - n1)) - 1
#         while s1 < (1 << l1):
#             if l2 == n2:
#                 s2 = 0
#                 s = make_state(flipmask, s1, s2, l1, l2)
#                 if is_repr_Z21(s, perm, block):
#                     s_list.append(s)
#             else:
#                 s2 = (1 << (l2 - n2)) - 1
#                 while s2 < (1 << l2):
#                     s = make_state(flipmask, s1, s2, l1, l2)
#                     if is_repr_Z21(s, perm, block):
#                         s_list.append(s)
#                     s2 = next_combination(s2)
#             s1 = next_combination(s1)
#     s_list = np.array(np.sort(s_list))
#     return len(s_list), s_list


# ###########################################
# # Z22
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def is_repr_Z22(s, perm0, perm1, block0, block1):
#     s_prime0 = perm_operation(s, perm0)
#     if (s_prime0 < s) or (block0==1 and s_prime0 == s):
#         return False, -1
#     s_prime1 = perm_operation(s, perm1)
#     if (s_prime1 < s) or (block1==1 and s_prime1 == s):
#         return False, -1
#     s_prime01 = perm_operation(s_prime1, perm0)
#     if (s_prime01 < s) or (block0+block1==1 and s_prime01 == s):
#         return False, -1

#     if s == s_prime0 == s_prime1 == s_prime01:
#         return True, 1  # 4 states combined into 1
#     elif no_equal([s, s_prime0, s_prime1, s_prime01]) :
#         return True, 4  # 4 states combined into 4
#     else:
#         return True, 2  # 4 states combined into 2


# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Z22_basis(L, perm0, block0, perm1, block1):
#     s_list = []
#     R_list = []
#     for s in range(1<<L):
#         is_repr_s, r = is_repr_Z22(s, perm0, perm1, block0, block1)
#         if is_repr_s:
#             s_list.append(s)
#             R_list.append(r)

#     return len(s_list), np.array(s_list), np.array(R_list)


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


# ###########################################
# # Ndiff Z22
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Ndiff_Z22_basis(L: int, flipmask: int, Ndiff: np.ndarray, perm0, block0, perm1, block1) -> tuple[int, np.ndarray]:
#     s_list = []
#     R_list = []
#     l = count_tot_down(flipmask)

#     for ndiff in Ndiff:
#         Nup = l + ndiff

#         if L == Nup:
#             real_s = 0 ^ flipmask

#             is_repr_real_s, r = is_repr_Z22(real_s, perm0, perm1, block0, block1)
#             if is_repr_real_s:
#                 s_list.append(real_s)
#                 R_list.append(r)
#             continue
        
#         s = (1 << (L - Nup)) - 1

#         # 在 N 个位置内生成具有 Nup 个1的所有可能组合
#         while s < (1 << L):
#             real_s = s ^ flipmask
#             s = next_combination(s)

#             is_repr_real_s, r = is_repr_Z22(real_s, perm0, perm1, block0, block1)
#             if is_repr_real_s:
#                 s_list.append(real_s)
#                 R_list.append(r)

#     s_list = np.array(s_list) 
#     indx = np.argsort(s_list)
#     s_list = s_list[indx]
#     R_list = np.array(R_list)[indx]
#     return len(s_list), s_list, R_list

# ############################################
# # Nup2 Z22
# #############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Nup2_Z22_basis(L: int, flipmask: int, Nup2: np.ndarray, perm0, block0, perm1, block1) -> tuple[int, np.ndarray]:
#     s_list = []
#     R_list = []
#     l2 = count_tot_down(flipmask)
#     l1 = L - l2
#     for n1, n2 in Nup2:
#         if l1 == n1:
#             s1 = 0
#             if l2 == n2:
#                 s2 = 0
#                 s = make_state(flipmask, s1, s2, l1, l2)
#                 is_repr_s, r = is_repr_Z22(s, perm0, perm1, block0, block1)
#                 if is_repr_s:
#                     s_list.append(s)
#                     R_list.append(r)
#             else:
#                 s2 = (1 << (l2 - n2)) - 1
#                 while s2 < (1 << l2):
#                     s = make_state(flipmask, s1, s2, l1, l2)
#                     is_repr_s, r = is_repr_Z22(s, perm0, perm1, block0, block1)
#                     if is_repr_s:
#                         s_list.append(s)
#                         R_list.append(r)
#                     s2 = next_combination(s2)
#             continue
#         s1 = (1 << (l1 - n1)) - 1
#         while s1 < (1 << l1):
#             if l2 == n2:
#                 s2 = 0
#                 s = make_state(flipmask, s1, s2, l1, l2)
#                 is_repr_s, r = is_repr_Z22(s, perm0, perm1, block0, block1)
#                 if is_repr_s:
#                     s_list.append(s)
#                     R_list.append(r)
#             else:
#                 s2 = (1 << (l2 - n2)) - 1
#                 while s2 < (1 << l2):
#                     s = make_state(flipmask, s1, s2, l1, l2)
#                     is_repr_s, r = is_repr_Z22(s, perm0, perm1, block0, block1)
#                     if is_repr_s:
#                         s_list.append(s)
#                         R_list.append(r)
#                     s2 = next_combination(s2)
#             s1 = next_combination(s1)

#     s_list = np.array(s_list) 
#     indx = np.argsort(s_list)
#     s_list = s_list[indx]
#     R_list = np.array(R_list)[indx]
#     return len(s_list), s_list, R_list



# ###########################################
# # Z23
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @njit(inline='always')
# def is_repr_Z23(s, perm0, perm1, perm2, block0, block1, block2):
#     s_prime0 = perm_operation(s, perm0)
#     if (s_prime0 < s) or (block0==1 and s_prime0 == s):
#         return False, -1
#     s_prime1 = perm_operation(s, perm1)
#     if (s_prime1 < s) or (block1==1 and s_prime1 == s):
#         return False, -1
#     s_prime2 = perm_operation(s, perm2)
#     if (s_prime2 < s) or (block2==1 and s_prime2 == s):
#         return False, -1
#     s_prime10 = perm_operation(s_prime0, perm1)
#     if (s_prime10 < s) or (block1+block0==1 and s_prime10 == s):
#         return False, -1
#     s_prime20 = perm_operation(s_prime0, perm2)
#     if (s_prime20 < s) or (block2+block0==1 and s_prime20 == s):
#         return False, -1
#     s_prime21 = perm_operation(s_prime1, perm2)
#     if (s_prime21 < s) or (block2+block1==1 and s_prime21 == s):
#         return False, -1
#     s_prime210 = perm_operation(s_prime10, perm2)
#     if (s_prime210 < s) or ((block2+block1+block0)%2==1 and s_prime210 == s):
#         return False, -1
    
#     # now s is representative
#     ss = [s, s_prime0, s_prime1, s_prime2, s_prime10, s_prime20, s_prime21, s_prime210]
#     s_prime_list = []
#     for si in ss:
#         for sj in s_prime_list:
#             if si == sj:
#                 break
#         else:
#             s_prime_list.append(si)
#     r = len(s_prime_list)
#     return True, r


# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Z23_basis(L, perm0, block0, perm1, block1, perm2, block2):
#     s_list = []
#     R_list = []
#     for s in range(1<<L):
#         is_repr_s, r = is_repr_Z23(s, perm0, perm1, perm2, block0, block1, block2)
#         if is_repr_s:
#             s_list.append(s)
#             R_list.append(r)
#     return len(s_list), np.array(s_list), np.array(R_list)


# config.CACHE_DIR = numba_cache_dir
# @njit
# def representative_Z23(s, perm0, perm1, perm2):
#     s_prime0 = perm_operation(s, perm0)
#     s_prime1 = perm_operation(s, perm1)
#     s_prime2 = perm_operation(s, perm2)
#     s_prime10 = perm_operation(s_prime0, perm1)
#     s_prime20 = perm_operation(s_prime0, perm2)
#     s_prime21 = perm_operation(s_prime1, perm2)
#     s_prime210 = perm_operation(s_prime10, perm2)

#     mint = min([s, s_prime0, s_prime1, s_prime2, s_prime10, s_prime20, s_prime21, s_prime210])
#     if mint == s:
#         c0, c1, c2 = 0, 0, 0
#     elif mint == s_prime0:
#         c0, c1, c2 = 1, 0, 0
#     elif mint == s_prime1:
#         c0, c1, c2 = 0, 1, 0
#     elif mint == s_prime2:
#         c0, c1, c2 = 0, 0, 1
#     elif mint == s_prime10:
#         c0, c1, c2 = 1, 1, 0
#     elif mint == s_prime20:
#         c0, c1, c2 = 1, 0, 1
#     elif mint == s_prime21:
#         c0, c1, c2 = 0, 1, 1
#     else:  # mint == s_prime210
#         c0, c1, c2 = 1, 1, 1
#     return mint, c0, c1, c2

# ###########################################
# # Nup Z23
# ##############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Ndiff_Z23_basis(L: int, flipmask: int, Ndiff: np.ndarray, perm0, block0, perm1, block1, perm2, block2) -> tuple[int, np.ndarray]:
#     s_list = []
#     R_list = []
#     l = count_tot_down(flipmask)

#     for ndiff in Ndiff:
#         Nup = l + ndiff

#         if L == Nup:
#             real_s = 0 ^ flipmask

#             is_repr_real_s, r = is_repr_Z23(real_s, perm0, perm1, perm2, block0, block1, block2)
#             if is_repr_real_s:
#                 s_list.append(real_s)
#                 R_list.append(r)
#             continue
        
#         s = (1 << (L - Nup)) - 1

#         # 在 N 个位置内生成具有 Nup 个1的所有可能组合
#         while s < (1 << L):
#             real_s = s ^ flipmask
#             s = next_combination(s)

#             is_repr_real_s, r = is_repr_Z23(real_s, perm0, perm1, perm2, block0, block1, block2)
#             if is_repr_real_s:
#                 s_list.append(real_s)
#                 R_list.append(r)

#     s_list = np.array(s_list) 
#     indx = np.argsort(s_list)
#     s_list = s_list[indx]
#     R_list = np.array(R_list)[indx]
#     return len(s_list), s_list, R_list

# ############################################
# # Nup2 Z23
# #############################################
# config.CACHE_DIR = numba_cache_dir
# @njit
# def construct_Nup2_Z23_basis(L: int, flipmask: int, Nup2: np.ndarray, perm0, block0, perm1, block1, perm2, block2) -> tuple[int, np.ndarray]:
#     s_list = []
#     R_list = []
#     l2 = count_tot_down(flipmask)
#     l1 = L - l2
#     for n1, n2 in Nup2:
#         if l1 == n1:
#             s1 = 0
#             if l2 == n2:
#                 s2 = 0
#                 s = make_state(flipmask, s1, s2, l1, l2)
#                 is_repr_s, r = is_repr_Z23(s, perm0, perm1, perm2, block0, block1, block2)
#                 if is_repr_s:
#                     s_list.append(s)
#                     R_list.append(r)
#             else:
#                 s2 = (1 << (l2 - n2)) - 1
#                 while s2 < (1 << l2):
#                     s = make_state(flipmask, s1, s2, l1, l2)
#                     is_repr_s, r = is_repr_Z23(s, perm0, perm1, perm2, block0, block1, block2)
#                     if is_repr_s:
#                         s_list.append(s)
#                         R_list.append(r)
#                     s2 = next_combination(s2)
#             continue
#         s1 = (1 << (l1 - n1)) - 1
#         while s1 < (1 << l1):
#             if l2 == n2:
#                 s2 = 0
#                 s = make_state(flipmask, s1, s2, l1, l2)
#                 is_repr_s, r = is_repr_Z23(s, perm0, perm1, perm2, block0, block1, block2)
#                 if is_repr_s:
#                     s_list.append(s)
#                     R_list.append(r)
#             else:
#                 s2 = (1 << (l2 - n2)) - 1
#                 while s2 < (1 << l2):
#                     s = make_state(flipmask, s1, s2, l1, l2)
#                     is_repr_s, r = is_repr_Z23(s, perm0, perm1, perm2, block0, block1, block2)
#                     if is_repr_s:
#                         s_list.append(s)
#                         R_list.append(r)
#                     s2 = next_combination(s2)
#             s1 = next_combination(s1)

#     s_list = np.array(s_list) 
#     indx = np.argsort(s_list)
#     s_list = s_list[indx]
#     R_list = np.array(R_list)[indx]
#     return len(s_list), s_list, R_list


