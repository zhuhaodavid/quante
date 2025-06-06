# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 20:55:08
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-06 13:09:16

from .....linalg.usenumba.numba_settings import njit, types, config, numba_cache_dir
import numpy as _np

config.CACHE_DIR = numba_cache_dir

@njit("i8(i8)")
def next_combination(x):
    # 生成下一个具有相同数量1的整数
    u = x & -x
    v = u + x
    return v + (((v ^ x) // u) >> 2)

config.CACHE_DIR = numba_cache_dir

# @njit(types.Tuple((types.float64, types.int64))(types.string, types.ListType(types.int64), types.int64, types.int64))
# def operateon(opnm:str, posn:_np.ndarray, s:int, N:int) -> int:
#     """
#     Apply the operator opnm to the state s.
#     """
#     t = s
#     coef = 1.
#     for i in range(1,len(opnm)+1):
#         oi = opnm[-i]
#         pi = posn[-i]
#         mask = 1 << (N - pi - 1)
#         if oi == '-':
#             if t & mask != 0:
#                 return -1, -1
#             t = t ^ mask
#         elif oi == '+':
#             if t & mask == 0:
#                 return -1, -1
#             t = t ^ mask
#         elif oi == 'I':
#             t = t
#         elif oi == 'Z':
#             t = t
#             coef *= 1. if t & mask == 0 else -1.
#         else:
#             raise ValueError("Invalid operator")
#     return coef, t

# 'Tuple((f8,i8))(i8[:],i8[:],i8,i8)', 
@njit(inline='always')
def operateon(opnm, posn, a, L):
    """
    Apply the operator opnm to the state s.
    """
    opco = 1.
    t = a + 0
    for i in range(1,len(opnm)+1):
        oi = opnm[-i]
        pi = posn[-i]
        mask = 1 << (L - pi - 1)
        if oi == 0:
            if t & mask != 0:
                return opco, -1
            # cnt = count_tot_down(t >> (L-pi))
            cnt = count_tot_down((mask - 1) & t)
            if (pi - cnt) & 1:
                opco *= -1
            t = t ^ mask
        elif oi == 1:
            if t & mask == 0:
                return opco, -1
            # cnt = count_tot_down(t >> (L-pi))
            cnt = count_tot_down((mask - 1) & t)
            if (pi - cnt) & 1:
                opco *= -1
            t = t ^ mask
        elif oi == 3 and t & mask != 0:
            opco *= 0.
        else:
            t = t
    return opco, t


config.CACHE_DIR = numba_cache_dir

@njit("i8(i8,i8)")
def invertbits(s, N):
    """将整数 s 的最后 N 个比特位翻转"""
    # 创建掩码，掩码的最后 N 位是 1，其余位是 0
    mask = (1 << N) - 1
    
    # 将整数 s 的最后 N 位与掩码进行异或，翻转这些位
    result = s ^ mask
    
    return result

config.CACHE_DIR = numba_cache_dir

@njit("i8(i8,i8)")
def reflectbits(s, N):
    """Reflect (reverse) the first N bits of the integer s."""
    result = 0
    for i in range(N):
        # Extract the i-th bit from s and place it in the reflected position
        if (s >> i) & 1:
            result |= 1 << (N - 1 - i)
    return result
config.CACHE_DIR = numba_cache_dir

@njit("i8(i8,i8,i8)")
def cyclebits(t, m, N):
    """Perform a cyclic permutation of the first N bits of integer t by m positions."""
    """Perform a cyclic permutation of the first N bits of integer t by m positions."""
    # Mask to isolate the first N bits
    mask = (1 << N) - 1
    
    # Extract the first N bits of t
    t_bits = t & mask
    
    # Perform a cyclic shift of these N bits by m positions
    m = m % N  # Handle cases where m is larger than N
    t_shifted = ((t_bits >> m) | (t_bits << (N - m))) & mask

    # Combine the shifted bits with the rest of the original number
    result = (t & ~mask) | t_shifted

    return result

config.CACHE_DIR = numba_cache_dir
@njit('i8(i8,i8,i8)')
def flip(a:int, i:int, j:int) -> int:
    if ((a >> i) & 1) == ((a >> j) & 1):
        return -1
    return a ^ ((1 << i) | (1 << j))

config.CACHE_DIR = numba_cache_dir
@njit('i8(i8)')  # 如何更快？ a.bit_count()
def count_tot_down(n:int) -> int:
    """Calculate the sum of binary digits in the integer s using a high-efficiency approach.
    hacker_popcnt
    """
    n -= (n >> 1) & 0x55555555
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333)
    n = ((n >> 4) + n) & 0x0F0F0F0F
    n += n >> 8
    n += n >> 16
    return n & 0x3F
    # count = 0
    # while s:
    #     count += s & 1  # Add the least significant bit to count
    #     s >>= 1  # Shift bits to the right by 1
    # return count

config.CACHE_DIR = numba_cache_dir
@njit("i8(i8[:],i8)")
def findstate(s_list:_np.ndarray, sb:int) -> int:
    M = len(s_list)
    b_min = 0
    b_max = M-1
    for _ in range(int(_np.log2(M)) + 1):
        b = b_min + (b_max - b_min)//2
        if sb < s_list[b]:
            b_max = b
        elif sb > s_list[b]:
            b_min = b + 1
        else:
            return b
    return -1

