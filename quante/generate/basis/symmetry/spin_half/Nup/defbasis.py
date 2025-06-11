# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 20:22:38
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-24 01:10:21

from ......basicfun.utils_numba import njit, config, numba_cache_dir
from ..bitsoperation import next_combination
import numpy as np

config.CACHE_DIR = numba_cache_dir
@njit('i8(i8,i8)')
def comb(n:int, k:int) -> int:
    """组合数，等价于 math.comb，但只能计算 int64 以内"""
    if k > n:
        return 0
    if (k == 0) or (k == n):
        return 1
    binom_ = 1
    keff = n - k if k > n // 2 else k
    for i in range(1, keff + 1):
        binom_ *= n - keff + i
        binom_ //= i
    return binom_


# config.CACHE_DIR = numba_cache_dir
# @njit("Tuple((i8,i8[:]))(i8,i8)")
# def construct_Nup_basis_naive(L:int, Nup:int) -> tuple[int, np.ndarray]:
#     """
#     Generate a list of integers representing the states of the magnetization block of size N with total magnetization tot_up.
#     """
#     M = comb(L, Nup)
#     s_list = np.empty(M, dtype=np.int64)
#     a = 0
#     for s in range(1<<L):
#         if count_tot_down(s) == Nup:
#             s_list[a] = s
#             a = a + 1
#     return M, s_list


config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8[:]))(i8,i8)")
def construct_Nup_basis(L: int, Nup: int) -> tuple[int, np.ndarray]:
    if L == Nup:
        return 1, np.array([0], dtype=np.int64)
    
    s_list = []
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << (L - Nup)) - 1

    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        s_list.append(s)
        s = next_combination(s)
    
    s_list = np.array(s_list, dtype=np.int64)
    
    return len(s_list), s_list

config.CACHE_DIR = numba_cache_dir
@njit
def convert_project_to_full_space(proj:np.ndarray, L:int, s_list:np.ndarray) -> np.ndarray:
    M, N = proj.shape
    res = np.zeros((1<<L, N), dtype=proj.dtype)
    for i in range(M):
        for j in range(N):
            mele = proj[i, j]
            # if np.isclose(mele, 0):
            #     continue
            res[s_list[i], j] = mele
    return res

