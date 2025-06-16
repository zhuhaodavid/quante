# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-08-15 09:52:35
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 19:07:59

# import time
import numpy as np
# from tqdm import tqdm
# from ...basicfun import println
from ....basicfun.utils_numba import njit, prange

@njit
def get_hammat(Tm:np.ndarray, Wr:np.ndarray, q:int, l:int) -> np.ndarray:
    hammat = np.zeros(shape=(l, l), dtype=np.float64)
    for i in prange(l):
        # 对角元
        hammat[i,i] += Tm[i % q]
        for j in range(l):
            hammat[i,j] += Wr[abs(i-j)]
    return hammat


@njit
def mat_k(Tm:np.ndarray, Wr:np.ndarray, k:float) -> np.ndarray:
    q = len(Tm)
    # 对 Tm 做傅里叶变换
    Ttildem = np.zeros(q, dtype=np.complex128)
    for i in range(q):
        angle = - 2j * np.pi * i/ q
        for n in range(q):
            Ttildem[i] += Tm[n] * np.exp(angle*n)/q
    
    res = np.zeros((q, q), dtype=np.complex128)
    for n in range(q):
        # 构造 Toeplitz 矩阵
        for nprime in range(q):
            res[n, nprime] += Ttildem[(n - nprime) % q]
        # 计算 Wmat 的对角元素
        tmp = 0
        exponent = 1j * (k + 2 * np.pi * n/ q)
        for r in range(len(Wr)):
            tmp += Wr[r] * np.exp(exponent * r)
        res[n, n] += tmp
    return res

@njit(parallel=True)
def engs_main(ks:np.ndarray, T:np.ndarray, W:np.ndarray) -> np.ndarray:
    n = len(ks)
    q = len(T)
    val = np.empty((n, q))
    for i in prange(len(ks)):
        mat = mat_k(T,W,k=ks[i])
        val[i,:] = np.linalg.eigvalsh(mat)
    return val
