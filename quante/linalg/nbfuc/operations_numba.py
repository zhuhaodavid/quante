# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-11-29 16:50:16
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-11 22:59:51

import numpy as _np
import numpy as np

from ...basicfun.utils_numba import njit, pnjit, prange, vectorize, numba_cache_dir, config

##########################################
# 格式转换，如 密矩阵和系数矩阵的转换
##########################################

config.CACHE_DIR = numba_cache_dir
@vectorize(['float64(float64)'], target='parallel', cache=True)
def parallel_exp_real(A):
    return _np.exp(A)

config.CACHE_DIR = numba_cache_dir
@vectorize(['complex128(complex128)'], target='parallel', cache=True)
def parallel_exp_complex(A):
    return _np.exp(A)

config.CACHE_DIR = numba_cache_dir
@vectorize(['complex128(float64, complex128)'], target='parallel', cache=True)
def parallel_expmul_rc(A, c):
    return _np.exp(A*c)

config.CACHE_DIR = numba_cache_dir
@vectorize(['complex128(complex128, complex128)'], target='parallel', cache=True)
def parallel_expmul_cc(A, c):
    return _np.exp(A*c)

config.CACHE_DIR = numba_cache_dir
@vectorize(['complex128(complex128, float64)'], target='parallel', cache=True)
def parallel_expmul_cr(A, c):
    return _np.exp(A*c)

config.CACHE_DIR = numba_cache_dir
@vectorize(['float64(float64, float64)'], target='parallel', cache=True)
def parallel_expmul_rr(A, c):
    return _np.exp(A*c)

@pnjit
def coo2array(xdata, ydata, zdata, dim) -> _np.ndarray:
    """稀疏矩阵 转 密矩阵"""
    mat = _np.zeros((dim, dim), dtype=zdata.dtype)
    s = xdata.size
    for i in prange(s):
        mat[xdata[i], ydata[i]] = zdata[i]
    return mat


@vectorize
def _uptrigindex(row_indx, col_indx, dim):
    """uptrig 将 (i,j) 指标变为生成的 list 的指标"""
    assert row_indx < dim - 1 and col_indx < dim - 1 and row_indx >= 0 and col_indx >= 0 and row_indx < col_indx
    return (dim - 1) * row_indx + col_indx - row_indx * (row_indx + 1) // 2 - 1

@vectorize
def _uptrigindex_inv(indices, dim):
    """uptrig 生成的 list 的指标变为上三角矩阵的指标"""
    assert indices <= dim * (dim - 1) // 2 and indices >= 0
    i = int(dim - 0.5 - _np.sqrt((dim - 0.5) ** 2 - 2 * indices))
    j = i + (indices - dim * i + (i + 1) * i // 2) + 1
    return i, j

@pnjit
def uptri2list(mat):
    dim = mat.shape[0]
    DiagElements = _np.diag(mat)
    UpRightElements = _np.empty(dim * (dim - 1) // 2, dtype=mat.dtype)
    for i in prange(dim):
        for j in range(i + 1, dim):
            UpRightElements[(dim - 1) * i + j - i * (i + 1) // 2 - 1] = mat[i, j]
    return DiagElements, UpRightElements


@pnjit
def list2uptri(lis):
    s = lis.size
    dim = int(0.5 + _np.sqrt(1 + 8 * s) / 2)
    assert dim**2 - dim == 2 * s
    mat = _np.zeros((dim, dim), dtype=lis.dtype)
    for k in prange(s):
        i = int(dim - 0.5 - _np.sqrt((dim - 0.5) ** 2 - 2 * k))
        j = i + (k - dim * i + (i + 1) * i // 2) + 1
        mat[i, j] = lis[k]
    return mat


import scipy

def dot_parallel(A, v, Yx=None):
    if scipy.sparse.issparse(A):
        n_row, n_col = A.shape
        assert n_col == v.shape[0]
        Ap = A.indptr
        Aj = A.indices
        Ax = A.data
        if Yx is None:
            dtype = _np.complex128 if _np.iscomplexobj(A) or _np.iscomplexobj(v) else _np.float64
            Yx = _np.empty(v.shape, dtype=dtype)
        if v.ndim == 1:
            _csr_matvec_parallel(n_row, Ap, Aj, Ax, v, Yx)
        else:
            n_vecs = v.shape[1]
            _csr_matvecs_parallel(n_row, n_vecs, Ap, Aj, Ax, v, Yx)
        return Yx
    if Yx is None:
        Yx = A.dot(v)
    else:
        _np.dot(A, v, out=Yx)
    return Yx


@njit(parallel=True, boundscheck=False)
def _csr_matvec_parallel(n_row, Ap, Aj, Ax, Xx, Yx):
    for i in prange(n_row):
        s = 0.
        for jj in range(Ap[i], Ap[i+1]):
            s += Ax[jj] * Xx[Aj[jj]]
        Yx[i] = s

@njit(parallel=True, boundscheck=False)
def _csr_matvecs_parallel(n_row, n_vecs, Ap, Aj, Ax, Xx, Yx):
    for i in prange(n_row):
        for ii in range(n_vecs):
            y = 0.
            for jj in range(Ap[i], Ap[i+1]):
                y += Ax[jj] * Xx[Aj[jj], ii]
            Yx[i, ii] = y


def addself(a, b, coef):
    if _np.iscomplexobj(a):
        addself_complex(a, b, coef)
    else:
        addself_float(a, b, coef)

@njit('void(complex128[:], complex128[:], float64)')
def addself_complex(a, b, coef):
    for i in prange(len(a)):
        bi = b[i]
        a[i] += coef * bi.real + (coef * bi.imag)*1j
            
@njit('void(float64[:], float64[:], float64)')
def addself_float(a, b, coef):
    for i in prange(len(a)):
        a[i] += coef * b[i]

def prodscale(a, coef):
    if _np.iscomplexobj(a):
        prodscale_complex(a, coef)
    else:
        prodscale_float(a, coef)

@njit('void(complex128[:], float64)')
def prodscale_complex(a, coef):
    for i in prange(len(a)):
        ai = a[i]
        a[i] = coef * ai.real + (coef * ai.imag)*1j

@njit('void(float64[:], float64)')
def prodscale_float(a, coef):
    for i in prange(len(a)):
        a[i] = coef * a[i]

@njit
def addtwo(a, b):
    for i in prange(len(a)):
        a[i] += b[i]

@njit
def _z_sum(h:float, L:int):
    mat = np.zeros((2 ** L,), dtype=float)
    for i in range(1<<L):

        zsum = 0.
        mask = 1 << (L-1)
        for j in range(L):
            if i & mask:
                zsum += - h[j]
            else:
                zsum += h[j]
            mask >>= 1
        mat[i] = zsum
    return mat

@njit
def _zz_sum(J:float, L:int, pbc:bool):
    mat = np.zeros((2 ** L,), dtype=float)
    for i in range(1<<L):

        zzsum = 0
        mask11 = 0b11 << (L-2)
        for j in range(L-1):
            if mask11 & i == 0 or mask11 & i == mask11:
                zzsum += 1
            mask11 >>= 1
        if (i & 1) == (i >> (L-1)) and pbc == 1:
            zzsum += 1
        
        mat[i] = J * (2 * zzsum - L) 
    return mat

  

config.CACHE_DIR = numba_cache_dir
@njit
def _quick_merge(res_pos, res_coef):
    total_len = len(res_pos)
    cur_coef = res_coef[0]
    cur_pos = 0
    prev_pos = res_pos[0]  # 引入局部变量存储上一个位置
    for i in range(1, total_len):
        tmp = res_pos[i]
        if (tmp == prev_pos).all():
            cur_coef += res_coef[i]
        else:
            res_pos[cur_pos] = prev_pos
            res_coef[cur_pos] = cur_coef
            cur_pos += 1
            cur_coef = res_coef[i]
            prev_pos = tmp
            
    res_pos[cur_pos] = res_pos[total_len-1]
    res_coef[cur_pos] = cur_coef
    mask = res_coef[:cur_pos + 1] != 0  # Remove zero coefficients
    return res_pos[:cur_pos+1][mask], res_coef[:cur_pos+1][mask]

