# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-11-29 16:50:16
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:24:34

import numpy as _np

from ....basicfun.utils_numba import njit, pnjit, prange, vectorize, numba_cache_dir, config

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


