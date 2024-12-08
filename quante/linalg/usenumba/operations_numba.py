# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-11-29 16:50:16
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-03 01:14:26

import numpy as _np
import numpy as np

from .numba_settings import njit, pnjit, prange, vectorize

##########################################
# 格式转换，如 密矩阵和系数矩阵的转换
##########################################

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

@pnjit("float64[:](float64[:,:], float64[:,:])")
def observe_states_float(vecs, O):
    n = vecs.shape[1]
    res = _np.empty(n, dtype=_np.float64)
    O_contiguous = _np.ascontiguousarray(O)
    for i in prange(n):
        v = vecs[:, i]
        v_conj = _np.ascontiguousarray(v.conj())
        O_v = _np.dot(O_contiguous, _np.ascontiguousarray(v))
        tmp = _np.dot(v_conj.T, O_v)
        res[i] = _np.real(tmp)
    return res

@pnjit("complex128[:](complex128[:,:], complex128[:,:])")
def observe_states_complex(vecs, O):
    n = vecs.shape[1]
    res = _np.empty(n, dtype=_np.complex128)
    O_contiguous = _np.ascontiguousarray(O)
    for i in prange(n):
        v = vecs[:, i]
        v_conj = _np.ascontiguousarray(v.conj())
        O_v = _np.dot(O_contiguous, _np.ascontiguousarray(v))
        tmp = _np.dot(v_conj.T, O_v)
        res[i] = tmp
    return res

@njit
def _factorial_prod(N, arr):
    arr[:int(N)] += 1

@njit
def _factorial_div(N, arr):
    arr[:int(N)] -= 1

@njit
def _to_long(arr):
    prod = 1
    for i, v in enumerate(arr):
        prod *= (i+1.0)**int(v)
    return prod


@njit
def clebsch(j1, j2, j3, m1, m2, m3):
    if m3 != m1 + m2:
        return 0
    
    # vmin = int(_np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    # vmax = int(_np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))
    vmin = int(_np.array([-j1 + j2 + m3, -j1 + m1, 0]).max())
    vmax = int(_np.array([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]).min())

    c_factor = _np.zeros((int(j1 + j2 + j3 + 1)), _np.int32)
    _factorial_prod(j3 + j1 - j2, c_factor)
    _factorial_prod(j3 - j1 + j2, c_factor)
    _factorial_prod(j1 + j2 - j3, c_factor)
    _factorial_prod(j3 + m3, c_factor)
    _factorial_prod(j3 - m3, c_factor)
    _factorial_div(j1 + j2 + j3 + 1, c_factor)
    _factorial_div(j1 - m1, c_factor)
    _factorial_div(j1 + m1, c_factor)
    _factorial_div(j2 - m2, c_factor)
    _factorial_div(j2 + m2, c_factor)
    C = _np.sqrt((2.0 * j3 + 1.0)*_to_long(c_factor))

    xdim = int(vmax + 1 - vmin)
    ydim = int(j1 + j2 + j3)
    s_factors = _np.zeros((xdim, ydim), _np.int32)
    # s_factors = _np.zeros(((vmax + 1 - vmin), (int(j1 + j2 + j3))), _np.int32)
    
    sign = (-1) ** (vmin + j2 + m2)
    for i,v in enumerate(range(vmin, vmax + 1)):
        factor = s_factors[i,:]
        _factorial_prod(j2 + j3 + m1 - v, factor)
        _factorial_prod(j1 - m1 + v, factor)
        _factorial_div(j3 - j1 + j2 - v, factor)
        _factorial_div(j3 + m3 - v, factor)
        _factorial_div(v + j1 - j2 - m3, factor)
        _factorial_div(v, factor)
    
    common_denominator = _np.zeros(ydim, _np.int32)
    for i in range(ydim):
        common_denominator[i] = - s_factors[:, i].min()    
    # common_denominator = -_np.min(s_factors, axis=0)

    numerators = s_factors + common_denominator
    S = sum([(-1)**i * _to_long(vec) for i,vec in enumerate(numerators)]) * \
        sign / _to_long(common_denominator)
    return C * S


@vectorize(['complex128(complex128)'], target='parallel')
def numba_exp(x):# -> Any:
    return _np.exp(x)

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
