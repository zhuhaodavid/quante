# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-11-29 16:50:16
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:08:47

import numpy as _np

from ....basicfun.utils_numba import njit, pnjit, prange, vectorize, numba_cache_dir, config


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

