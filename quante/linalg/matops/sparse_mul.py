# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:17:43
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-23 21:34:25

import numpy as _np
from scipy.sparse import csr_array, csr_matrix, dia_array, dia_matrix, issparse

def dot_parallel(A, v, Yx=None, a=None, overwrite=False):
    if Yx is None:
        if not isinstance(A, _np.ndarray) and not issparse(A):
            return A.dot(v)
        dtype = _np.complex128 if _np.iscomplexobj(A) or _np.iscomplexobj(v) else _np.float64
        Yx = _np.zeros(v.shape, dtype=dtype) 
    if isinstance(A, (csr_array, csr_matrix)):
        csr_matvec(A, v, Yx, a=a, overwrite=overwrite)
    elif isinstance(A, (dia_array, dia_matrix)) and all(A.offsets == [0]):
        dia_matvec(A.diagonal(), v, Yx, a=a, overwrite=overwrite)
    else:
        _np.dot(A, v, out=Yx)
    return Yx

def csr_matvec(A, v, Yx, a=None, overwrite=False):
    n_row, n_col = A.shape
    assert n_col == v.shape[0]
    Ap = A.indptr
    Aj = A.indices
    Ax = A.data
    if overwrite and a is None:
        if v.ndim == 1:
            from .nbfuc.sparse_mul_nb import _csr_matvec_parallel1
            _csr_matvec_parallel1(n_row, Ap, Aj, Ax, v, Yx)
        else:
            n_vecs = v.shape[1]
            from .nbfuc.sparse_mul_nb import _csr_matvecs_parallel1
            _csr_matvecs_parallel1(n_row, n_vecs, Ap, Aj, Ax, v, Yx)
    elif overwrite and a is not None:
        if v.ndim == 1:
            from .nbfuc.sparse_mul_nb import _csr_matvec_parallel2
            _csr_matvec_parallel2(n_row, Ap, Aj, Ax, v, Yx, a)
        else:
            n_vecs = v.shape[1]
            from .nbfuc.sparse_mul_nb import _csr_matvecs_parallel2
            _csr_matvecs_parallel2(n_row, n_vecs, Ap, Aj, Ax, v, Yx, a)
    elif not overwrite and a is None:
        if v.ndim == 1:
            from .nbfuc.sparse_mul_nb import _csr_matvec_parallel3
            _csr_matvec_parallel3(n_row, Ap, Aj, Ax, v, Yx)
        else:
            n_vecs = v.shape[1]
            from .nbfuc.sparse_mul_nb import _csr_matvecs_parallel3
            _csr_matvecs_parallel3(n_row, n_vecs, Ap, Aj, Ax, v, Yx)
    else:
        if v.ndim == 1:
            from .nbfuc.sparse_mul_nb import _csr_matvec_parallel4
            _csr_matvec_parallel4(n_row, Ap, Aj, Ax, v, Yx, a)
        else:
            n_vecs = v.shape[1]
            from .nbfuc.sparse_mul_nb import _csr_matvecs_parallel4
            _csr_matvecs_parallel4(n_row, n_vecs, Ap, Aj, Ax, v, Yx, a)
    return Yx


def dia_matvec(diag, v, Yx=None, a=None, overwrite=False):
    n = len(diag)
    if overwrite and a is None:
        if v.ndim == 1:
            from .nbfuc.sparse_mul_nb import _dia_matvec_parallel1
            _dia_matvec_parallel1(diag, v, Yx, n)
        else:
            n_vecs = v.shape[1]
            from .nbfuc.sparse_mul_nb import _dia_matvecs_parallel1
            _dia_matvecs_parallel1(diag, v, Yx, n, n_vecs)
    elif overwrite and a is not None:
        if v.ndim == 1:
            from .nbfuc.sparse_mul_nb import _dia_matvec_parallel2
            _dia_matvec_parallel2(diag, v, Yx, n, a)
        else:
            n_vecs = v.shape[1]
            from .nbfuc.sparse_mul_nb import _dia_matvecs_parallel2
            _dia_matvecs_parallel2(diag, v, Yx, n, n_vecs, a)
    elif not overwrite and a is None:
        if v.ndim == 1:
            from .nbfuc.sparse_mul_nb import _dia_matvec_parallel3
            _dia_matvec_parallel3(diag, v, Yx, n)
        else:
            n_vecs = v.shape[1]
            from .nbfuc.sparse_mul_nb import _dia_matvecs_parallel3
            _dia_matvecs_parallel3(diag, v, Yx, n, n_vecs)
    else:
        if v.ndim == 1:
            from .nbfuc.sparse_mul_nb import _dia_matvec_parallel4
            _dia_matvec_parallel4(diag, v, Yx, n, a)
        else:
            n_vecs = v.shape[1]
            from .nbfuc.sparse_mul_nb import _dia_matvecs_parallel4
            _dia_matvecs_parallel4(diag, v, Yx, n, n_vecs, a)
