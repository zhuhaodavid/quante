# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:17:43
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:19:03

import numpy as _np
import scipy
from .nbfuc.sparse_mul_nb import _csr_matvec_parallel, _csr_matvecs_parallel

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

