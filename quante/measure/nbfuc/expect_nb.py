# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 18:52:19
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 18:54:47

import numpy as _np
from ...basicfun.utils_numba import pnjit, prange, numba_cache_dir, config


config.CACHE_DIR = numba_cache_dir
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

config.CACHE_DIR = numba_cache_dir
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

