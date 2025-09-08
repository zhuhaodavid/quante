# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-08-12 15:55:13
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-08 19:47:13

from ....basicfun.utils_numba import njit, vectorize

from ....basicfun.utils_numba import prange as prange
import numpy as _np

@njit
def _get_full_sprem(gdeng, tmp, L):
    res = []
    for i in prange(2**L):
        res.append(gdeng + 2 * _np.sum(tmp[_decimal(i, L) == 1]))
    return res


@njit
def _decimal(num, L):
    arry = _np.zeros(L, dtype=_np.int32)
    for j in range(L):
        arry[j] = num % 2
        num = num // 2
        if num == 0:
            break
    return arry


@vectorize
def _logcosh(x):
    if x < 16:
        return _np.log(_np.cosh(x))
    else:
        return x - _np.log(2)

