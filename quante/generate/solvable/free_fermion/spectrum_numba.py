# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-08-12 15:55:13
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-09 17:21:17

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


@njit(parallel=True)
def get_full_sprem_pblock(gdeng, omega, L, tag, h):
    reslist = _np.zeros(2**L, dtype=omega.dtype)
    ptag = _np.ones(2**L, dtype=_np.int32)
    vtag = _np.ones(2**L, dtype=_np.int32)
    if h < 0:
        vtag *= -1
    tmp3 = 1 - L % 2
    for i in prange(2**L):
        tmp2 = _decimal(i, L) == 1
        res = gdeng + 2 * _np.sum(omega[tmp2])
        ii = sum(tmp2)
        if _np.prod(tag[tmp2]) == -(-1)**((ii+1)//2):
            ptag[i] = -1
        if ii % 2 == tmp3:
            vtag[i] *= -1
        reslist[i] = res
    return reslist, ptag, vtag


