# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:41:37
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 09:57:18

import numpy as _np
from .....basicfun.utils_numba import njit, numba_cache_dir, config

config.CACHE_DIR = numba_cache_dir
@njit
def _factorial_prod(N, arr):
    arr[:int(N)] += 1

config.CACHE_DIR = numba_cache_dir
@njit
def _factorial_div(N, arr):
    arr[:int(N)] -= 1

config.CACHE_DIR = numba_cache_dir
@njit
def _to_long(arr):
    prod = 1
    for i, v in enumerate(arr):
        prod *= (i+1.0)**int(v)
    return prod

config.CACHE_DIR = numba_cache_dir
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