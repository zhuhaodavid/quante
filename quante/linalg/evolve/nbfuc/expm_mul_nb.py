# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:20:07
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 17:15:24

import numpy as _np
from ....basicfun.utils_numba import njit, prange, pnjit

from ...matops.sparse_mul import dot_parallel

class NpLinearAlgebraUtils:

    @staticmethod
    def update_device(x0):
        pass

    @staticmethod
    def apply(A, x):
        return dot_parallel(A, x)
        return A @ x

    @staticmethod
    def norm(x):
        return _np.linalg.norm(x)

    @staticmethod
    def inner(x, y):
        return vdot(x, y)
        # return np.vdot(x, y)

    @staticmethod
    def zeros_like(x):
        return _np.zeros_like(x)

    @staticmethod
    def zeros(shape, dtype=None):
        return _np.zeros(shape, dtype=dtype)

    @staticmethod
    def add_(x, y, alpha=None):
        addself(x, y, alpha)
        return x

    @staticmethod
    def sub_(x, y, alpha=None):
        addself(x, y, -alpha)
        return x

    @staticmethod
    def div_(x, alpha):
        prodscale(x, 1/alpha)
        return x

    @staticmethod
    def mul_(x, alpha):
        prodscale(x, alpha)
        return x

    @staticmethod
    def matmul(A, B):
        return A @ B

    @staticmethod
    def isrealobj(x):
        return _np.isrealobj(x)



@pnjit
def vdot(x, y):
    res = 0
    for i in prange(len(x)):
        res += _np.conj(x[i]) * y[i]
    return res

def addself(a, b, coef):
    if _np.iscomplexobj(a):
        addself_complex(a, b, coef)
    else:
        addself_float(a, b, coef)

@pnjit
def addself_complex(a, b, coef):
    for i in prange(len(a)):
        bi = b[i]
        a[i] += coef * bi.real + (coef * bi.imag)*1j
            
@pnjit('void(float64[:], float64[:], float64)')
def addself_float(a, b, coef):
    for i in prange(len(a)):
        a[i] += coef * b[i]

def prodscale(a, coef):
    if _np.iscomplexobj(a):
        prodscale_complex(a, coef)
    else:
        prodscale_float(a, coef)

@pnjit('void(complex128[:], float64)')
def prodscale_complex(a, coef):
    for i in prange(len(a)):
        ai = a[i]
        a[i] = coef * ai.real + (coef * ai.imag)*1j

@pnjit('void(float64[:], float64)')
def prodscale_float(a, coef):
    for i in prange(len(a)):
        a[i] = coef * a[i]

@pnjit
def addtwo(a, b):
    for i in prange(len(a)):
        a[i] += b[i]