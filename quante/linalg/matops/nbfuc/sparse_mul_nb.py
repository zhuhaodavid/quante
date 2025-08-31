# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:17:33
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 18:59:00

from ....basicfun.utils_numba import njit, prange


@njit(parallel=True, boundscheck=False)
def _csr_matvec_parallel1(n_row, Ap, Aj, Ax, Xx, Yx):
    for i in prange(n_row):
        s = 0.
        for jj in range(Ap[i], Ap[i+1]):
            s += Ax[jj] * Xx[Aj[jj]]
        Yx[i] = s

@njit(parallel=True, boundscheck=False)
def _csr_matvecs_parallel1(n_row, n_vecs, Ap, Aj, Ax, Xx, Yx):
    for i in prange(n_row):
        for ii in range(n_vecs):
            y = 0.
            for jj in range(Ap[i], Ap[i+1]):
                y += Ax[jj] * Xx[Aj[jj], ii]
            Yx[i, ii] = y

@njit(parallel=True, boundscheck=False)
def _csr_matvec_parallel2(n_row, Ap, Aj, Ax, Xx, Yx, a):
    for i in prange(n_row):
        s = 0.
        for jj in range(Ap[i], Ap[i+1]):
            s += Ax[jj] * Xx[Aj[jj]]
        Yx[i] = a * s

@njit(parallel=True, boundscheck=False)
def _csr_matvecs_parallel2(n_row, n_vecs, Ap, Aj, Ax, Xx, Yx, a):
    for i in prange(n_row):
        for ii in range(n_vecs):
            y = 0.
            for jj in range(Ap[i], Ap[i+1]):
                y += Ax[jj] * Xx[Aj[jj], ii]
            Yx[i, ii] = a * y

@njit(parallel=True, boundscheck=False)
def _csr_matvec_parallel3(n_row, Ap, Aj, Ax, Xx, Yx):
    for i in prange(n_row):
        s = 0.
        for jj in range(Ap[i], Ap[i+1]):
            s += Ax[jj] * Xx[Aj[jj]]
        Yx[i] += s

@njit(parallel=True, boundscheck=False)
def _csr_matvecs_parallel3(n_row, n_vecs, Ap, Aj, Ax, Xx, Yx):
    for i in prange(n_row):
        for ii in range(n_vecs):
            y = 0.
            for jj in range(Ap[i], Ap[i+1]):
                y += Ax[jj] * Xx[Aj[jj], ii]
            Yx[i, ii] += y

@njit(parallel=True, boundscheck=False)
def _csr_matvec_parallel4(n_row, Ap, Aj, Ax, Xx, Yx, a):
    for i in prange(n_row):
        s = 0.
        for jj in range(Ap[i], Ap[i+1]):
            s += Ax[jj] * Xx[Aj[jj]]
        Yx[i] += a * s

@njit(parallel=True, boundscheck=False)
def _csr_matvecs_parallel4(n_row, n_vecs, Ap, Aj, Ax, Xx, Yx, a):
    for i in prange(n_row):
        for ii in range(n_vecs):
            y = 0.
            for jj in range(Ap[i], Ap[i+1]):
                y += Ax[jj] * Xx[Aj[jj], ii]
            Yx[i, ii] += a * y



@njit(parallel=True, boundscheck=False)
def _dia_matvec_parallel1(diag, v, Yx, n):
    for i in prange(n):
        Yx[i] = diag[i] * v[i]

@njit(parallel=True, boundscheck=False)
def _dia_matvecs_parallel1(diag, v, Yx, n, n_vecs):
    for i in prange(n):
        for ii in range(n_vecs):
            Yx[i, ii] = diag[i] * v[i, ii]

@njit(parallel=True, boundscheck=False)
def _dia_matvec_parallel2(diag, v, Yx, n, a):
    for i in prange(n):
        Yx[i] = a * diag[i] * v[i]

@njit(parallel=True, boundscheck=False)
def _dia_matvecs_parallel2(diag, v, Yx, n, n_vecs, a):
    for i in prange(n):
        for ii in range(n_vecs):
            Yx[i, ii] = a * diag[i] * v[i, ii]

@njit(parallel=True, boundscheck=False)
def _dia_matvec_parallel3(diag, v, Yx, n):
    for i in prange(n):
        Yx[i] += diag[i] * v[i]

@njit(parallel=True, boundscheck=False)
def _dia_matvecs_parallel3(diag, v, Yx, n, n_vecs):
    for i in prange(n):
        for ii in range(n_vecs):
            Yx[i, ii] += diag[i] * v[i, ii]

@njit(parallel=True, boundscheck=False)
def _dia_matvec_parallel4(diag, v, Yx, n, a):
    for i in prange(n):
        Yx[i] += a * diag[i] * v[i]

@njit(parallel=True, boundscheck=False)
def _dia_matvecs_parallel4(diag, v, Yx, n, n_vecs, a):
    for i in prange(n):
        for ii in range(n_vecs):
            Yx[i, ii] += a * diag[i] * v[i, ii]
