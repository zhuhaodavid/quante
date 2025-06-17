# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:17:33
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:21:00

from ....basicfun.utils_numba import njit, prange


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

