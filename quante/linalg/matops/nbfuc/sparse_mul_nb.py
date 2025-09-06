# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:17:33
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-05 14:01:16

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


# @njit(parallel=True, fastmath=True)
# def _gram(s_indptr, s_indices, s_data, grammat):
#     M = len(s_indptr) - 1
#     for i in prange(M):
#         i_start, i_end = s_indptr[i], s_indptr[i+1]
#         i_idx = s_indices[i_start:i_end]
#         i_data = s_data[i_start:i_end]
#         for j in range(i, M):
#             j_start, j_end = s_indptr[j], s_indptr[j+1]
#             j_idx = s_indices[j_start:j_end]
#             j_data = s_data[j_start:j_end]
#             ct1 = 0
#             ct2 = 0
#             while ct1 < i_idx.shape[0] and ct2 < j_idx.shape[0]:
#                 if i_idx[ct1] < j_idx[ct2]:
#                     ct1 += 1
#                 elif i_idx[ct1] > j_idx[ct2]:
#                     ct2 += 1
#                 else:
#                     grammat[i, j] += i_data[ct1] * np.conj(j_data[ct2])
#                     ct1 += 1
#                     ct2 += 1
#             if i != j:
#                 grammat[j, i] = np.conj(grammat[j, i])

# def gram(A, format='AAT'):
#     if format == 'ATA':
#         A = A.T.conj()
#     elif format == 'AAT':
#         pass
#     else:
#         raise ValueError("format must be 'ATA' or 'AAT'")

#     A = A.tocsr()
#     m, n = A.shape
#     grammat = np.zeros((m, m), dtype=A.dtype)
#     _gram(A.indptr, A.indices, A.data, grammat)
#     return grammat

#  this is not faster than scipy