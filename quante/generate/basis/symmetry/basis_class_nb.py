# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-19 14:26:06
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-17 15:54:01

from ....linalg.usenumba.numba_settings import njit, config, numba_cache_dir, pnjit, prange

config.CACHE_DIR = numba_cache_dir
@njit
def _is_diagonal(row,col):
	for i in range(row.size):
		if row[i] != col[i]:
			return False
	return True

config.CACHE_DIR = numba_cache_dir
@njit
def _update_diag(diag,ind,ME):
	for i in range(ind.size):
		diag[ind[i]] += ME[i]

config.CACHE_DIR = numba_cache_dir
@pnjit
def add_(a, b):
    for i in prange(len(a)):
        a[i] += b[i]

import scipy.sparse as _sp
import numpy as np

def _get_index_type(Ns):
    if Ns < np.iinfo(np.int16).max:
        return np.int16
    elif Ns < np.iinfo(np.int32).max:
        return np.int32
    else:
        return np.int64

def coodiaglists2csr(row_result, col_result, ele_result, diag, n_row, index_type, dtype):
    nnz = sum(len(row) for row in row_result) + (len(diag) if diag is not None else 0)
    new_index_type = _get_index_type(nnz)
    Bp = np.zeros(n_row+1, dtype=new_index_type)
    Bj = np.zeros(nnz, dtype=index_type)
    Bx = np.zeros(nnz, dtype=dtype)
    for row in row_result:
        addBp(Bp, row)
    if diag is not None:
        addone(Bp)
    getBp(Bp, nnz, n_row)
    for row, col, ele in zip(row_result, col_result, ele_result):
        writeA2B(row, col, ele, Bp, Bj, Bx)
    if diag is not None:
        writediag(diag, Bp, Bj, Bx)
    ajustBp(Bp, n_row)
    csr = _sp.csr_array((Bx, Bj, Bp), shape=(n_row, n_row), dtype=dtype)
    return sum_duplicates(csr)

import scipy.sparse._sparsetools as _sparsetools

def sum_duplicates(csr):
    if csr.has_canonical_format:
        return csr
    if not csr.has_sorted_indices:
        csr_sort_indices(len(csr.indptr) - 1, csr.indptr, csr.indices, csr.data)
        # _sparsetools.csr_sort_indices(len(csr.indptr) - 1, csr.indptr,csr.indices, csr.data)
        csr.has_sorted_indices = True

    M, N = csr._swap(csr.shape)
    _sparsetools.csr_sum_duplicates(M, N, csr.indptr, csr.indices,
                                    csr.data)

    csr.prune()  # nnz may have changed
    csr.has_canonical_format = True
    return csr

config.CACHE_DIR = numba_cache_dir
@njit(parallel=True)
def csr_sort_indices(n_row, Ap, Aj, Ax):
    for i in prange(n_row):
        row_start = Ap[i]
        row_end = Ap[i+1]
        
        row_length = row_end - row_start
        tempj = np.empty(row_length, dtype=Aj.dtype)
        tempx = np.empty(row_length, dtype=Ax.dtype)
        tempj[:] = Aj[row_start:row_end]
        tempx[:] = Ax[row_start:row_end]
        
        sort_indices = tempj.argsort()
        Aj[row_start:row_end] = tempj[sort_indices]
        Ax[row_start:row_end] = tempx[sort_indices]

def coolists2csr2(row_result, col_result, ele_result, n_row, dtype):
    conc_row = np.concatenate(row_result)
    conc_col = np.concatenate(col_result)
    conc_ele = np.concatenate(ele_result)
    return _sp.csr_array((conc_ele, (conc_row, conc_col)), shape=(n_row, n_row), dtype=dtype)

config.CACHE_DIR = numba_cache_dir
@njit
def addBp(Bp, row):
    for n in prange(len(row)):
        Bp[row[n]] += 1

config.CACHE_DIR = numba_cache_dir
@njit
def addone(Bp):
    for n in prange(len(Bp)):
        Bp[n] += 1

config.CACHE_DIR = numba_cache_dir
@njit
def getBp(Bp, nnz, n_row):
    cumsum = 0
    for i in range(n_row):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[n_row] = nnz

config.CACHE_DIR = numba_cache_dir
@njit
def writeA2B(row, col, ele, Bp, Bj, Bx):
    for n in range(len(ele)):
        row_n = row[n]
        dest = Bp[row_n]
        Bj[dest] = col[n]
        Bx[dest] = ele[n]
        Bp[row_n] += 1

config.CACHE_DIR = numba_cache_dir
@njit
def writediag(diag, Bp, Bj, Bx):
    for i in prange(len(diag)):
        dest = Bp[i]
        Bj[dest] = i
        Bx[dest] = diag[i]
        Bp[i] += 1

config.CACHE_DIR = numba_cache_dir
@njit
def ajustBp(Bp, n_row):
    last = 0
    for i in range(n_row):
        temp = Bp[i]
        Bp[i] = last
        last = temp
