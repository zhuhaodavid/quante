# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-14 00:59:50
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-14 23:10:04

import numpy as np
import cupy as cp
import torch as tc
from numba import cuda
import cupyx as cpx

from ...linalg.usenumba.numba_settings import numba_cache_dir, config
from ...generate.symmetry.basis_class_nb import _is_diagonal, _update_diag, add_, _get_index_type
from ...generate.symmetry.spin_half.bitsoperation import operateon

def to_matrix_cuda(basis, eachterm, hascomplex):
    from ...generate.symmetry.spin_half.noblock.defclass import SpinHalfBasisNoBlock
    if isinstance(basis, (SpinHalfBasisNoBlock, )):
        Bp_cuda, Bj_cuda, Bx_cuda, n_row = matrix_fullcuda(basis, eachterm, hascomplex)
    else:
        Bp_cuda, Bj_cuda, Bx_cuda, n_row = matrix_semicuda(basis, eachterm, hascomplex)
    Bp_torch = tc.as_tensor(cp.asarray(Bp_cuda), device='cuda')
    Bj_torch = tc.as_tensor(cp.asarray(Bj_cuda), device='cuda')
    Bx_torch = tc.as_tensor(cp.asarray(Bx_cuda), device='cuda')
    return tc.sparse_csr_tensor(Bp_torch, Bj_torch, Bx_torch, (n_row, n_row))
    # return Bp_torch, Bj_torch, Bx_torch, (n_row, n_row)
    # return cpx.scipy.sparse.csr_matrix((cp.asarray(Bx_cuda), cp.asarray(Bj_cuda), cp.asarray(Bp_cuda)), shape=(n_row, n_row))

def matrix_semicuda(basis, op_list, hascomplex):
    diag = None
    dtype = np.complex128 if hascomplex or basis.default_complex else np.float64
    # 预设内存，避免反复分配内存
    real_Ns = basis.Ns * basis._double_Ns
    index_type = _get_index_type(real_Ns)
    index_type = np.int32 if index_type == np.int16 else index_type
    
    row_init = np.empty(real_Ns, dtype=index_type)
    col_init = np.empty(real_Ns, dtype=index_type)
    ele_int = np.empty(real_Ns, dtype=dtype)
    
    row_result = []
    col_result = []
    ele_result = []
    
    for opnm, posn, coef in op_list:
        row, col, ele = basis._Op(opnm, posn, coef, row_init, col_init, ele_int)  # 主要的时间花费，一半时间花在这里
        if(len(ele)>0):
            if row is None:
                if diag is None:
                    diag = ele
                else:
                    add_(diag, ele)
                    # diag += ele
            elif _is_diagonal(row,col):
                if diag is None:
                    diag = np.zeros(basis.Ns,dtype=dtype)
                _update_diag(diag,row,ele)
            else:
                row_result.append(cp.asarray(row))
                col_result.append(col)
                ele_result.append(ele)
    
    # 从这一步开始在 GPU 上运行
    if len(ele_result) > 0:
        return coodiaglists2csr(row_result, col_result, ele_result, diag, basis.Ns, index_type, dtype)
    else:
        return diag


# 一开始就子 GPU 上运行
def matrix_fullcuda(self, op_list, hascomplex):
    diag = None
    off_diag = None
    dtype = cp.complex128 if hascomplex or self.default_complex else cp.float64
    # 预设内存，避免反复分配内存
    real_Ns = self.Ns * self._double_Ns
    index_type = _get_index_type(real_Ns)
    index_type = np.int32 if index_type == np.int16 else index_type
    
    row_init = cp.empty(real_Ns, dtype=index_type)
    col_init = cp.empty(real_Ns, dtype=index_type)
    ele_init = cp.zeros(real_Ns, dtype=dtype)
    
    row_result = []
    col_result = []
    ele_result = []
    
    nthreads, nblocks = get_thread_blocks(real_Ns)
    for opnm, posn, coef in op_list:
        opnm_cuda = cp.asarray(opnm)
        posn_cuda = cp.asarray(posn)
        coef_cuda = cp.asarray([coef])
        noblock_op[nblocks,nthreads](opnm_cuda, posn_cuda, coef_cuda, self.L, self.Ns, row_init, col_init, ele_init)
        mask = cp.logical_not(cp.abs(ele_init)==0.0)
        row, col, ele = row_init[mask], col_init[mask], ele_init[mask]
        if(len(ele)>0):
            if row is None:
                if diag is None:
                    diag = ele
                else:
                    # add_(diag, ele)
                    diag += ele
            elif cp.allclose(row, col):
                if diag is None:
                    diag = cp.zeros(self.Ns,dtype=dtype)
                _update_diag_cuda[nblocks,nthreads](diag,row,ele)
            else:
                ele_result.append(ele)
                row_result.append(row)
                col_result.append(col)
        
    if len(ele_result) > 0:
        Bp_cuda, Bj_cuda, Bx_cuda, n_row = coodiaglists2csr(row_result, col_result, ele_result, diag, self.Ns, index_type, dtype)   # 主要的时间花费，一半时间花在这里
        del row_result, col_result, ele_result
        cp.get_default_memory_pool().free_all_blocks()
        return sum_duplicates(Bp_cuda, Bj_cuda, Bx_cuda, n_row)  # 主要的时间花费，一半时间花在这里
    else:
        raise diag

def coodiaglists2csr(row_result, col_result, ele_result, diag, n_row, index_type, dtype):
    nnz = sum(len(row) for row in row_result) + (len(diag) if diag is not None else 0)
    new_index_type = _get_index_type(nnz)
    new_index_type = np.int32 if new_index_type == np.int16 else new_index_type
    lenBp = n_row + 1
    Bp_cuda = cp.zeros(n_row + 1, dtype=new_index_type)
    
    for row in row_result:
        nthreads, nblocks = get_thread_blocks(len(row))
        addBp_cuda[nblocks,nthreads](Bp_cuda, row)
    
    if diag is not None:
        nthreads, nblocks = get_thread_blocks(lenBp)
        addone_cuda[nblocks,nthreads](Bp_cuda)
    
    nnz_cuda = cp.array([nnz], dtype=new_index_type)
    n_row_cuda = cp.array([n_row], dtype=new_index_type)
    getBp_cuda[nblocks,nthreads](Bp_cuda, nnz_cuda, n_row_cuda)
    
    Bj_cuda = cp.empty(nnz, dtype=index_type)
    Bx_cuda = cp.empty(nnz, dtype=dtype)
    for row, col, ele in zip(row_result, col_result, ele_result):
        nthreads1, nblocks1 = get_thread_blocks(len(ele))
        writeA2B_cuda[nblocks1,nthreads1](row, cp.asarray(col), cp.asarray(ele), Bp_cuda, Bj_cuda, Bx_cuda)
    
    if diag is not None:
        nthreads1, nblocks1 = get_thread_blocks(len(diag))
        writediag_cuda[nblocks1,nthreads1](diag, Bp_cuda, Bj_cuda, Bx_cuda)
    
    ajustBp_cuda[nblocks,nthreads](Bp_cuda, n_row_cuda)
    # return Bp_cuda, Bj_cuda, Bx_cuda, n_row
    return Bp_cuda, Bj_cuda, Bx_cuda, n_row

def sum_duplicates(Bp_cuda, Bj_cuda, Bx_cuda, n_row):
    nthreads, nblocks = get_thread_blocks(n_row)
    needsum = cp.zeros(1, dtype=bool)
    csr_sort_indices_cuda[nblocks, nthreads](n_row, Bp_cuda, Bj_cuda, Bx_cuda, needsum)
    
    # need_sum_duplicates[nblocks, nthreads](n_row, Bp_cuda, Bj_cuda, needsum)
    if needsum[0]:
        newAp = cp.empty(n_row+1, dtype=cp.int32)
        newAp[0] = 0
        csr_sum_duplicates_cuda[nblocks, nthreads](n_row, Bp_cuda, Bj_cuda, Bx_cuda, newAp)
        Bp_cuda[:] = cp.cumsum(newAp)
        mask = Bj_cuda!=-1
        nnz = cp.sum(mask)
        Bx_cuda[:nnz] = Bx_cuda[mask]
        Bj_cuda[:nnz] = Bj_cuda[mask]
        return Bp_cuda, Bj_cuda[:nnz], Bx_cuda[:nnz], n_row
    else:
        return Bp_cuda, Bj_cuda, Bx_cuda, n_row

# ==============下面是辅助函数================

def _get_index_type(Ns):
    if Ns < np.iinfo(np.int16).max:
        return np.int16
    elif Ns < np.iinfo(np.int32).max:
        return np.int32
    else:
        return np.int64

def get_thread_blocks(real_Ns):
    nthreads = 256
    nblocks = (real_Ns // nthreads) + 1
    if nblocks < 128:
        nblocks = 128
        nthreads = (real_Ns // nblocks) + 1
    return nthreads, nblocks


config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def noblock_op(opnm, posn, coef, L, M, row_init, col_init, ME_init):
    indx = cuda.grid(1)
    if indx < M:
        opco, t = operateon(opnm, posn, indx, L)
        if t != -1:
            row_init[indx] = t
            col_init[indx] = indx
            ME_init[indx] = opco * coef[0]
        else:
            ME_init[indx] = 0.0

config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def _update_diag_cuda(diag,ind,ME):
    idx = cuda.grid(1)
    if idx < ind.size:
        diag[ind[idx]] += ME[idx]


config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def addBp_cuda(Bp, row):
    idx = cuda.grid(1)
    if idx < len(row):
        # cuda.atomic.add(Bp, row[idx], 1)
        Bp[row[idx]] += 1
    
config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def addone_cuda(Bp):
    idx = cuda.grid(1)
    if idx < len(Bp):
        Bp[idx] += 1

config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def getBp_cuda(Bp, nnz, n_row):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tid == 0:  # todo 只有一个线程执行，怎么优化
        cumsum = 0
        for i in range(n_row[0]):
            temp = Bp[i]
            Bp[i] = cumsum
            cumsum += temp
        Bp[n_row] = nnz[0]

config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def writeA2B_cuda(row, col, ele, Bp, Bj, Bx):
    idx = cuda.grid(1)
    if idx < len(ele):
        row_n = row[idx]
        dest = cuda.atomic.add(Bp, row_n, 1)
        Bj[dest] = col[idx]
        Bx[dest] = ele[idx]

config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def writediag_cuda(diag, Bp, Bj, Bx):
    idx = cuda.grid(1)
    if idx < len(diag):
        dest = cuda.atomic.add(Bp, idx, 1)
        Bj[dest] = idx
        Bx[dest] = diag[idx]

config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def ajustBp_cuda(Bp, n_row):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tid == 0:
        last = 0
        for i in range(n_row[0]):
            temp = Bp[i]
            Bp[i] = last
            last = temp

config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def csr_sort_indices_cuda(n_row, Ap, Aj, Ax, need):
    idx = cuda.grid(1)
    if idx < n_row:
        row_start = Ap[idx]
        row_end = Ap[idx+1]
        row_length = row_end - row_start
        # 通常不长,冒泡排序足够
        for i in range(row_length):  # 外层循环，控制轮数
            for j in range(row_start, row_end - 1 - i):  # 内层循环，逐渐减少比较范围
                if Aj[j] > Aj[j+1]:
                    Aj[j], Aj[j+1] = Aj[j+1], Aj[j]
                    Ax[j], Ax[j+1] = Ax[j+1], Ax[j]
                elif Aj[j] == Aj[j+1]:
                    need[0] = True

config.CACHE_DIR = numba_cache_dir
@cuda.jit(cache=True)
def csr_sum_duplicates_cuda(n_row, Ap, Aj, Ax, newAp):
    idx = cuda.grid(1)
    if idx < n_row:
        row_start = Ap[idx]
        row_end = Ap[idx + 1]
        # 同步
        nzz = row_start
        while row_start < row_end:
            j = Aj[row_start]
            x = Ax[row_start]
            row_start += 1
            while row_start < row_end and Aj[row_start] == j:
                x += Ax[row_start]
                row_start += 1
            Aj[nzz] = j
            Ax[nzz] = x
            nzz += 1
        for i in range(nzz, row_end):
            Aj[i] = -1
        newAp[idx+1] = nzz - Ap[idx]