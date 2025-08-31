# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-09 18:40:16
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 15:35:13


import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch as tc
from warnings import warn
sp = scipy.sparse

def to_sparse(tsr, *, device=None, dtype=None) -> tc.Tensor:
    assert tsr.ndim == 2, "只能处理二维张量"
    if dtype is None:
        dtype = tc.complex128 if np.iscomplexobj(tsr) else tc.float64
        
    if isinstance(tsr, (sp.csr_array, sp.csr_matrix)):
        return tc.sparse_csr_tensor(tsr.indptr, tsr.indices, tsr.data, tsr.shape, dtype=dtype, device=device)
    elif isinstance(tsr, (sp.coo_array, sp.coo_matrix)):
        return tc.sparse_coo_tensor([tsr.row, tsr.col], tsr.data, tsr.shape, dtype=dtype, device=device)
    elif isinstance(tsr, (sp.csc_array, sp.csc_matrix)):
        return tc.sparse_csc_tensor(tsr.indptr, tsr.indices, tsr.data, tsr.shape, dtype=dtype, device=device)
    else:
        raise NotImplementedError(
            f"converting {type(tsr)} to torch.Tensor not supported yet."
        )

def to_scipy(tsr):
    if tsr.is_sparse_csr:  # csr
        return sp.csr_array((tsr.values().cpu().numpy(), 
                    tsr.col_indices().cpu().numpy(),
                    tsr.crow_indices().cpu().numpy(), 
        ), shape=tsr.shape)
    elif tsr.is_sparse:
        rowcol = tsr._indices().cpu().numpy()
        data = tsr._values().cpu().numpy()
        return sp.coo_array((data, rowcol), shape=tsr.shape)
    elif tsr.layout == tc.sparse_csc:
        return sp.csc_array((tsr.values().cpu().numpy(),
                tsr.row_indices().cpu().numpy(),
                tsr.ccol_indices().cpu().numpy()),
            shape=tsr.shape)
    else:
        raise NotImplementedError(
            f"Sparse layout {type(tsr)} not supported yet."
        )

def to_csr(tsr:scipy.sparse.csr_array, *, device=None, dtype=None) -> tc.Tensor:
    """
    将一个 scipy csr 格式转换为 torch csr 格式
    """
    assert tsr.ndim == 2, "只能处理二维张量"
    if dtype is None:
        dtype = tc.complex128 if np.iscomplexobj(tsr) else tc.float64
    tsr = tsr.tocsr()
    return tc.sparse_csr_tensor(tsr.indptr, tsr.indices, tsr.data, tsr.shape, dtype=dtype, device=device)

def eye(n:int, *, device=None, dtype=None) -> tc.Tensor:
    """
    生成一个单位矩阵
    """
    return to_csr(scipy.sparse.eye(n, format='csr'), device=device, dtype=dtype)

def trace(tsr:tc.Tensor) -> tc.Tensor:
    """
    计算一个 csr 格式的矩阵的迹
    """
    if isinstance(tsr, scipy.sparse.csr_array):
        return tsr.trace()
    elif isinstance(tsr, np.ndarray):
        return np.trace(tsr)
    elif isinstance(tsr, tc.Tensor) and not tsr.is_sparse_csr:
        return tc.trace(tsr)
    
    if tsr.is_cuda:
        try:
            import cupyx as cpx
            import cupy as cp
            cpres = cpx.scipy.sparse.csr_matrix((cp.asarray(tsr.values()), cp.asarray(tsr.col_indices()), cp.asarray(tsr.crow_indices())), shape=tsr.shape).diagonal().sum()
            res = tc.tensor(cpres, device=tsr.device)
            del cpres
            cp.get_default_memory_pool().free_all_blocks()
            return res
        except ImportError:
            warn("cupy is not avilable, use scipy")
            spres = scipy.sparse.csr_array((tsr.values().cpu().numpy(), tsr.col_indices().cpu().numpy(), tsr.crow_indices().cpu().numpy())).trace()
            return tc.tensor(spres, device=tsr.device)
    else:
        spres = scipy.sparse.csr_array((tsr.values().numpy(), tsr.col_indices().numpy(), tsr.crow_indices().numpy())).trace()
        return tc.tensor(spres, device=tsr.device)

def cptrace(tsr):
    return tsr.diagonal().sum()

def norm(tsr:tc.Tensor, ord=None) -> tc.Tensor:

    """
    Norm of a sparse matrix

    This function is able to return one of seven different matrix norms,
    depending on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : a sparse matrix
        Input sparse matrix.
    ord : {non-zero int, inf, -inf, 'fro'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned.

    Returns
    -------
    n : float or ndarray

    Notes
    -----
    Some of the ord are not implemented because some associated functions like,
    _multi_svd_norm, are not yet available for sparse matrix.

    This docstring is modified based on numpy.linalg.norm.
    https://github.com/numpy/numpy/blob/main/numpy/linalg/linalg.py

    The following norms can be calculated:

    =====  ============================
    ord    norm for sparse matrices
    =====  ============================
    None   Frobenius norm
    'fro'  Frobenius norm
    inf    max(sum(abs(x), axis=1))
    -inf   min(sum(abs(x), axis=1))
    0      abs(x).sum(axis=axis)
    1      max(sum(abs(x), axis=0))
    -1     min(sum(abs(x), axis=0))
    2      Spectral norm (the largest singular value)
    -2     Not implemented
    other  Not implemented
    =====  ============================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
        Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
    """
    if isinstance(tsr, scipy.sparse.csr_array):
        return scipy.sparse.linalg.norm(scipy.sparse.csr_matrix(tsr), ord)
    elif isinstance(tsr, np.ndarray):
        return np.linalg.norm(tsr, ord)
    elif isinstance(tsr, tc.Tensor) and not tsr.is_sparse_csr:
        return tc.norm(tsr, ord)
    ord = np.inf if ord is tc.inf else ord
    
    if tsr.is_cuda:
        try:
            if ord in (None, 'fro', 'f'):
                return tc.norm(tsr.values())
            if ord == 2:
                raise ImportError("cupy 不支持 2 范数")
            import cupyx as cpx
            import cupy as cp
            cpres = cpx.scipy.sparse.csr_matrix((cp.asarray(tsr.values()), cp.asarray(tsr.col_indices()), cp.asarray(tsr.crow_indices())), shape=tsr.shape)
            res = tc.tensor(cpnorm(cpres, ord), device=tsr.device)
            del cpres
            cp.get_default_memory_pool().free_all_blocks()
            return res
        except ImportError as e:
            warn(str(e) + ", use scipy")
            spmat = scipy.sparse.csr_matrix((tsr.values().cpu().numpy(), tsr.col_indices().cpu().numpy(), tsr.crow_indices().cpu().numpy()))
            spnorm = scipy.sparse.linalg.norm(spmat, ord)
            return tc.tensor(spnorm, device=tsr.device)
    else:
        spmat = scipy.sparse.csr_matrix((tsr.values().numpy(), tsr.col_indices().numpy(), tsr.crow_indices().numpy()))
        spnorm = scipy.sparse.linalg.norm(spmat, ord)
        return tc.tensor(spnorm, device=tsr.device)

def cpnorm(tsr, ord=None):
    from cupyx.scipy.sparse.linalg import norm
    if ord == 1:
        return abs(tsr).sum(axis=0).max()
    elif ord == np.inf:
        return abs(tsr).sum(axis=1).max()
    return norm(tsr, ord)