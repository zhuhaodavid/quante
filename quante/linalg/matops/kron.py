# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:10:19
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-27 22:27:39


import math
import numpy as _np
import scipy.sparse as _sparse
import itertools
from typing import Union, Optional
import functools

# __all__ = [
#     "kron",
#     "kron_power",
#     "matrix_power",
#     "ikron",
# ]

def kron(
    *ops: Union[_np.ndarray, _sparse.sparray],
    stype: Optional[str] = None,  # 后面的参数只有当输入有稀疏矩阵时才有用
    coo_build: bool = False, 
    chopped: bool = True,
    parallel: bool = False
    ) -> Union[_np.ndarray, _sparse.sparray]:
    """
    计算多个算符的克罗内克积(无论是密集矩阵/稀疏矩阵)。
    只要有一个是稀疏的，结果就是稀疏的。
    
    Examples
    ---------
    >>> op1 = np.array([[1, 2], [3, 4]])
    >>> op2 = np.array([[5, 6], [7, 8]])
    >>> kron(op1, op2)
    array([[ 5,  6, 10, 12],
           [15, 18, 20, 24],
           [ 7,  8, 14, 16],
           [21, 24, 28, 32]])

    Parameters
    ----------
    - ops (Union[_np.ndarray, _sparse.sparray]): 要计算克罗内克积的算符。
    - stype (str, 可选): 返回的稀疏矩阵的类型，可以是 "csr" (压缩稀疏行)、"bsr" (块稀疏行)、"csc" (压缩稀疏列) 或 "coo" (坐标格式)。默认为None。
    - chopped (bool, 可选): 是否在计算之前将算符中的零值设置为0。默认为True。
    - parallel (bool, 可选): 是否使用并行计算。默认为False。
    
    """
    tmp_stype = "coo" if coo_build or stype == "coo" else None
    
    if parallel:
        from ...basicfun.utils_numba import parallel_reduce
        reducer = parallel_reduce
    else:
        reducer = functools.reduce
        
    kron_fuc = functools.partial(kron_dispatch, stype=tmp_stype, chopped=chopped)
    X = reducer(kron_fuc, ops)  # 通过 reduce 定义的程序将 kron 连续作用到 ops 上
    
    if stype is not None:
        return X.asformat(stype)
    
    if coo_build or (_sparse.issparse(X) and X.format == "coo"):
        return X.asformat("csr")
    return X

def kron_dispatch(a, b, stype=None, chopped=True):
    if _sparse.issparse(a) or _sparse.issparse(b):
        return _kron_sparse(a, b, stype=stype, chopped=chopped)

    return _np.kron(a, b)
 

def _kron_sparse(a, b, stype=None, chopped=True):
    if stype is None:
        stype = (
            "bsr"
            if isinstance(b, _np.ndarray) or b.format == "bsr"
            else b.format
            if isinstance(a, _np.ndarray)
            else "csc"
            if a.format == "csc" and b.format == "csc"
            else "csr"
        )
    res = _sparse.kron(a, b).asformat(stype)
    if chopped:
        chop(res)
    return res


def kron_power(A, n):
    if n == 1:
        return A
    elif n % 2 == 0:
        half_power = kron_power(A, n // 2)
        return kron_dispatch(half_power, half_power)
    else:
        return kron_dispatch(A, kron_power(A, n - 1))

def matrix_power(A, n):
    if n == 1:
        return A
    elif n % 2 == 0:
        half_power = matrix_power(A, n // 2)
        return half_power @ half_power
    else:
        return A @ matrix_power(A, n - 1)

def chop(data, tol=1.0e-15, inplace=True):
    """
    将密集或稀疏数组中小于某个阈值的值设置为0。

    Parameters
    ----------
    - qob (密集或稀疏向量或算符): 要处理的量子对象。
    - tol (float, 可选): 低于该阈值的值将被设置为0，阈值是 ``max(abs(qob))`` 的分数。默认为1e-10。
    - inplace (bool, 可选): 是否在输入数组上操作或返回副本。默认为False。
    """
    minm = _np.abs(data).max() * tol  # minimum value tolerated
    if not inplace:
        data = data.copy()
    if _sparse.issparse(data):
        data.data.real[_np.abs(data.data.real) < minm] = 0.0
        if _np.issubdtype(data.dtype, _np.complexfloating):
            data.data.imag[_np.abs(data.data.imag) < minm] = 0.0
        data.eliminate_zeros()
    else:
        data.real[_np.abs(data.real) < minm] = 0.0
        data.imag[_np.abs(data.imag) < minm] = 0.0
    return data

#####################################
# ikron
#####################################

# todo 重写这段代码
def ikron(
    ops: Union[_np.ndarray, _sparse.sparray],
    dims: _np.ndarray,
    inds: Union[int, _np.ndarray],
    sparse: Optional[bool] = None,
    stype: Optional[str] = None,
    coo_build: bool = False,
    parallel: bool = False,
) -> Union[_np.ndarray, _sparse.sparray]:
    """
    多个单位一起生成
    
    Examples
    --------
    >>> xy = op.pauli_matrix('xy')
    >>> dims = (2, ) * 10
    >>> mat = qla.ikron(xy, dims, [3,4])

    得到的就是 iiixyiiiii
    """
    # 确保ops是一个列表
    if isinstance(ops, (_np.ndarray, _sparse.spmatrix, _sparse.sparray)):
        ops = (ops,)
    # 获取ops中所有操作数的数据类型
    dtype = common_type(*ops)
    # 确保维度和坐标已经被展平
    if _np.ndim(dims) > 1:
        raise NotImplementedError
    # 确保inds是一个列表
    elif _np.ndim(inds) == 0:
        inds = (inds,)
    # 从ops列表中推断稀疏性
    if sparse is None:
        sparse = any(_sparse.issparse(op) for op in ops)
    # 去除重复的inds
    if len(set(inds)) != len(inds):
        newops = []
        newinds = []
        for i, indsi in enumerate(inds):
            if indsi not in newinds:
                newops.append(ops[i])
                newinds.append(indsi)
            else:
                k = newinds.index(indsi)
                newops[k] = newops[k] @ ops[i]
        inds, ops = newinds, newops
    # 创建一个按索引排序的操作数列表
    inds, ops = zip(*sorted(zip(inds, itertools.cycle(ops))))
    inds, ops = set(inds), iter(ops)
    eye_kws = {
        "sparse": sparse,
        "stype": "coo",
        "dtype": dtype,
    }
    def gen_ops():
        cff_id = 1  # 记录压缩相邻的单位矩阵
        cff_ov = 1  # 记录叠加操作到多个维度
        for ind, dim in enumerate(dims):
            # 检查是否在此放置操作数
            if ind in inds:
                # 检查是否需要前置的单位矩阵
                if cff_id > 1:
                    yield eye(cff_id, **eye_kws)
                    cff_id = 1  # 重置单位矩阵大小
                # 检查是否是放置块中的第一个子系统
                if cff_ov == 1:
                    op = next(ops)
                    sz_op = op.shape[0]
                # 最终维度（块或总维度）-> 放置操作数
                if cff_ov * dim == sz_op or dim == -1:
                    yield op
                    cff_ov = 1
                # 累积子维度
                else:
                    cff_ov *= dim
            # 检查是否在多个子系统中叠加操作数
            elif cff_ov > 1:
                cff_ov *= dim
            # 否则累积相邻的单位矩阵
            else:
                cff_id *= dim
        # 检查是否需要后置的单位矩阵
        if cff_id > 1:
            yield eye(cff_id, **eye_kws)
    # 返回克罗内克积
    return kron(
        *gen_ops(),
        stype=stype,
        coo_build=coo_build,
        parallel=parallel
    )


_COMPLEX_DTYPES = {"complex64", "complex128"}
_DOUBLE_DTYPES = {"float64", "complex128"}
_DTYPE_MAP = {
    (False, False): "float32",
    (False, True): "float64",
    (True, False): "complex64",
    (True, True): "complex128",
}

def common_type(*arrays):
    """Quick compute the minimal dtype sufficient for ``arrays``."""
    dtypes = {array.dtype.name for array in arrays}
    has_complex = not _COMPLEX_DTYPES.isdisjoint(dtypes)
    has_double = not _DOUBLE_DTYPES.isdisjoint(dtypes)
    return _DTYPE_MAP[has_complex, has_double]

def eye(dim, sparse=False, stype="csr", dtype=float):
    if sparse:
        return _sparse.eye(dim, dtype=dtype, format=stype)
    else:
        return _np.eye(dim, dtype=dtype)
    
