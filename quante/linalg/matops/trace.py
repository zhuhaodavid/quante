# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:12:15
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:26:33

# todo: 需要优化

import math
import numpy as _np
import scipy.sparse as _sparse
from numbers import Integral

__all__ = [
    "partial_trace",
    "itrace"
]

 # todo quimb 写的太复杂，重写来实现这个功能
def partial_trace(p, dims, keep):
    """
    计算 p 的部分迹，dims 是各个格点的维数，keep 是要保留的格点位置
    
    Examples
    --------
    >>> L = 4
    >>> vec = np.random.rand(2**L)  # 对稀疏向量存在问题
    >>> rho_red = qla.partial_trace(vec, [2]*L, [1])
    >>> print(rho_red)  # np.einsum('abcdaecd->be', rho.reshape([2]*(2*L)))
    
    以及对矩阵：
    
    >>> L = 4
    >>> vec = ed.rdmat_rho(2**L, sparse=True)  # 或 False，必须是厄密的
    >>> rho_red = qla.partial_trace(vec, [2]*L, [1])
    >>> print(rho_red)
    """
    try:
        ndim = dims.ndim
    except AttributeError:
        ndim = len(_find_shape_of_nested_int_array(dims))

    if ndim >= 2:
        raise NotImplementedError
        # dims, keep = dim_map(dims, keep)

    if _sparse.issparse(p):
        return _partial_trace_simple(p, dims, keep)

    return _partial_trace_dense(p, dims, keep)

def _find_shape_of_nested_int_array(x):
    """Take a n-nested list/tuple of integers and find its array shape."""
    shape = [len(x)]
    sub_x = x[0]
    while not _np.issubdtype(type(sub_x), _np.integer):
        shape.append(len(sub_x))
        sub_x = sub_x[0]
    return tuple(shape)

def _partial_trace_simple(p, dims, keep):
    """Simple partial trace made up of consecutive single subsystem partial
    traces, augmented by 'compressing' the dimensions each time.
    """
    dims, keep = dim_compress(dims, keep)
    if len(keep) == 1:
        return _trace_keep(p, dims, *keep)
    lmax = max(enumerate(dims), key=lambda ix: (ix[0] not in keep) * ix[1])[0]
    p = _trace_lose(p, dims, lmax)
    dims = (*dims[:lmax], *dims[lmax + 1 :])
    keep = {(ind if ind < lmax else ind - 1) for ind in keep}
    return _partial_trace_simple(p, dims, keep)

def dim_compress(dims, inds):
    """Compress neighbouring subsytem dimensions.

    Take some dimensions and target indices and compress both, i.e.
    merge adjacent dimensions that are both either in ``dims`` or not. For
    example, if tensoring an operator onto a single site, with many sites
    the identity, treat these as single large identities.
    """
    if isinstance(inds, Integral):
        inds = (inds,)

    dims, inds = zip(*_dim_compressor(dims, inds))
    inds = tuple(i for i, b in enumerate(inds) if b)

    return dims, inds

def _dim_compressor(dims, inds):  # pragma: no cover
    """Helper function for ``dim_compress`` that does the heavy lifting.

    Parameters
    ----------
    dims : sequence of int
        The subsystem dimensions.
    inds : sequence of int
        The indices of the 'marked' subsystems.

    Returns
    -------
    generator of (int, int)
        Sequence of pairs of new dimension subsystem with marked flag {0, 1}.
    """
    blocksize_id = blocksize_op = 1
    autoplace_count = 0
    for i, dim in enumerate(dims):
        if dim < 0:
            if blocksize_op > 1:
                yield (blocksize_op, 1)
                blocksize_op = 1
            elif blocksize_id > 1:
                yield (blocksize_id, 0)
                blocksize_id = 1
            autoplace_count += dim
        elif i in inds:
            if blocksize_id > 1:
                yield (blocksize_id, 0)
                blocksize_id = 1
            elif autoplace_count < 0:
                yield (autoplace_count, 1)
                autoplace_count = 0
            blocksize_op *= dim
        else:
            if blocksize_op > 1:
                yield (blocksize_op, 1)
                blocksize_op = 1
            elif autoplace_count < 0:
                yield (autoplace_count, 1)
                autoplace_count = 0
            blocksize_id *= dim
    yield (
        (blocksize_op, 1)
        if blocksize_op > 1
        else (blocksize_id, 0)
        if blocksize_id > 1
        else (autoplace_count, 1)
    )

def _trace_keep(p, dims, keep):
    """Simple partial trace where the single subsytem
    at ``keep`` is kept.
    """
    dims = _np.asarray(dims)
    s = dims[keep]
    a = math.prod(dims[:keep])
    b = math.prod(dims[keep + 1 :])
    rhos = _np.zeros(shape=(s, s), dtype=_np.complex128)
    for i in range(s):
        for j in range(i, s):
            for k in range(a):
                i_i = b * i + s * b * k
                i_f = b * i + s * b * k + b
                j_i = b * j + s * b * k
                j_f = b * j + s * b * k + b
                rhos[i, j] += p[i_i:i_f, j_i:j_f].trace()
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos

def _trace_lose(p, dims, lose):
    """Simple partial trace where the single subsytem at ``lose``
    is traced out.
    """
    dims = _np.asarray(dims)
    e = dims[lose]
    a = math.prod(dims[:lose])
    b = math.prod(dims[lose + 1 :])
    rhos = _np.zeros(shape=(a * b, a * b), dtype=_np.complex128)
    for i in range(a * b):
        for j in range(i, a * b):
            i_i = e * b * (i // b) + (i % b)
            i_f = e * b * (i // b) + (i % b) + (e - 1) * b + 1
            j_i = e * b * (j // b) + (j % b)
            j_f = e * b * (j // b) + (j % b) + (e - 1) * b + 1
            rhos[i, j] = p[i_i:i_f:b, j_i:j_f:b].trace()
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos

def _partial_trace_dense(p, dims, keep):
    """Perform partial trace of a dense matrix."""
    if isinstance(keep, Integral):
        keep = (keep,)
    shp = p.shape
    if len(shp) == 1 or (len(shp) == 2 and (shp[0] == 1 or shp[1] == 1)):  # p = psi
        p = _np.asarray(p).reshape(dims)
        lose = ind_complement(keep, len(dims))
        p = _np.tensordot(p, p.conj(), (lose, lose))
        d = int(p.size**0.5)
        return p.reshape((d, d))
    else:
        p = _np.asarray(p).reshape((*dims, *dims))
        total_dims = len(dims)
        lose = ind_complement(keep, total_dims)
        lose2 = tuple(ind + total_dims for ind in lose)
        p = itrace(p, (lose, lose2))
    d = int(p.size**0.5)
    return p.reshape((d, d))

def ind_complement(inds, n):
    """Return the indices below ``n`` not contained in ``inds``."""
    return tuple(i for i in range(n) if i not in inds)

def itrace(a, axes=(0, 1)):
    """slower than np.enisum"""
    # Single index pair to trace out
    if isinstance(axes[0], Integral):
        return _np.trace(a, axis1=axes[0], axis2=axes[1])
    elif len(axes[0]) == 1:
        return _np.trace(a, axis1=axes[0][0], axis2=axes[1][0])

    # Multiple index pairs to trace out
    gone = set()
    for axis1, axis2 in zip(*axes):
        # Modify indices to adjust for traced out dimensions
        mod1 = sum(x < axis1 for x in gone)
        mod2 = sum(x < axis2 for x in gone)
        gone |= {axis1, axis2}
        a = _np.trace(a, axis1=axis1 - mod1, axis2=axis2 - mod2)
    return a

