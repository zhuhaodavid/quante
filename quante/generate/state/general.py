# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 16:07:40
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 18:01:04

import numpy as _np
import scipy.sparse as _sparse
from typing import Optional

def onehot(i, dim, dtype=complex, sparse=False):
    r"""Constructs a unit vector ket:
    
    Parameters
    ----------
    i : int
        Which index should the single non-zero, unit entry.
    dim : int
        Total size of hilbert space.

    Examples
    --------
    >>> qt.generate.state.basis_state(1,4)
    [[0.+0.j]
     [1.+0.j]
     [0.+0.j]
     [0.+0.j]]
    """
    if sparse:
        data = _np.array([1.0], dtype=dtype)
        row = _np.array([i])
        col = _np.array([0])
        return _sparse.coo_array((data, (row, col)), shape=(dim, 1)).tocsr()
    else:
        x = _np.zeros((dim, 1), dtype=dtype)
        x[i] = 1.0
        return x


def assemble(coef:_np.ndarray, posn:_np.ndarray, dim:int, sparse=False):
    coef = _np.asarray(coef)
    if sparse:
        cols = _np.zeros_like(posn)
        return _sparse.coo_array((coef, (posn, cols))).tocsr()
    else:
        x = _np.zeros((dim, 1), dtype=coef.dtype)
        for c, p in zip(coef, posn):
            x[p] = c
        return x


def random(dim: int, n: int = 1, dtype: type = complex, seed: Optional[int] = None, density = None) -> _np.ndarray:
    """generate a random vector and normalize it.

    Parameters
    ----------
    dim : int
        Dimension of the vector.
    n : int, optional
        The number of columns. Default is 1.
    dtype : type, optional
        Data type of the output, default is complex.
    seed : int, optional
        Seed for the random number generator, for reproducibility. Default is None.
    density : float, optional
        If specified, the density of the sparse matrix. If None, a dense matrix is generated.

    Returns
    -------
    _np.ndarray
        A normalized random vector of shape (dim, n) or a sparse matrix in CSR format if density is specified.
    """
    rng = _np.random.default_rng(seed)

    if density is not None:
        ket = _sparse.coo_array(_sparse.random(dim, n, format="coo", density=density))
        if isinstance(dtype, complex):
            ket.data = rng.standard_normal((ket.nnz,)) + 1j * rng.standard_normal((ket.nnz,))
        else:
            ket.data = rng.standard_normal((ket.nnz,))
        ket = ket.asformat("csr")
        ket[:] /= _np.sum(ket.conj() * ket, axis=0)**0.5 # type: ignore
        return ket

    if issubclass(dtype, complex):
        ket = rng.standard_normal(size=(dim, n)) + 1j * rng.standard_normal(size=(dim, n))
        ket[:] /= _np.linalg.norm(ket, axis=0)
        return ket
    else:
        ket = rng.standard_normal(size=(dim, n))
        ket[:] /= _np.linalg.norm(ket, axis=0)
        return ket

