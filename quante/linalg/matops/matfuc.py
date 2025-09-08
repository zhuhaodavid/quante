# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:05:36
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-08 16:05:04

import numpy as _np
import scipy.linalg as _sla
import scipy.sparse as _sparse

def sort(arr):
    if not _np.iscomplexobj(arr):
        return _np.sort(arr)
    idx = _np.lexsort((arr.imag, _np.round(arr.real, 10)))
    return arr[idx]


def norm(v: _np.ndarray) -> float:
    """无论是 torch.Tensor, numpy.ndarray 还是 scipy.sparse.sparray，都可以计算范数"""
    try:
        return v.norm()
    except AttributeError:
        try: 
            return _sparse.linalg.norm(v)
        except TypeError:
            return _np.linalg.norm(v)


def _eigh_hermitian_matrix(A:_np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
    from ..decomp.eig_modified import eigh
    return eigh(A)


def exp(A:_np.ndarray, c: float | complex | int | None = None) -> _np.ndarray:
    """计算数组的指数
    
    对于大型数组，使用 Numba 加速计算
    
    Examples
    --------
    >>> L = 100000000
    >>> mat = np.random.randn(L) #+ 1j * np.random.randn(L)
    >>> with Timer('real 1'):
    >>>     mat1 = qt.linalg.exp(mat)
    >>> with Timer('real 2'):
    >>>     mat2 = np.exp(mat)
    >>> show(np.allclose(mat1, mat2))
    >>> L = 100000000
    >>> mat = np.random.randn(L) + 1j * np.random.randn(L)
    >>> with Timer('real 1'):
    >>>     mat1 = qt.linalg.exp(mat)
    >>> with Timer('real 2'):
    >>>     mat2 = np.exp(mat)
    >>> show(np.allclose(mat1, mat2))
    """
    if c is None:
        if _np.iscomplexobj(A):
            from .nbfuc.matfuc_nb import parallel_exp_complex
            return parallel_exp_complex(A)
        else:
            from .nbfuc.matfuc_nb import parallel_exp_real
            return parallel_exp_real(A)
    else:
        if not _np.iscomplexobj(A) and not _np.iscomplex(c):
            from .nbfuc.matfuc_nb import parallel_expmul_rr
            return parallel_expmul_rr(A, c)
        elif not _np.iscomplexobj(A) and _np.iscomplex(c):
            from .nbfuc.matfuc_nb import parallel_expmul_rc
            return parallel_expmul_rc(A, c)
        elif _np.iscomplexobj(A) and not _np.iscomplex(c):
            from .nbfuc.matfuc_nb import parallel_expmul_cr
            return parallel_expmul_cr(A, c)
        elif _np.iscomplexobj(A) and _np.iscomplex(c):
            from .nbfuc.matfuc_nb import parallel_expmul_cc
            return parallel_expmul_cc(A, c)


old_scipy = False
from scipy.linalg import expm as spexpm
if _np.linalg.norm(spexpm([[0.0, 0.1j], [0.9j, 0.0]]) - 
    _np.array([[0.95533649+0.j         ,0.        +0.09850674j], 
    [0.        +0.88656062j ,0.95533649+0.j        ]])) > 1e-6:
    old_scipy = True
# !! note: bug in scipt.expm in 1.15.2 on conda platform.
# !! singular matrix will lose accurate when cal inv in this eig method.

def expm(A:_np.ndarray, c: float | complex | None = None, isherm: bool | None = None, isdiag: bool = False) -> _np.ndarray:
    """Exponential Matrix, Hermitian matrix can be accelerated
    """
    if isdiag:
        diag_elements = _np.diag(A)
        new_diag_elements = exp(diag_elements, c)
        return _np.diag(new_diag_elements)

    if isherm is None:
        isherm = _np.allclose(A, A.conj().T)

    if isherm:
        eigenvalues, eigenstates = _eigh_hermitian_matrix(A)
        exp_eigval = exp(eigenvalues, c)
        if _np.iscomplexobj(eigenstates) and not _np.iscomplexobj(exp_eigval):
            exp_eigval = exp_eigval.astype(complex)
        return (eigenstates * exp_eigval) @ eigenstates.conj().transpose()
    else:
        if c is not None:
            A = A * c
        if old_scipy:
            try:
                import torch as tc
                return tc.linalg.matrix_exp(tc.tensor(A)).numpy()
            except ImportError:
                eigenvalues, eigenstates = _np.linalg.eig(A)
                exp_eigval = exp(eigenvalues, c)
                if _np.iscomplexobj(eigenstates) and not _np.iscomplexobj(exp_eigval):
                    exp_eigval = exp_eigval.astype(complex)
                return (eigenstates * exp_eigval) @ _np.linalg.inv(eigenstates)
        return spexpm(A)


def sqrtm(A:_np.ndarray, isherm: bool | None = None) -> _np.ndarray:
    """Square root Matrix, Hermitian matrix can be accelerated

    Args:
        A (np.ndarray): Matrix

    Returns:
        np.ndarray: square root Matrix
    """
    if isherm is None:
        isherm = _np.allclose(A, A.T.conj())
    if isherm:
        eigenvalues, eigenstates = _eigh_hermitian_matrix(A)
        if _np.iscomplexobj(eigenstates) and not _np.iscomplexobj(eigenvalues):
            eigenvalues = eigenvalues.astype(complex)
        return eigenstates * _np.sqrt(eigenvalues) @ eigenstates.T.conj()
    else:
        return _sla.sqrtm(A)


def logm(A:_np.ndarray, isherm = None) -> _np.ndarray:
    """Logarithm Matrix, Hermitain can be accelerated

    Args:
        A (np.ndarray): Matrix

    Returns:
        np.ndarray: Logarithm matrix
    """
    if isherm is None:
        isherm = _np.allclose(A, A.T.conj())
    if isherm:
        eigenvalues, eigenstates = _eigh_hermitian_matrix(A)
        return eigenstates * _np.log(eigenvalues.astype(complex)) @ eigenstates.T.conj()
    else:
        return _sla.logm(A)


def isherm(A: _np.ndarray, tol=1e-10) -> bool:
    """Check if a matrix is Hermitian.

    Args:
        A (np.ndarray): Matrix

    Returns:
        bool: True if Hermitian, False otherwise
    """
    if (type(A).__module__.startswith("torch") 
        and str(A.layout).startswith("torch.sparse_")):
        from ...bridge.torch_utils import tonp
        A = tonp(A)
    diff = A - A.T.conj()
    return abs(diff).max().item() < tol