# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 18:50:18
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-04 15:53:25

import numpy as _np
import scipy.sparse as _sparse
from typing import overload, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    'expect',
]
    
def observe_states(vecs:_np.ndarray, O:_np.ndarray) -> _np.ndarray:
    """
    计算 vecs 的观测值：
    
    Examples
    --------
    >>> mat = qla.rdmat(100, dtype=np.complex128)
    >>> eigs = qla.rdmat(100, dtype=np.complex128)
    >>> qla.observe_states(eigs, mat)
    """
    if  _np.issubdtype(vecs.dtype, _np.floating) and _np.issubdtype(O.dtype, _np.floating):
        from .nbfuc.expect_nb import observe_states_float
        return observe_states_float(vecs, O)
    else:
        from .nbfuc.expect_nb import observe_states_complex
        return observe_states_complex(vecs.astype(complex), O.astype(complex))


def real_if_close(val):
    if hasattr(val, "cpu") and hasattr(val, "numpy"):
        # val is a torch tensor
        return _np.real_if_close(val.cpu().numpy())
    else:
        try:
            return _np.real_if_close(val)
        except AttributeError:
            return val

def _matmat(mat, state):
    if hasattr(mat, 'is_complex'):
        matcomplex, statecomplex = mat.is_complex(), state.is_complex()
        if (matcomplex and statecomplex) or (not matcomplex and not statecomplex):
            return mat @ state
        if mat.is_complex():
            if str(mat.layout).startswith("torch.sparse_"):
                return mat @ state.to(mat.dtype)
            return mat.real @ state + 1j * (mat.imag @ state)
        elif state.is_complex():
            return mat @ state.real + 1j * (mat @ state.imag)
        else:
            raise ValueError("mat and state must be the same type")
    else:
        return mat @ state

def _isdiagonal(mat):
    return (
        isinstance(mat, (_sparse.dia_array, _sparse.dia_matrix)) and
        all(mat.offsets == [0])
    )

@overload
def expect(mat:_np.ndarray|_sparse.sparray, state:_np.ndarray, isdm=False) -> _np.ndarray: ...

@overload
def expect(mat:'torch.Tensor', state:'torch.Tensor', isdm=False) -> _np.ndarray: ...

def expect(mat, state, isdm=False) -> _np.ndarray:
    """计算期望值.

    mat 与 state 需要同为 Tensor 或者同不为 Tensor
    这个函数可以解决 Tensor 的 real @ complex 的报错
    对于对角矩阵，这个函数进行了优化
    对于多个态的期望，这个函数也进行了优化

    Parameters
    ----------
    mat : _np.ndarray | tc.Tensor | sps.sparray
        观测量
    state : _np.ndarray | tc.Tensor
        态矢量，第二个指标可以表示多个态的编号(如何 isdm=False)
    isdm : bool, optional
        是否是密度矩阵, by default False

    Returns
    -------
    _np.ndarray | tc.Tensor
        期望值
    """
    if isinstance(mat, list):
        return real_if_close([
            expect(m, state, isdm=isdm) for m in mat
        ])
    if not isdm:
        if state.ndim == 1 or (
            state.shape[1] == 1 or state.shape[0] == 1
        ):
            state = state.reshape(-1)
            if _isdiagonal(mat):
                matdiag = mat.diagonal()
                res = _matmat(state.conj(), (matdiag * state))
            else:
                res = _matmat(state.conj(), _matmat(mat, state))
            return real_if_close(res).item()
        elif state.ndim == 2:
            if _isdiagonal(mat):
                matdiag = mat.diagonal()
                res = _np.sum(state.conj() * (matdiag.reshape(-1, 1) * state), 
                              axis=0)
            elif mat.ndim == 1:
                res = (state.conj() * (mat * state)).sum(0)
            elif isinstance(mat, _np.ndarray):
                res = observe_states(state, mat)
            else:
                res = (state.conj() * _matmat(mat, state)).sum(0)
            return real_if_close(res)
        else:
            raise ValueError("state must be a 1D or 2D array for state vector")
    else:
        if state.ndim == 2:
            if _isdiagonal(mat):
                res = (mat.diagonal() * state.diagonal()).sum()
            else:
                res = _matmat(mat, state).trace()
            return real_if_close(res).item()
        elif state.ndim == 3:
            if _isdiagonal(mat):
                res = (mat.diagonal().reshape(-1,1,1) * state).trace(axis1=0, axis2=1)
            else:
                try:
                    res = _matmat(mat, state.swapaxes(0, 1))
                    if hasattr(res, 'is_complex'):
                        res = res.diagonal(offset=0, dim1=0, dim2=1).sum(-1)
                    else:
                        res = res.trace(axis1=0, axis2=1)
                except:
                    res = [
                        _matmat(mat, state[:,:,i]).trace().item() for i in range(state.shape[2])
                    ]
            return real_if_close(res)
        else:
            raise ValueError("state must be a 2D array for density matrix")

