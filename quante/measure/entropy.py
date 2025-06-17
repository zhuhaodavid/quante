# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:35:54
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 09:49:04

import numpy as _np
import math

def entanglement_spectrum(
    state: _np.ndarray, 
    left_number: int, 
    basis = None
) -> _np.ndarray:
    """The entanglement spectrum of a pure state.
    
    Parameters
    ----------
    state : ndarray
        The pure state, can be 1D or 2D array.
        - 1D array: the single state
        - 2D array: the multiple states with shape `(dim, num_states)`
    left_number : int
        The number of spins on the left side of the bipartition.
    basis : SpinBasis, optional
        The basis of the state, by default None.

    Returns
    -------
    ndarray | float
        The entanglement spectrum of the state.
        
    Examples
    --------
    >>> L = 10
    >>> ham = qt.generate.operas.heisenberg_operator(L=10)
    >>> basis = qt.generate.basis.spin_basis(L=L, Nup=5, kblock=1)
    >>> hammat = ham.to_matrix(basis)
    >>> val, vec = qt.linalg.eigh(hammat, k=1)
    >>> print(qt.measure.entanglement_spectrum(vec, L=L, left_number=L//2, basis=basis))
    [0.70710678 0.70710678 0.         0.         0.         0.         0.         0.        ]
    """
    if basis is not None:
        if state.ndim == 1:
            state = state.reshape(-1,1)
        state = basis.recover(state)
        L = basis.L
    else:
        D = state.shape[0] if state.ndim == 1 else state.shape[1]
        L = int(math.log2(D))
        assert D == 1 << L, "The dimension of the state is not 2^L"
    matrix = state.T.reshape(-1,1<<left_number,1<<L-left_number)
    return _np.linalg.svd(matrix, compute_uv=False) # type: ignore

def entanglement_entropy(
    states: _np.ndarray, 
    left_number: int, 
    basis = None
) -> _np.ndarray | float:
    """The entanglement entropy of pure states.
    
    Parameters
    ----------
    states : ndarray
        The pure states, can be 1D or 2D array.
        - 1D array: the single state
        - 2D array: the multiple states with shape `(dim, num_states)`
    L : int
        The number of spins.
    left_number : int
        The number of spins on the left side of the bipartition.
    basis : SpinBasis, optional
        The basis of the state, by default None.

    Returns
    -------
    ndarray | float
        The entanglement entropy of the states.
        
    Examples
    --------
    >>> L = 10
    >>> ham = qt.generate.operas.heisenberg_operator(L=10)
    >>> basis = qt.generate.basis.spin_basis(L=L, Nup=5, kblock=1)
    >>> hammat = ham.to_matrix(basis)
    >>> val, vec = qt.linalg.eigh(hammat, k=1)
    >>> print(qt.measure.entanglement_entropy(vec, L=L, left_number=L//2, basis=basis))
    0.6931471805599453
    """
    ee = entanglement_spectrum(states, left_number, basis)
    # ee.shape = (items, spectrum)
    ee = _np.where(ee > 0, ee, 1)  # Replace zeros with 1 to make log(1)=0
    res = (-2) * _np.sum(ee**2 * _np.log(ee), axis=1)
    if res.size == 1:
        return res[0]
    return res

def entropy(a, rank=None, base=_np.e) -> _np.float64:
    """计算 von Neumann 熵.
    
    如果 `a` 是密度矩阵，那么计算：
    
    .. math::
        -\\operatorname{tr} a \\log a
        
    通过 `rank` 可以指定计算的本征值个数。
    
    如果 `a` 是本征值，那么计算：
    
    .. math::
        -\\sum_{i=1}^n a_i \\log a_i
    
    其中 :math:`n` 是 `a` 的维度。

    Examples
    --------
    >>> L = 6
    >>> mat = ed.rdmat_rho(2**L, sparse=True, density=0.5)
    >>> etp = qla.entropy(mat)
    >>> print(etp)
    >>> 
    >>> vals = qla.eigvals(mat)
    >>> etp = qla.entropy(vals)
    >>> print(etp)
    
    可以直接输入本征值
    """
    if _np.ndim(a) == 1:
        evals = a
    elif _np.ndim(a) == 2 and (a.shape[0] == 1 or a.shape[1] == 1):
        evals = a.flatten()
    else:
        from ..linalg.decomp import eigvals
        if rank is None:
            evals = eigvals(a)
        else:  # know that not all eigenvalues needed
            evals = eigvals(a, k=rank, which="LM")

    evals = evals[evals > 0.0]
    return _np.real_if_close([_np.sum(-evals * _np.log2(evals)) / _np.log2(base)])[0]


def entropy_page(Dim_sub:int, Dim_tot:int) -> float:
    """计算 Page 熵。
    
    Page 熵指 Hilbert 空间中一个随机态的熵。
    
    Parameters
    ----------
    Dim_sub : int
        子空间维数
    Dim_tot : int
        空间维数

    Returns
    -------
    float
        Page 熵
    
    Examples
    --------
    计算 12 个自旋，二分为 6 个自旋的 Page 熵：
    
    >>> import quante.measure as qm
    >>> L = 12
    >>> vals = qm.entropy_page(2**(L//2), 2**L)
    >>> vals
    3.6590254932605575
    
    随机这样一个态，计算它的二分纠缠熵：
    
    >>> import quante as qt
    >>> vec = qt.generate.state.random(dim=2**12)
    >>> rho = qt.linalg.partial_trace(vec, [2]*L, range(L//2))
    >>> qt.measure.entropy(rho)
    3.6520327465312925
    
    可以看到是非常接近的。
    
    References
    ----------
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.71.1291
    """
    # * Ensure m is the smaller subsystem
    if Dim_sub <= Dim_tot//Dim_sub:
        m, n = Dim_sub, Dim_tot//Dim_sub
    else:
        m, n = Dim_tot//Dim_sub, Dim_sub
        
    s = 0.
    for k in range(n+1, Dim_tot+1):
        s += 1 / k
    return s - (m-1)/(2*n)

