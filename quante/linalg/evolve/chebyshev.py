# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 18:35:52
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 18:36:45

import numpy as _np
from scipy.special import jv

def chebyshev_evolve(mat:_np.ndarray, initstate:_np.ndarray, t:float, max_eng:float, min_eng:float, N:int) -> _np.ndarray:
    """ Chebyshev evolution of a state under a Hamiltonian, `exp( - 1j H t) |initstate>`.
    This function uses Chebyshev polynomial expansion to evolve the state under the Hamiltonian mat.
    
    # (max_eng - min_eng) * t ~ O(1) works better

    # todo: 自动计算误差，通过误差推出循环：

    Parameters
    ----------
    mat : np.ndarray
        the Hamiltonian matrix
    initstate : np.ndarray
        the initial state vector
    t : float
        the time parameter for evolution
    max_eng : float
        maximum energy eigenvalue of the Hamiltonian
    min_eng : float
        minimum energy eigenvalue of the Hamiltonian
    N : int
        the number of Chebyshev polynomials to use

    Returns
    -------
    np.ndarray
        the final state vector after evolution
    
    Notes
    -----
    This is a Chebyshev polynomial expansion method for time evolution.
    If you need to speed up, consider using gpu torch for mat @ xxx.
    For larger scale calculations, consider using petsc, related c++ program see
    https://github.com/Phyzch/Chebyshev_method
    When choosing parameters, let (max_eng - min_eng) * t ~ O(1)
      
    Example
    -------
    >>> L, t, N = 5, 1., 10
    >>> mat = qt.generate.matrix.heisenberg_matrix(L=L)
    >>> initstate = np.random.randn(mat.shape[0])
    >>> initstate /= np.linalg.norm(initstate)
    >>> max_eng, min_eng = np.max(np.linalg.eigvalsh(mat)), np.min(np.linalg.eigvalsh(mat))
    >>> finalstate = chebyshev_evolve(mat, initstate, t, max_eng, min_eng, N)
    >>> np.linalg.norm(finalstate - qt.linalg.expm(mat, c=-t*1j) @ initstate)
    np.float64(1.5768894460867202e-08)
    """
    a = (max_eng + min_eng) / 2
    b = (max_eng - min_eng) / 2
    tmp_state0 = initstate.copy()
    tmp_state1 = (mat @ initstate - a * initstate)/b  #!! main time
    finalstate_cheb = jv(0, b*t) * tmp_state0 * _np.exp(-1j*a*t)
    finalstate_cheb += 2 * (-1j) * jv(1, b*t) * tmp_state1 * _np.exp(-1j*a*t)
    for k in range(2,N):
        tmp_state0 = (2/b) * (mat @ tmp_state1 - a * tmp_state1) - tmp_state0  #!! main time
        tmp_state1, tmp_state0 = tmp_state0, tmp_state1
        finalstate_cheb += 2 * (-1j)**k * jv(k, b*t) * tmp_state1 * _np.exp(-1j*a*t)
    return finalstate_cheb
