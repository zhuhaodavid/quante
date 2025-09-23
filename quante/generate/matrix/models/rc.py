# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-16 10:50:15
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-16 11:06:33

import numpy as np

def u1m2():
    """Return the gate of second moment of U(1) symmetry.

    .. math::
        \overline{U^* \otimes U \otimes U^* \otimes U} = \mathcal{T}
    
    Notes
    -----
    .. code-block:: text
        |        
        |     ┌─┴───┴─┐           1   2
        |    ┌┴┴───┴─┐|         ┌─┴───┴─┐ 
        |   ┌┴┴───┴─┐|┘   -->   |       |  
        |  ┌┴┴───┴─┐|┘          └─┬───┬─┘  
        |  |       |┘             3   4
        |  └─┬───┬─┘ 
    
    References
    ----------
    [1]. T. Rakovszky, F. Pollmann, and C. W. von Keyserlingk, Diffusive hydrodynamics of out-of-time-ordered correlators with charge conservation, https://doi.org/10.1103/PhysRevX.8.031058
    
    [2]. X. Turkeshi, P. Calabrese, and A. D. Luca, Quantum Mpemba Effect in Random Circuits, https://doi.org/10.1103/5d6p-8d1b
    """
    basis = np.eye(6)
    I = np.zeros((2, 3, 3, 36))
    I[0,0,0,:] = np.kron(basis[0], basis[0])
    I[0,0,1,:] = np.kron(basis[0], basis[1]) + np.kron(basis[1], basis[0])
    I[0,0,2,:] = np.kron(basis[1], basis[1])
    I[0,1,0,:] = np.kron(basis[0], basis[2]) + np.kron(basis[2], basis[0])
    I[0,1,1,:] = np.kron(basis[0], basis[5]) + np.kron(basis[1], basis[2]) + np.kron(basis[2], basis[1]) + np.kron(basis[5], basis[0])
    I[0,1,2,:] = np.kron(basis[1], basis[5]) + np.kron(basis[5], basis[1])
    I[0,2,0,:] = np.kron(basis[2], basis[2])
    I[0,2,1,:] = np.kron(basis[2], basis[5]) + np.kron(basis[5], basis[2])
    I[0,2,2,:] = np.kron(basis[5], basis[5])
    I[1,0,0,:] = np.kron(basis[0], basis[0])
    I[1,0,1,:] = np.kron(basis[0], basis[4]) + np.kron(basis[4], basis[0])
    I[1,0,2,:] = np.kron(basis[4], basis[4])
    I[1,1,0,:] = np.kron(basis[0], basis[3]) + np.kron(basis[3], basis[0])
    I[1,1,1,:] = np.kron(basis[0], basis[5]) + np.kron(basis[4], basis[3]) + np.kron(basis[3], basis[4]) + np.kron(basis[5], basis[0])
    I[1,1,2,:] = np.kron(basis[4], basis[5]) + np.kron(basis[5], basis[4])
    I[1,2,0,:] = np.kron(basis[3], basis[3])
    I[1,2,1,:] = np.kron(basis[3], basis[5]) + np.kron(basis[5], basis[3])
    I[1,2,2,:] = np.kron(basis[5], basis[5])

    res = np.zeros((36, 36), dtype=float)
    for s in [0,1]:
        for Q, dQ in enumerate([1,2,1]):
            if dQ == 1:
                res += I[s, Q, Q].reshape(-1, 1) @ I[s, Q, Q].reshape(1, -1)/2
            else:
                res += (1/(dQ**2 - 1)) * (
                    I[s, Q, Q].reshape(-1, 1) @ I[s, Q, Q].reshape(1, -1) -
                    (1/dQ) * (I[s, Q, Q].reshape(-1, 1) @ I[1-s, Q, Q].reshape(1, -1))
                )
        for Q1, dQ1 in enumerate([1,2,1]):
            for Q2, dQ2 in enumerate([1,2,1]):
                if Q1 != Q2:
                    res += (1/(dQ1 * dQ2)) * (I[s, Q1, Q2].reshape(-1, 1) @ I[s, Q1, Q2].reshape(1, -1))
    return res
 
