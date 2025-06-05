# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2023-09-25 19:22:03
# @Last Modified by:   dzwang
# @Last Modified time: 2024-09-09 08:50:07
import copy
import numpy as _np
import numpy.linalg as _nla
import scipy.linalg as _sla
from ..generate.matrix import pauli_matrix
from ..linalg.eig_modified import eigh
from ..linalg.operations import expm, kron


__all__ = ["get_dissipative_uniform", "get_dissipative_boundary_driven"]


# =======================
#  Dissipative operator
# =======================

def _get_dissipative_pre_term(Lindblad:_np.ndarray) -> _np.ndarray:
    """
    Op_L:  Lindblad operator
    𝒟: disspative operator
    """
    Id = pauli_matrix("I")
    part_one = kron(Lindblad, Lindblad.conj())
    part_two = kron(Lindblad.T.conj()@Lindblad, Id)
    part_three = kron(Id, Lindblad.T@Lindblad.conj())
    return 2*part_one - (part_two + part_three)


def get_dissipative_uniform(L:int, c:float, gamma:float, dt:float) -> list[_np.ndarray]:
    from ..linalg.operations import expm
    Op_z = pauli_matrix("z")
    localD = _get_dissipative_pre_term(Op_z, c)
    exp_localD = expm(localD, dt*gamma)
    exp_localDs = [copy.deepcopy(exp_localD) for _ in range(L)]
    return exp_localDs


def get_dissipative_boundary_driven(gamma:float, mu:float) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Compute the dissipative operators for boundary-driven systems.
    
    Args:
        gamma (float): Dissipative strength parameter.
        mu (float): Chemical potential at the boundary.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Dissipative operators for the left and right boundaries.
    """
    Sp = pauli_matrix("p")  # Raising operator
    Sm = pauli_matrix("m")  # Lowering operator
    
    plus = _np.sqrt(gamma * (1 + mu))  # Plus term for boundary
    minus = _np.sqrt(gamma * (1 - mu))  # Minus term for boundary
    
    dissipative_left = (_get_dissipative_pre_term(plus * Sp) + 
                        _get_dissipative_pre_term(minus * Sm))
    dissipative_right = (_get_dissipative_pre_term(plus * Sm) + 
                         _get_dissipative_pre_term(minus * Sp))
    
    return dissipative_left, dissipative_right
