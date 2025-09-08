# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 20:35:58
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-08 17:24:01

import numpy as _np

from ..pauli import pauli_matrix
from ...solvable.heisenberg import heisenberg_matrix, ising_matrix, xxz_matrix

from typing import Union, Literal, overload
number = Union[int, float, complex]

__all__ = [
    "local_hamiltonian_spin_1D",
    "heisenberg_matrix",
    "ising_matrix",
    "xxz_matrix",
]

# ================================
# local_hamiltonian_spin_1D
# ================================

@overload
def local_hamiltonian_spin_1D(
    model_key:Literal["XX", "TFI"], 
    pauli:bool=True, 
    J = None,
    h = None,
) -> _np.ndarray:
    ...

@overload
def local_hamiltonian_spin_1D(
    model_key:Literal["XXZ"], 
    pauli:bool=True, 
    J = None,
    h = None,
    Δ = None,
) -> _np.ndarray:
    ...

@overload
def local_hamiltonian_spin_1D(
    model_key:Literal["TLFI"], 
    pauli:bool=True, 
    J = None,
    h = None,
    g = None,
) -> _np.ndarray:
    ...

def local_hamiltonian_spin_1D(
    model_key:Literal["XX", "XXZ", "TFI", "TLFI"], 
    pauli:bool=True, 
    **kwargs
) -> _np.ndarray:
    """
    Get local Hamiltonian term of 1D spin chain.
    
    Parameters
    ----------
    model_key : str
        The model key, can be one of the following:
        - "XXZ": `J` (xx+yy+ `Δ` zz)+ `h` z
        - "XX": `J` (xx+yy)+ `h` z
        - "TFI": `J` zz + `h` x
        - "TLFI": `J` zz + `g` x + `h` z
    pauli : bool, optional
        If True, return Pauli matrices, otherwise return spin matrices. Defaults to True.
    **kwargs : dict
        The parameters for the model, such as `J`, `Δ`, `h`, `g`.
        
    Returns
    -------
    ndarray
        The local Hamiltonian matrix.
    
    Examples
    --------
    >>> qt.generate.matrix.local_hamiltonian_spin_1D("XXZ", J = 0.1, Δ = 1,h = 0.1)
    >>> qt.generate.matrix.local_hamiltonian_spin_1D("XX", J = 0.1, h = 0.1)
    >>> qt.generate.matrix.local_hamiltonian_spin_1D("TFI", J = 0.1, h = 0.1)
    >>> qt.generate.matrix.local_hamiltonian_spin_1D("TLFI", J = 0.1, h = 0.1, g=1)
        
    Notes
    -----
    The filde terms need divide 2 in the input.

    params would convert inputs (such as x=1., y=2.) into a dictionary (such as {'x': 1., 'y': 2.})
    """
    model_dict = {
        "XX": "{J}*(XX+YY) + {h}*(ZI+IZ)",
        "XXZ": "{J}*(XX+YY+{Δ}*ZZ) + {h}/2*(ZI+IZ)",
        "TFI": "{J}*ZZ + {h}/2*(XI+IX)",
        "tTFI": "{J}*ZZ + {g}/2*(ZI+IZ) + {h}/2*(XI+IX)"
    }
    if model_key in model_dict:
        model_value = model_dict[model_key]
    else:
        raise NotImplementedError(f"'{model_key}' is NOT in model list.")
    if not pauli:
        model_value.replace("X", "x").replace("Y", "y").replace("Z", "z")
    model_value = model_value.format(**kwargs)
    return pauli_matrix(model_value)
