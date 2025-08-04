# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 20:35:58
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 19:55:17

import numpy as _np

from ...basis.basis_wrapped import _check_spin_number # type: ignore
from ..pauli import pauli_matrix

from typing import Optional, Union, Literal, overload
number = Union[int, float, complex]

__all__ = [
    "local_hamiltonian_spin_1D",
    "heisenberg_matrix",
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

## ================================
# heisenberg_matrix
## ===============================

def heisenberg_matrix(
    L, 
    j: Union[number, tuple[number, number, number]] = 1.0, 
    h: Union[number, tuple[number, number, number]] = 0.0, 
    pauli: bool = False, 
    S: Union[int, float, str] = 1/2, 
    cyclic:bool = False,
    Nup:Optional[int]=None, 
    kblock:Optional[int]=None, 
    pblock:Optional[int]=None, 
    zblock:Optional[int]=None,
    pzblock:Optional[int]=None,
    jmblock:Optional[Union[int, tuple[int, int]]]=None,
    sparse: bool = False
) -> _np.ndarray:
    """
    总是生成矩阵，而不是稀疏矩阵
    
    等价于：
    
    >>> ham = qt.generate.operas.heisenberg_operator(L, j, h, cyclic)
    >>> basis = qt.generate.basis.spin_basis(...)
    >>> mat = ham.to_matrix(basis, pauli, sparse=False)
    
    对于维数较小的矩阵比较高效
    """
    S = _check_spin_number(S)
    from ...basis import spin_basis
    basis = spin_basis(L, S=S, Nup=Nup, kblock=kblock, pblock=pblock, zblock=zblock, pzblock=pzblock, jmblock=jmblock)
    if sparse:
        from ...operas.spin import heisenberg_operator
        ham = heisenberg_operator(L, j, h, cyclic)
        return ham.to_matrix(basis, pauli=pauli, sparse=True)
    try:
        # 尝试使用针对heisenberg链的方法
        try:
            jx, jy, jz = j # type: ignore
        except TypeError:
            jx = jy = jz = j
        assert _np.isclose(jx, jy) and h == 0 and S == 0.5
        if pauli:
            jx = jx * 4
            jz = jz * 4
        return basis._heimat(jx, jz, cyclic) # type: ignore
    except:
        # 如果失败使用一般方法
        from ...operas.spin import heisenberg_operator
        ham = heisenberg_operator(L, j, h, cyclic)
        return ham.to_matrix(basis, pauli=pauli, sparse=False)

