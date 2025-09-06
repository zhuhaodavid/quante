# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-02 13:20:19
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-02 18:39:30

import numpy as np
from typing import Optional, Union, Literal
number = Union[int, float, complex]

from ..basis.basis_wrapped import _check_spin_number

## ================================
# heisenberg model
## ================================

def heisenberg_matrix(
    L, 
    j: number | tuple = 1.0, 
    *,
    hx: number | np.ndarray = 0.0,
    hy: number | np.ndarray = 0.0,
    hz: number | np.ndarray = 0.0, 
    jxy: number | np.ndarray = 0.0,
    jyx: number | np.ndarray = 0.0,
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
):
    r"""Generate Heisenberg model Hamiltonian matrix.

    ..math::
        H = \sum_{i} \left( J_x S_i^x S_{i+1}^x + J_y S_i^y S_{i+1}^y + J_z S_i^z S_{i+1}^z \right)
            + \sum_{i} \left( h_x S_i^x + h_y S_i^y + h_z S_i^z \right)
    where:
    >>> jx, jy, jz = j
    >>> hx, hy, hz = h

    by default, 
    >>> jx = jy = jz = 1.0
    >>> hx = hy = hz = 0.0

    This function is equivalent to:
    >>> ham = qt.generate.operas.heisenberg_operator(L, j, h, cyclic)
    >>> basis = qt.generate.basis.spin_basis(...)
    >>> mat = ham.to_matrix(basis, pauli, sparse=False)
    """
    S = _check_spin_number(S)
    from ..basis import spin_basis
    basis = spin_basis(
        L, S=S, Nup=Nup, kblock=kblock, pblock=pblock, 
        zblock=zblock, pzblock=pzblock, jmblock=jmblock
    )
    from ..operas.spin import heisenberg_operator
    ham = heisenberg_operator(L, j, hx=hx, hy=hy, hz=hz, jxy=jxy, jyx=jyx, cyclic=cyclic)
    return ham.to_matrix(basis, pauli=pauli, sparse=sparse)
    

def xxx_infinite_ground_energy(j, pauli=False):
    r"""Ground state energy for the infinite Heisenberg model.

    .. math::
        H = j \sum_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z)
    """
    return j * (0.5 - 2 * np.log(2))/2 * (4 if pauli else 1)

def xxx_finite_approx_ground_energy(L, j, pauli=False):
    r"""Approximate ground state energy for the *pbc* Heisenberg model.

    .. math::
        H = j \sum_{i=1}^{L-1} (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z)

    with error about o(1/ln^3(L))

    [1] Nickel, Bernie. "Scaling corrections to the ground state energy
    of the spin-½ isotropic anti-ferromagnetic Heisenberg chain." Journal of
    Physics Communications 1.5 (2017): 055021
    """
    from .free_fermion import spectrum as ff
    return j * ff.XXX_gdenergy_pbc_approx(L) * (4 if pauli else 1)

def xy_infinite_ground_energy(jx, jy, jxy, jyx, hz, pauli=False):
    r"""Ground state energy for the infinite XY model.

    .. math::
        H = \sum_i (J_x S_i^x S_{i+1}^x + J_y S_i^y S_{i+1}^y + 
                    J_{xy} S_i^x S_{i+1}^y + J_{yx} S_i^y S_{i+1}^x)
            + h_z \sum_i S_i^z
    """
    from .free_fermion.spectrum import _XY_gdenergy_inf
    return _XY_gdenergy_inf(jxx=jx, jyy=jy, jxy=jxy, jyx=jyx, hz=hz, pauli=pauli)

def xy_finite_ground_energy(L, jx, jy, jxy, jyx, hz, pauli=False):
    r"""Ground state energy for the finite XY model with *obc*.

    .. math:: 
    H = \sum_{i = 0}^{L - 2} 
            \left(
            j^{xx}_i s^{x}_{i} s^{x}_{i + 1} + 
            j^{yy}_i s^{y}_{i} s^{y}_{i + 1} + 
            j^{xy}_i s^{x}_{i} s^{y}_{i + 1} + 
            j^{yx}_i s^{y}_{i} s^{x}_{i + 1}
            \right)
        + \sum_{i = 0}^{L - 1} h^{z}_i s^{z}_{i}
    
    Notes
    -----
    - `jx, jy, jxy, jyx, hz` can be either numbers or lists
    - if `jxy == jyx == 0` and `jx == jy`, this function is efficient.
        otherwise, it involves an eigen decomposition of a dim-L matrix.
    """
    from .free_fermion.spectrum import _XY_omega
    omega = _XY_omega(L, jxx=jx, jyy=jy, jxy=jxy, jyx=jyx, hz=hz, pauli=pauli)
    return - np.sum(omega)
    
def xy_spectrum(L, jx, jy, jxy, jyx, hz, pauli=False):
    r"""Ground state energy for the finite XY model with *obc*.

    .. math:: 
    H = \sum_{i = 0}^{L - 2} 
            \left(
            j^{xx}_i s^{x}_{i} s^{x}_{i + 1} + 
            j^{yy}_i s^{y}_{i} s^{y}_{i + 1} + 
            j^{xy}_i s^{x}_{i} s^{y}_{i + 1} + 
            j^{yx}_i s^{y}_{i} s^{x}_{i + 1}
            \right)
        + \sum_{i = 0}^{L - 1} h^{z}_i s^{z}_{i}
    
    Notes
    -----
    - `jx, jy, jxy, jyx, hz` can be either numbers or lists
    - if `jxy == jyx == 0` and `jx == jy`, this function is efficient.
        otherwise, it involves an eigen decomposition of a dim-L matrix.
    - since it returns the full spectrum, `L` can not be too large (e.g. L < 20)
    """
    from .free_fermion.spectrum import XY_energies
    return XY_energies(L=L, jxx=jx, jyy=jy, jxy=jxy, jyx=jyx, hz=hz, pauli=pauli)

def xy_evolve(L, jx, jy, jxy, jyx, hz, init_state, tlist, 
              obs_name:Literal['particle_number'], 
              obs_para=None, pauli=False):
    r"""Evolve the obc finite XY model from a product state.

    .. math:: 
        H = \sum_{i = 0}^{L - 2} 
            \left(
            j^{xx}_i s^{x}_{i} s^{x}_{i + 1} + 
            j^{yy}_i s^{y}_{i} s^{y}_{i + 1} + 
            j^{xy}_i s^{x}_{i} s^{y}_{i + 1} + 
            j^{yx}_i s^{y}_{i} s^{x}_{i + 1}
            \right)
        + \sum_{i = 0}^{L - 1} h^{z}_i s^{z}_{i}
    
    This model can be mapped to free fermion.

    For more tools, use free fermion techniques in `quante.generate.solvable.gaussian_state`.
    """
    from ..operas.spin import heisenberg_operator
    model = heisenberg_operator(L=L, j=(jx,jy,0), hz=hz, jxy=jxy, jyx=jyx)
    model = model.jw_transfer(pauli=pauli)

    from .gaussian_state.pairing import PairingState
    
    h, coef_I = model.BdG_ham()
    state = PairingState.from_product_state(init_state)
    if obs_name == 'particle_number':
        for s in state.evolve(h, tlist):
            yield s.particle_number(obs_para)
    else:
        raise NotImplementedError(f"Observable {obs_name} not implemented yet.")

def xx_evolve(L, j, h, init_state, tlist, 
              obs_name:Literal['particle_number', 'entanglement', 'reduced_density_matrix'], 
              obs_para=None, pauli=False):
    r"""Evolve the obc finite XY model from a product state.

    .. math::
        H = \sum_{i = 1}^{L - 1} J_i\left( S_i^x S_{i+1}^x + S_i^y S_{i+1}^y  \right) + \sum_{i = 1}^{L} h^z_i S^z_i
    
    This model can be mapped to free fermion.

    For more tools, use free fermion techniques in `quante.generate.solvable.gaussian_state`.
    """
    from ..operas.spin import heisenberg_operator
    model = heisenberg_operator(L=L, j=(j,j,0), hz=h)
    model = model.jw_transfer(pauli=pauli)

    from .gaussian_state.slater import SlaterState
    
    h, coef_I = model.single_particle_ham()
    state = SlaterState.from_product_state(init_state)
    if obs_name == 'particle_number':
        for s in state.evolve(h, tlist):
            yield s.particle_number(obs_para)
    elif obs_name == 'entanglement':
        for s in state.evolve(h, tlist):
            yield s.entanglement(obs_para)
    elif obs_name == 'reduced_density_matrix':
        for s in state.evolve(h, tlist):
            yield s.reduced_density_matrix(*obs_para)
    else:
        raise NotImplementedError(f"Observable {obs_name} not implemented yet.")

