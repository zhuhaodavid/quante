# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 18:08:18
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-30 20:45:59

# quspin_basis 是 quspin 库中的源码，这里提供一个简单的封装，使得用户可以更方便地使用 quspin_basis 中的 basis 类。

from quspin.basis import (
    spin_basis_general,
    spin_basis_1d,
    spinless_fermion_basis_1d,
    spinful_fermion_basis_1d,
    boson_basis_1d
)
import numpy as np

class spin_basis_2d(spin_basis_general):
    def __init__(self,
        Lx:int, Ly:int, pauli:bool, 
        Nup:int|None=None, 
        pxblock:int|None=None,
        pyblock:int|None=None,
        pzxblock:int|None=None,
        pzyblock:int|None=None,
        zblock:int|None=None,
        kxblock:int|None=None, 
        kyblock:int|None=None,
        **blocks
    ):
        """This function shows how to use quspin to generate 2d spin basis.

        The px, py, pzx, pzy, z, kx, ky blocks can be used simply here.

        Parameters
        ----------
        Lx : int
            the length of x direction
        Ly : int
            the length of y direction
        pauli : bool
            whether to use pauli matrices, True for pauli matrices, False for spin operators
        Nup : int | None, optional
            the number of spin up, by default None
        pxblock : int | None, optional
            the x parity block, by default None
        pyblock : int | None, optional
            the y parity block, by default None
        pzxblock : int | None, optional
            the x parity and z parity block, by default None
        pzyblock : int | None, optional
            the y parity and z parity block, by default None
        zblock : int | None, optional
            the z parity block, by default None
        kxblock : int | None, optional
            the x momentum block, by default None
        kyblock : int | None, optional
            the y momentum block, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
            
        if pauli is True:
            _pauli = -1
        elif pauli is False:
            _pauli = 0
        else:
            _pauli = pauli

        N_2d = Lx * Ly  # total number of sites
        s = np.arange(N_2d)  # sites [0,1,2,..]
        x = s % Lx  # x positions for sites
        y = s // Lx  # y positions for sites

        for i,j in blocks:
            assert len(i) == N_2d
            assert isinstance(j, int)
            
        if kxblock is not None and 'kxblock' not in blocks:
            T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
            blocks['kxblock'] = (T_x, kxblock)

        if kyblock is not None and 'kyblock' not in blocks:
            T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
            blocks['kyblock'] = (T_y, kyblock)

        if pxblock is not None and 'pxblock' not in blocks:
            P_x = (Lx - x - 1) + Lx * y
            blocks['pxblock'] = (P_x, pxblock)

        if pyblock is not None and 'pyblock' not in blocks:
            P_y = x + Lx * (Ly - y - 1)
            blocks['pyblock'] = (P_y, pyblock)

        if pzxblock is not None and 'pzxblock' not in blocks:
            PZ_x = - ((Lx - x - 1) + Lx * y + 1)
            blocks['pzxblock'] = (PZ_x, pzxblock)

        if pzyblock is not None and 'pzyblock' not in blocks:
            PZ_y = - (x + Lx * (Ly - y - 1) + 1)
            blocks['pzyblock'] = (PZ_y, pzyblock)

        if zblock is not None and 'zblock' not in blocks:
            Z = -(s + 1)
            blocks['zblock'] = (Z, zblock)

        super().__init__(N_2d, S='1/2', pauli=_pauli, Nup=Nup, **blocks)
        self.Lx = Lx
        self.Ly = Ly
 


def spin_basis(
    L:int, 
    pauli:int, 
    S:str="1/2", 
    m:float|None = None, 
    Nup:int|None=None, 
    a:int|None = None, 
    kblock:int|None=None, 
    pblock:int|None=None, 
    zblock:int|None=None, 
    pzblock:int|None=None,
    zAblock:int|None=None, 
    zBblock:int|None=None
) -> spin_basis_1d:
    """This function shows how to use quspin to generate spin basis.

    Parameters
    ----------
    L : int
        length of the chain
    S : str
        spin quantum number, e.g. "1/2", "1", "3/2", by default "1/2"
    m : float, optional
        total magnetization, by default None
    a : int | None, optional
        the unit cell of translation symmetry, by default None
    kblock : int | None, optional
        momentum block, the unit cell is defined by a, by default None
    pblock : int | None, optional
        parity block, reflection symmetry, by default None
    zblock : int | None, optional
        spin inversion symmetry, by default None
    pzblock : int | None, optional
        both parity and spin inversion symmetry, by default None
    zAblock : int | None, optional
        spin inversion symmetry on A sublattice, by default None
    zBblock : int | None, optional
        spin inversion symmetry on B sublattice, by default None
    """
    blocks = {'a': a,
              "kblock": kblock, 
              "pblock": pblock, 
              "zblock": zblock, 
              "pzblock": pzblock, 
              "zAblock": zAblock, 
              "zBblock": zBblock}
    return spin_basis_1d(L=L, S=S, Nup=Nup, m=m, pauli=pauli, **blocks)


def fermion_basis(
    L:int,
    Nf:int|list|None = None, 
    nf:float|None = None, 
    a:int|None = None, 
    kblock:int|None=None, 
    pblock:int|None=None
) -> spinless_fermion_basis_1d:
    """This function shows how to use quspin to generate spinless fermion basis.

    Parameters
    ----------
    L : int
        length of the chain
    Nf : int | list | None, optional
        the number of fermions, can be an integer or a list, the list represents different excitation numbers forming subspaces, by default None
    nf : float | None, optional
        the density of fermions, i.e., the ratio of the number of fermions to the total number, by default None
    a : int | None, optional
        the unit cell of translation symmetry, by default None
    kblock : int | None, optional
        momentum block, the unit cell is defined by a, by default None
    pblock : int | None, optional
        parity block, reflection symmetry, by default None
    """
    blocks = {'a': a, "kblock": kblock, "pblock": pblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return spinless_fermion_basis_1d(L=L, Nf=Nf, nf=nf, **blocks)

def spinful_fermion_basis(
    L:int, 
    Nf:tuple[int, list]|None = None, 
    nf: tuple[float]|None = None, 
    double_occupancy:bool=True, 
    a: int|None = None, 
    kblock:int|None = None, 
    pblock:int|None = None, 
    sblock: int|None = None, 
    psblock:int|None = None
) -> spinful_fermion_basis_1d:
    """This function shows how to use quspin to generate spinful fermion basis.

    Parameters
    ----------
    L : int
        length of the chain
    Nf : tuple[int, list] | None, optional
        the number of fermions, can be an integer or a list, the list represents different excitation numbers forming subspaces, the first number represents the number of spin-up fermions, and the second number represents the number of spin-down fermions, by default None
    nf : tuple[float] | None, optional
        the density of fermions, i.e., the ratio of the number of fermions to the total number, the first number represents the density of spin-up fermions, and the second number represents the density of spin-down fermions, by default None
    double_occupancy : bool, optional
        whether to allow double occupancy, i.e., the presence of both spin-up and spin-down fermions at the same site, by default True
    a : int | None, optional
        the unit cell of translation symmetry, by default None
    kblock : int | None, optional
        momentum block, the unit cell is defined by a, by default None
    pblock : int | None, optional
        parity block, reflection symmetry, by default None
    sblock : int | None, optional
        spin exchange symmetry, by default None
    psblock : int | None, optional
        combined spin exchange and reflection symmetry, by default None
    """
    blocks = {'a': a, "kblock": kblock, "pblock": pblock, "sblock": sblock, "psblock": psblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return spinful_fermion_basis_1d(L=L, Nf=Nf, nf=nf, double_occupancy=double_occupancy, **blocks)

def boson_basis(
    L:int, 
    Nb:int|list|None = None, 
    nb:float|None = None, 
    sps:int|None=2, 
    a:int|None = None, 
    kblock:int|None=None, 
    pblock:int|None=None, 
    cblock:int|None = None, 
    pcblock:int|None = None, 
    cAblock:int|None = None, 
    cBblock:int|None=None
) -> boson_basis_1d:
    """This function shows how to use quspin to generate boson basis.

    Parameters
    ----------
    L : int
        length of the chain
    Nb : int | list | None, optional
        the number of bosons, can be an integer or a list, the list represents different excitation numbers forming subspaces, by default None
    nb : float | None, optional
        the density of bosons, i.e., the ratio of the number of bosons to the total number, by default None
    sps: int | None, optional
        the number of states per site, or on-site Hilbert space dimension, by default None
    a : int | None, optional
        the unit cell of translation symmetry, by default None
    kblock : int | None, optional
        momentum block, the unit cell is defined by a, by default None
    pblock : int | None, optional
        parity block, reflection symmetry, by default None
    cblock : int | None, optional
        (hard-core) particle-hole symmetry, by default None
    pcblock : int | None, optional
        particle-hole and reflection symmetry, by default None
    cAblock : int | None, optional
        particle-hole symmetry on even sites, by default None
    cBblock : int | None, optional
        particle-hole symmetry on odd sites, by default None
    """
    blocks = {'a': a, "kblock": kblock, "pblock": pblock, "cblock":cblock, "pcblock":pcblock, "cAblock": cAblock, "cBblock": cBblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return boson_basis_1d(L=L, Nb=Nb, nb=nb, sps=sps, **blocks)
