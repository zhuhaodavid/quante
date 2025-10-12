# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 18:08:18
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-12 14:57:17

# quspin_basis 是 quspin 库中的源码，这里提供一个简单的封装，使得用户可以更方便地使用 quspin_basis 中的 basis 类。

from quspin.basis import (
    spin_basis_general,
    spin_basis_1d,
    spinless_fermion_basis_1d,
    spinful_fermion_basis_1d,
    boson_basis_1d
)
import scipy.sparse as sp    
import numpy as np
from typing import Literal

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
    pauli:bool, 
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
    if pauli is True:
        _pauli = -1
    elif pauli is False:
        _pauli = 0
    else:
        raise ValueError(f"pauli should be either True or False, not {pauli}")
    blocks = {'a': a,
              "kblock": kblock, 
              "pblock": pblock, 
              "zblock": zblock, 
              "pzblock": pzblock, 
              "zAblock": zAblock, 
              "zBblock": zBblock}
    return spin_basis_1d(L=L, S=S, Nup=Nup, m=m, pauli=_pauli, **blocks)


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


def real_if_close(mat):
    if isinstance(mat, np.ndarray):
        return np.real_if_close(mat)
    else:
        mat.data = np.real_if_close(mat.data)
        return mat


class spin_super_basis_fast(spin_basis_2d):
    """spin super basis for the Liouvillian (super operator) in master equation.

    ctype = 'chain' or 'ladder'
    For chain:
    .. code-block:: text
        |    ---> x
        |   |    
        | y V  0     1      2               L-1
        |      o --- o  --- o  ---  ...  --- o
        |      o --- o  --- o  ---  ...  --- o
        |      L    L+1    L+2             2L-1
        |  
    
    For ladder:
    .. code-block:: text
        |    ---> y
        |   |
        | x V  0     2      4               2L-2
        |      o --- o  --- o  ---  ...  --- o
        |      o --- o  --- o  ---  ...  --- o
        |      1     3      5               2L-1

    Parameters
    ----------
    L : int
        length of the chain or ladder
    pauli : bool, optional
        whether to use pauli matrices
    Nup : int, optional
        number of up spins, by default None
    pblock : int, optional
        momentum sector, by default None
    zblock : int, optional
        spin inversion sector, by default None
    kblock : int, optional
        momentum sector, by default None
    """
    def __init__(
            self, L, pauli:bool, 
            indx_order:Literal['stacked', 'snake']='stacked',
            Nup=None, pblock=None, kblock=None, flip:bool=False,
            **blocks
        ):
        self.L = L
        self.indx_order = indx_order
        self.flip = flip

        self.block_names = list(blocks.keys())
        if pblock is not None:
            self.block_names.append('pblock')
        if kblock is not None:
            self.block_names.append('kblock')
        if Nup is not None:
            self.block_names.append('Nup')

        if indx_order == 'stacked':
            Lx, Ly = L, 2
            _blocks = {
                "pxblock": pblock,
                "kxblock": kblock,
            }
            if flip:
                _real_block = "pzyblock"
            else:
                _real_block = "pyblock"
        elif indx_order == 'snake':
            Lx, Ly = 2, L
            _blocks = {
                "pyblock": pblock,
                "kyblock": kblock,
            }
            if flip:
                _real_block = "pzxblock"
            else:
                _real_block = "pxblock"
        else:
            raise ValueError(f"indx_order should be 'stacked' or 'snake', but not {indx_order}")

        assert _real_block not in blocks, f"{_real_block} should not in blocks {blocks}"
        blocks.update(_blocks)
        self._real_block = _real_block
        self._superblock = blocks.copy()
        super().__init__(Lx=Lx, Ly=Ly, pauli=pauli, Nup=Nup, **blocks)
        self._sym_basis = None
        self._asym_basis = None
        self._P = None
    
    
    @property
    def sym_basis(self):
        blocks = self._superblock.copy()
        if self._sym_basis is None:
            blocks[self._real_block] = 0
            self._sym_basis = spin_basis_2d(
                Lx=self.Lx, Ly=self.Ly, pauli=self._pauli, 
                Nup=self._pcon_args['Nup'], **blocks
            )
        return self._sym_basis
    
    @property
    def asym_basis(self):
        blocks = self._superblock.copy()
        if self._asym_basis is None:
            blocks[self._real_block] = 1
            self._asym_basis = spin_basis_2d(
                Lx=self.Lx, Ly=self.Ly, pauli=self._pauli, 
                Nup=self._pcon_args['Nup'], **blocks
            )
        return self._asym_basis
    
    def project_matrix(self):
        if self._P is None:
            if self.Ns == 2**(2*self.L):
                P_sym = self.sym_basis.get_proj(np.complex128)
                P_antisym = 1j*self.asym_basis.get_proj(np.complex128)
            else:
                P0 = self.get_proj(np.complex128)
                P_sym = P0.conj().T @ self.sym_basis.get_proj(np.complex128)
                P_antisym = 1j*(P0.conj().T @ self.asym_basis.get_proj(np.complex128))
            
            self._P = sp.hstack([P_sym, P_antisym], format='csr')
        return self._P
    
    def realify(self, liou_mat):
        P = self.project_matrix()
        res = P.conj().T @ liou_mat @ P
        return real_if_close(res)