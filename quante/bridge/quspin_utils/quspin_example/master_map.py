# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-24 12:28:43
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-10 21:55:13

from typing import Literal
import numpy as np
# from quspin.operators import hamiltonian
import scipy.sparse as sp

from ..quspin_extension_wrap import spin_basis_2d, hamiltonian, spin_basis_general
from ....generate.operas.spin import SpinOper
from ....generate.operas.super_oper import Lindbladian

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

def lindbladian(
    L:int, 
    ham:SpinOper, 
    jump_ops:list[SpinOper],
    basis:spin_super_basis,
    indx_order:Literal['stacked', 'snake']='stacked',
    flip:bool=False,
    check_symm:bool=False
):
    assert isinstance(basis, spin_basis_general), "basis must be an instance of spin_super_basis"
    liou = Lindbladian(L=L, ham=ham, jump_ops=jump_ops, indx_order=indx_order, flip=flip)

    if isinstance(basis, spin_super_basis_fast):
        assert flip == basis.flip, "flip must be the same as basis.flip"
        if basis.indx_order != indx_order:
            raise ValueError(f"basis.indx_order: {basis.indx_order} not the same as indx_order: {indx_order}")
        liou_mat = hamiltonian(liou, basis=basis, dtype=np.complex128, check_symm=check_symm).tocsr()
        return basis.realify(liou_mat)
    else:
        return hamiltonian(liou, basis=basis, dtype=np.complex128, check_symm=check_symm).tocsr()


