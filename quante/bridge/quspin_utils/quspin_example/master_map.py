# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-24 12:28:43
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-10 22:00:17

from typing import Literal
import numpy as np
# from quspin.operators import hamiltonian
import scipy.sparse as sp

from ..quspin_extension_wrap import spin_basis_2d, hamiltonian, spin_basis_general, spin_super_basis, spin_super_basis_fast
from ....generate.operas.spin import SpinOper
from ....generate.operas.super_oper import Lindbladian

def real_if_close(mat):
    if isinstance(mat, np.ndarray):
        return np.real_if_close(mat)
    else:
        mat.data = np.real_if_close(mat.data)
        return mat



def lindbladian(
    L:int, 
    ham:SpinOper, 
    jump_ops:list[SpinOper],
    basis:spin_super_basis | spin_super_basis_fast,
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
    elif isinstance(basis, spin_super_basis):
        assert not flip
        if indx_order == 'stacked':
            raise ValueError(f"basis.indx_order: {basis.indx_order} not the same as indx_order: {indx_order}")
        liou_mat = hamiltonian(liou, basis=basis, dtype=np.complex128, check_symm=check_symm).tocsr()
        return basis.realify(liou_mat)
    else:
        return hamiltonian(liou, basis=basis, dtype=np.complex128, check_symm=check_symm).tocsr()


