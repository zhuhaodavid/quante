# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 22:31:07
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-19 12:28:39

from ...basis_class import SpinHalfBasis
import scipy.sparse as sp
from typing import Union
import numpy as np

class SpinHalfBasisNupPZBlock(SpinHalfBasis):
    def __init__(self, L: int, Nup: int, pzblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L//2 的整数。
        - pzblock (Optional[int]): 旋转对称性块。-1 或 1。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        self.pzblock = pzblock
        self._validate_pzblock()
        from .defbasis import construct_Nup_pzblock_basis
        self.Ns, self.s_list = construct_Nup_pzblock_basis(self.L, self.Nup, self.pzblock)

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 0 <= self.Nup <= self.L and isinstance(self.Nup, int), "Nup should be an integer between 0 and N"

    def _validate_pzblock(self) -> None:
        assert self.pzblock in [-1, 1], "pzblock should be -1 or 1"
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from ..pzblock.matrixele import single_sparse_matrix_element
        return single_sparse_matrix_element(opnm, posn, coef, self.L, self.pzblock, self.Ns, self.s_list, row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        from ..pzblock.matrixele import heisenberg_matrix_element
        return heisenberg_matrix_element(self.L, self.Ns, self.pzblock, jxy=jxy, jz=jz, s_list=self.s_list, cyclic=cyclic)

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        from .defbasis import recover
        state = np.zeros((self.Ns,1), dtype=np.float64)
        state[index, 0] = 1
        return recover(state, self.L, self.pzblock, self.s_list, state.dtype)

    def projection_matrix(self):
        from .defbasis import recover
        state = np.eye(self.Ns, dtype=np.float64)
        return recover(state, self.L, self.pzblock, self.s_list, state.dtype)

    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .defbasis import recover
        return recover(state, self.L, self.pzblock, self.s_list, state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .defbasis import construct_Nup_pzblock_basis
        for Nup in range(L//2+1):
            for pz in [-1, 1]:
                Ns, _ = construct_Nup_pzblock_basis(L, Nup, pz)
                print(f"Nup = {Nup:>3}, pz = {pz:>3}: {Ns:>3}")
