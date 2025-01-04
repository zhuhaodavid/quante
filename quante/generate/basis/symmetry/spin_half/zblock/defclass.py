# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 22:04:59
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-19 12:29:33


from ...basis_class import SpinHalfBasis
import scipy.sparse as sp
from typing import Union
import numpy as np

class SpinHalfBasisZBlock(SpinHalfBasis):
    def __init__(self, L: int, zblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L 的整数。
        - zblock (Optional[int]): 旋转对称性块。-1 或 1。
        """
        super().__init__(L)
        self.zblock = zblock
        self._validate_zblock()
        from .defbasis import construct_zblock_basis
        self.Ns, self.s_list = construct_zblock_basis(self.L, self.zblock)
        self.default_complex = False

    def _validate_zblock(self) -> None:
        assert self.zblock in [-1, 1], "zblock should be -1 or 1"
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrixele import single_sparse_matrix_element
        return single_sparse_matrix_element(opnm, posn, coef, self.L, self.zblock, self.Ns, self.s_list, row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        from .matrixele import heisenberg_matrix_element
        return heisenberg_matrix_element(self.L, self.Ns, self.zblock, jxy=jxy, jz=jz, s_list=self.s_list, cyclic=cyclic)

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        from .defbasis import recover
        state = np.zeros((self.Ns,1), dtype=np.float64)
        state[index, 0] = 1
        return recover(state, self.L, self.zblock, self.s_list, state.dtype)

    def projection_matrix(self):
        from .defbasis import recover
        state = np.eye(self.Ns, dtype=np.float64)
        return recover(state, self.L, self.zblock, self.s_list, state.dtype)
        # from .zblock_basis import projective
        # return projective(self.s_list, self.Ns, self.L, self.zblock)

    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .defbasis import recover
        return recover(state, self.L, self.zblock, self.s_list, state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .defbasis import construct_zblock_basis
        for z in [-1, 1]:
            Ns, _ = construct_zblock_basis(L, z)
            print(f"z = {z:>3}: {Ns:>3}")
