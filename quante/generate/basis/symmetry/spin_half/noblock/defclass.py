# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-05 20:47:47
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-27 14:31:17

from ...basis_class import SpinHalfBasis
import numpy as np
from .matrixele import diag_matrix_element, single_sparse_matrix_element

class SpinHalfBasisNoBlock(SpinHalfBasis):
    def __init__(self, L: int) -> None:
        """
        参数:
        - L (int): 系统的大小。
        """
        super().__init__(L)
        self.s_list = range(1 << L)
        self.default_complex: bool = False
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        if self._isdiag(opnm, posn):
            return None, None, diag_matrix_element(opnm, posn, coef, self.L, self.Ns, ME_init.dtype)
        return single_sparse_matrix_element(opnm, posn, coef, self.L, self.Ns, row_init, col_init, ME_init)
        
    def _heimat(self, jxy, jz, cyclic=False):
        from .matrixele import heisenberg_matrix_element
        return heisenberg_matrix_element(self.L, jxy=jxy, jz=jz, cyclic=cyclic)
    
    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        vec = np.zeros(1 << self.L, dtype=np.float64)
        vec[index] = 1.
        return vec
    
    def projection_matrix(self):
        return np.eye(1 << self.L)
    
    def project(self, state):
        return state
    
    def recover(self, state):
        return state
    
    @classmethod
    def print_dims(cls, L:int):
        print(f"dim = {1 << L}")
