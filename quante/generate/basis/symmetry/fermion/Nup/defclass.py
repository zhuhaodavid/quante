# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 20:22:38
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-06 12:58:39

from ...basis_class import FermionBitBasis
import scipy.sparse as sp
from typing import Union
import numpy as np
from .matrixele import single_sparse_matrix_element, diag_matrix_element

class FermionBitBasisNup(FermionBitBasis):
    def __init__(self, L: int, Nf: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L 的整数。
        """
        super().__init__(L)
        self.Nf = Nf
        self._validate_Nup()
        from .defbasis import construct_Nup_basis
        self.Ns, self.s_list = construct_Nup_basis(self.L, self.Nf)
        self.default_complex = False

    def _validate_Nup(self) -> None:
        assert self.Nf is not None and 0 <= self.Nf <= self.L and isinstance(self.Nf, int), "Nup should be an integer between 0 and N"
        
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        if self._isdiag(opnm, posn):
            return None, None, diag_matrix_element(opnm, posn, coef, self.L, self.Ns, self.s_list, ME_init.dtype)
        return single_sparse_matrix_element(opnm, posn, coef, self.L, self.Ns, self.s_list, row_init, col_init, ME_init)

    def __getitem__(self, index):
        return self.to_full_space(index)
    
    def to_full_space(self, index:int, from_Nup_space:bool = False):
        assert not from_Nup_space, "does not support from_Nup_space"
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        vec = np.zeros(1 << self.L, dtype=np.float64)
        vec[self.s_list[index]] = 1.
        return vec
    
    def projection_matrix(self):
        proj = np.eye(self.Ns, dtype=np.float64)
        from .defbasis import convert_project_to_full_space
        return convert_project_to_full_space(proj, self.L, self.s_list)
    
    def project(self, state):
        assert state.shape[0] == 1 << self.L, "state should be a vector of length 2**L"
        from .matrixele import project
        return project(state, self.Ns, self.s_list)
    
    def recover(self, state: np.ndarray) -> np.ndarray:
        dim1, dim2 = state.shape
        assert dim1 == self.Ns, f"state should be a matrix of shape ({self.Ns}, N)"
        vec = np.zeros((1 << self.L, dim2), dtype=state.dtype)
        vec[self.s_list, :] = state
        return vec
    
    @classmethod
    def print_dims(cls, L:int):
        import math
        for Nup in range(L+1):
            print(f"Nup = {Nup}: {math.comb(L, Nup)}")