# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-05 19:06:21
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-30 14:56:21


from ...basis_class import SpinHalfBasis
import scipy.sparse as sp
import numpy as np
from typing import Union

class SpinHalfBasisNupKBlock(SpinHalfBasis):
    def __init__(self, L: int, Nup: int, kblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L 的整数。
        - kblock (Optional[int]): 动量块。从 0 到 L-1 的整数。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        self.kblock = kblock
        self._validate_kblock()
        from .defbasis import construct_Nup_kblock_basis
        self.Ns, self.s_list, R_list = construct_Nup_kblock_basis(self.L, self.Nup, self.kblock)
        self.other_params["R_list"] = R_list

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 0 <= self.Nup <= self.L and isinstance(self.Nup, int), "Nup should be an integer between 0 and N"
    
    def _validate_kblock(self) -> None:
        assert self.kblock is not None and 0 <= self.kblock <= self.L - 1 and isinstance(self.kblock, int), f"kblock should be an integer between 0 and {self.L - 1}"
        
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from ..kblock.matrixele import single_sparse_matrix_element
        return single_sparse_matrix_element(opnm, posn, coef, self.L, self.kblock, self.Ns, self.s_list, self.other_params["R_list"], row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        assert cyclic, "Only cyclic boundary condition is supported for kblock basis."
        from ..kblock.matrixele import heisenberg_matrix_element
        return heisenberg_matrix_element(self.L, self.Ns, self.kblock, jxy=jxy, jz=jz, s_list=self.s_list, R_list=self.other_params["R_list"])

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        from .defbasis import recover
        state = np.zeros((self.Ns, 1), dtype=np.complex128)
        state[index, 0] = 1
        return recover(state, self.L, self.kblock, self.s_list, self.other_params["R_list"])

    def projection_matrix(self):
        from .defbasis import recover
        state = np.eye(self.Ns, dtype=np.complex128)
        return recover(state, self.L, self.kblock, self.s_list, self.other_params["R_list"])
    
    def project(self, state):
        assert state.shape[0] == 1 << self.L, "state should be a vector of length 2**L"
        from ..kblock.matrixele import project
        return project(state, self.L, self.kblock, self.Ns, self.s_list, self.other_params["R_list"])
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .defbasis import recover
        return recover(np.complex128(state), self.L, self.kblock, self.s_list, self.other_params["R_list"])
    
    @classmethod
    def print_dims(cls, L:int):
        from .defbasis import construct_Nup_kblock_basis
        for Nup in range(L+1):
            for k in range(L):
                Ns, _, _ = construct_Nup_kblock_basis(L, Nup, k)
                print(f"Nup = {Nup}, k = {k}: {Ns}")
