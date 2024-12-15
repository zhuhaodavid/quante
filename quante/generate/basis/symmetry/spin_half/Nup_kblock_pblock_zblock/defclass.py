# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-04 23:37:56
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-19 12:28:25

from ...basis_class import SpinHalfBasis
import scipy.sparse as sp
from typing import Union
import numpy as np

class SpinHalfBasisNupKPZBlock(SpinHalfBasis):
    def __init__(self, L: int, Nup: int , kblock: int, pblock: int, zblock: int) -> None:
        """
        参数：
        - N (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。必须取 N//2。
        - kblock (Optional[int]): 动量块。从 0 到 N//2 的整数。
        - pblock (Optional[int]): 反演对称性块。-1 或 1。
        - zblock (Optional[int]): 旋转对称性块。-1 或 1。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        self.kblock = kblock
        self._validate_kblock()
        self.pblock = pblock
        self._validate_pblock()
        self.zblock = zblock
        self._validate_zblock()
        from .defbasis import construct_Nup_kblock_pblock_zblock_basis
        self.Ns, self.s_list, R_list, m_list, c_list = construct_Nup_kblock_pblock_zblock_basis(self.L, self.kblock, self.pblock, self.zblock)
        self.other_params["R_list"] = R_list
        self.other_params["m_list"] = m_list
        self.other_params["c_list"] = c_list
        self.default_complex = False
        self._double_Ns = 4

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 2*self.Nup == self.L and isinstance(self.Nup, int), "Nup must be N//2 when using Nup and kblock pblock zblock simutaniuously"

    def _validate_kblock(self) -> None:
        assert self.kblock is not None and 0 <= self.kblock <= self.L // 2 and isinstance(self.kblock, int), f"kblock should be an integer between 0 and {self.L // 2} when using kblock and pblock simutaniuously"

    def _validate_pblock(self) -> None:
        assert self.pblock in [-1, 1], "pblock should be -1 or 1"
    
    def _validate_zblock(self) -> None:
        assert self.zblock in [-1, 1], "zblock should be -1 or 1"

    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrixele import single_sparse_matrix_element
        return single_sparse_matrix_element(opnm, posn, coef, self.L, self.kblock, self.pblock, self.zblock, self.Ns, self.s_list, self.other_params["R_list"], self.other_params["m_list"], self.other_params["c_list"], row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        assert cyclic, "Only cyclic boundary condition is supported for kblock basis."
        from .matrixele import heisenberg_matrix_element
        return heisenberg_matrix_element(self.L, self.Ns, self.kblock, self.pblock, self.zblock, jxy=jxy, jz=jz, s_list=self.s_list, R_list=self.other_params["R_list"], m_list=self.other_params["m_list"], c_list=self.other_params["c_list"])
 
    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        from .defbasis import recover
        state = np.zeros((self.Ns, 1), dtype=np.float64)
        state[index, 0] = 1.
        return recover(state, self.L, self.kblock, self.pblock, self.zblock, self.s_list, self.other_params["R_list"], self.other_params["m_list"], self.other_params["c_list"], state.dtype)

    def projection_matrix(self):
        from .defbasis import recover
        state = np.eye(self.Ns, dtype=np.float64)
        return recover(state, self.L, self.kblock, self.pblock, self.zblock, self.s_list, self.other_params["R_list"], self.other_params["m_list"], self.other_params["c_list"], state.dtype)
    
    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        # return self.projection_matrix() @ state
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .defbasis import recover
        return recover(state, self.L, self.kblock, self.pblock, self.zblock, self.s_list, self.other_params["R_list"], self.other_params["m_list"], self.other_params["c_list"], state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .defbasis import construct_Nup_kblock_pblock_zblock_basis
        for k in range(L//2+1):
            for p in [-1,1]:
                for z in [-1,1]:
                    Ns, _, _, _, _ = construct_Nup_kblock_pblock_zblock_basis(L, k, p, z)
                    print(f"Nup = {L//2:<4}, k = {k:<4}, p = {p:<4}, z = {z:<4}:  {Ns:<4}")