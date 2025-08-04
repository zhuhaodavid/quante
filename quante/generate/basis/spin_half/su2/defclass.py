# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-06 10:23:56
# @Last Modified by:   hzhu
# @Last Modified time: 2025-04-16 19:18:11

from typing import Union, Optional
from ...basis_class import SpinHalfBasis
import math
import numpy as _np
import scipy.sparse as _sp

class SpinHalfBasisSU2(SpinHalfBasis):
    def __init__(self, L:int, J:Union[float, int], m:Optional[Union[float, int]] = None) -> None:
        """
        分块对角化的结果为：
        
        J is even:
            ┌--------┬--------------------------------------------┐
            | J = 0  |                                            |
            | m = 0  |                                            |
            ├--------┼--------┬---------------┐                   |
            |        | J = 1  |               |                   |
            |        | m = -1 |               |                   |
            |        ├--------┼-------┐       |                   |
            |        |        | J = 1 |       |                   |
            |        |        | m = 0 |       |                   |
            |        |        └-------┼-------┤                   |
            |        |                | J = 1 |                   |
            |        |                | m = 1 |                   |
            |        └----------------┴-------┼-                  |
            |                                   ...               |
            |                                       ...           |
            |                                          -┼---------┤
            |                                           | J = L/2 |
            |                                           | m = L/2 |
            └-------------------------------------------┴---------┘
        
        J is odd:
            ┌--------┬--------┬------------------------------------┐
            | J = 1/2|        |                                    |
            | m =-1/2|        |                                    |
            ├--------┼--------┤                                    |
            |        | J = 1/2|                                    |
            |        | m = 1/2|                                    |
            ├--------┴--------┼--------┬-----------                |
            |                 | J = 3/2|                           |
            |                 | m =-3/2|                           |
            |                 ├--------┼--------┐                  |
            |                 |        | J = 3/2|                  |
            |                 |        | m = 1/2|                  |
            |                 |        └--------┼-                 |
            |                 |                  ...               |
            |                                        ...           |
            |                                           -┼---------┤
            |                                            | J = L/2 |
            |                                            | m = L/2 |
            └--------------------------------------------┴---------┘
        
        !! 相同 J 不同 m 的矩阵是相同的
        
        如下代码可以验证维数：
        
        Examples
        --------
        >>> L = 10
        >>> dim = 0
        >>> for J in np.arange(L/2, -1/2, -1):
        >>>     basis = SpinBasisSU2(L, J, 0)
        >>>     dim += basis.Ns * (2*J+1)
        >>> print(dim)
        """
        super().__init__(L)
        self._isvalid_J(L, J)
        self.J = J
        self._isvalid_m(J, m)
        self.m = m if m is not None else J
        self.Nup = int(L/2 + m)
        self.Ns = 1 if 2*J == L else math.comb(L, int(L/2-J)) - math.comb(L, int(L/2-J-1))
        from .defbasis import get_jlist
        self.s_list = get_jlist(L, J)
        from ..Nup.defbasis import construct_Nup_basis
        self.other_params["m_list"] = construct_Nup_basis(L, self.Nup)[1]
        self.default_complex = False
    
    def _isvalid_J(self, L, J):
        tmp = L/2 - J
        assert int(tmp) == tmp and 0 <= int(2*J) <= L, "J should be 0, 1, ..., L/2 或 1/2, 3/2, ..., L/2"
    
    def _isvalid_m(self, J, m):
        # m 为 -J, -J+1, ..., J-1, J 中的一个数
        tmp = J - m # tmp 应该为整数
        assert int(tmp) == tmp and -J <= m <= J, "m should be -J, -J+1, ..., J-1, J"
    
    def _sparse_matrix(self, op_list, hascomplex, savememory=None):
        from ..Nup.defclass import SpinHalfBasisNup
        basis_Nup = SpinHalfBasisNup(self.L, self.Nup)
        matrix_Nup = basis_Nup._sparse_matrix(op_list, hascomplex)
        proj = _sp.csr_array(self.projection_matrix())
        return proj.T @ matrix_Nup @ proj

    def __getitem__(self, index):
        return self.to_full_space(index, from_Nup_space=False)
        
    def to_full_space(self, index:int, from_Nup_space:bool = False):
        from .defbasis import get_ci
        if self.Nup == self.L:
            assert index == 0, "only one state"
        else:
            tmp = abs(math.comb(self.L, self.Nup) - math.comb(self.L, self.Nup - 1))
            assert index <= tmp, f"only {tmp} state"
        
        m_list = self.other_params["m_list"]
        if from_Nup_space:
            return get_ci(self.s_list[index], m_list, self.L).T
        else:
            res = _np.zeros(2**self.L, dtype=_np.float64)
            res[m_list] = get_ci(self.s_list[index], m_list, self.L)
            return res
    
    def projection_matrix(self):
        """
        Project from Nup = L/2 + m block to j block
        
        if m is not given, use m = j (space of the highest weight states)
        """
        from .defbasis import get_ci
        return get_ci(self.s_list.reshape(-1,1), self.other_params["m_list"], self.L).T

    def project(self, state):
        return self.projection_matrix().T @ state
    
    def recover(self, state):
        return self.projection_matrix() @ state
    
    @classmethod
    def print_dims(cls, L):
        """print the number of path for each j: LCN - LC(N-1)
        打印不同 (j, m) 对应空间的维数 （不同的 j 相同的 m 的空间维数相同）
        """
        import math
        out = "   J  |   num  |   dim   \n"
        out += "-----------------------\n"
        
        J = f"{L/2}"
        num = f"{L + 1}"
        dim = f"{1}"
        out += " "*(5-len(J)) + J + f" |   " + num + " "*(5-len(num)) + f"|  " + dim + "\n"
        
        for j in _np.arange(L/2-1,-1/2,-1):
            J = f"{j}"
            num = f"{int(2*j) + 1}"
            dim = math.comb(L, int(L/2-j)) - math.comb(L, int(L/2-j-1))
            out += " "*(5-len(J)) + J + f" |   " + f"{num}" + " "*(5-len(num)) + f"|  {dim}\n"
        out += "-----------------------\n"
        out += r"note: \sum num * dim = 2^L"
        print(out)
