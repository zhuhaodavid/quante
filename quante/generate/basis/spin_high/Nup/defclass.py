# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-12 22:03:20
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-15 20:28:47

from ...basis_class import SpinHighBasis
from typing import Union
import numpy as np

class SpinHighBasisNup(SpinHighBasis):
    def __init__(self, L: int, S:Union[int,float], Nup:int) -> None:
        super().__init__(L, S)
        self.Nup = Nup
        self._validate_Nup()
        self.Ns = self._get_Ns(self.L, self.S, self.Nup)
        from .defbasis import generate_all_states2
        self.s_list = generate_all_states2(self.L, self.Ns, self.local_dim, self.Nup)
        self.default_complex = False
    
    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 0 <= self.Nup <= self.L * self.local_dim and isinstance(self.Nup, int), "Nup should be an integer between 0 and N"
    
    def _Op(self, opnm: str, posn: int, coef: Union[int, float, complex], row_init: np.ndarray, col_init: np.ndarray, ME_init: np.ndarray) -> np.ndarray:
        from .matrixele import single_sparse_matrix_element
        return single_sparse_matrix_element(opnm, posn, coef, self.L, self.S, self.Ns, self.s_list, row_init, col_init, ME_init)

    @classmethod
    def _get_Ns(cls, L: int, S: Union[int, float], Nup: int) -> int:
        d = int(2*S)
        n = L
        m = Nup
        
        # 创建 dp 数组，初始化为 0
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # 边界条件：没有数时，和为 0 的情况有 1 种
        dp[0][0] = 1
        
        # 动态规划求解
        for i in range(1, n + 1):
            for j in range(m + 1):
                dp[i][j] = sum(dp[i-1][j-k] for k in range(min(d, j) + 1))
        
        # 返回 dp[n][m]，即为答案
        return dp[n][m]
    
    @classmethod
    def print_dims(cls, L:int, S:Union[int,float]):
        for Nup in range(L*int(2*S)+1):
            print(f"Nup = {Nup}: {cls._get_Ns(L, S, Nup)}")

