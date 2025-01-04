# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-05 20:47:47
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-19 12:29:46

import numpy as np
from typing import Union
from ...basis_class import SpinHighBasis

class SpinHighBasisNoBlock(SpinHighBasis):
    def __init__(self, L: int, S:Union[int,float] = 0.5) -> None:
        """
        参数:
        - L (int): 系统的大小。
        """
        super().__init__(L, S)
        self.s_list = range(int(2*S+1) ** L)
        
    def _Op(self, opnm: str, posn: int, coef: Union[int, float, complex], row_init: np.ndarray, col_init: np.ndarray, ME_init: np.ndarray) -> np.ndarray:
        from .matrixele import single_sparse_matrix_element
        return single_sparse_matrix_element(opnm, posn, coef, self.L, self.S, self.Ns, row_init, col_init, ME_init)
    