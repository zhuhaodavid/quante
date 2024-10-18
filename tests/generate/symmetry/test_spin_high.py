# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-13 13:46:39
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-30 13:05:36


import unittest
import numpy as np
import quante as qt
import sys
sys.dont_write_bytecode = True

import os
# 增加当前路径的上级路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
great_grandparent_dir = os.path.dirname(grandparent_dir)
sys.path.append(great_grandparent_dir)


class TestSpinHigh(unittest.TestCase):
    
    def test_noblock(self):
        L = 6
        j = np.random.randn(3)
        h = np.random.randn(3)
        ham = qt.generate.operas.heisenberg_operator(L, j=j, h=h)
        ham = ham.expandxy()
        
        for S in np.arange(0.5, 2, 0.5):
            basis = qt.generate.basis.spin_basis(L=L, S=S)
            mat1 = ham.to_matrix(basis)
            mat2 = qt.generate.matrix.parallel_build_matrix(*ham.split_data(), L, S)
            self.assertTrue(np.allclose(mat1, mat2))

    # def test_Nup(self):
    #     L = 10
    #     ham = qt.generate.operas.heisenberg_operator(L=L)
    #     ham = ham.expandxy()

    #     from quante.generate.symmetry.spin_high.Nup.defclass import SpinHighBasisNup

    #     S = 0.5
    #     Nup = 9

    #     basis = SpinHighBasisNup(L, S, Nup)
    #     mat1 = basis._matrix(*ham._convert_to_typed_list(), dtype=np.float64)
        
    #     basis = qt.generate.basis.spin_basis(L=L, S=S, Nup=Nup)
    #     mat2 = ham.to_matrix(basis)
        
    #     self.assertTrue(np.allclose(mat1, mat2))

if __name__ == "__main__":
    unittest.main()