# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-30 19:59:42
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:59:30

import unittest
import numpy as np
import quante as qt

class TestKIM(unittest.TestCase):
    def test_KIM_matrix(self):
        L = 10
        op = qt.generate.operas
        basis = qt.generate.basis.spin_basis(L)
        
        b = 1.
        ham = op.sum(b * op.x(i) for i in range(L))
        mat1 = ham.to_matrix(basis, pauli=True)
        mat1 = qt.linalg.expm(-1j*mat1)
        
        J = 1.
        h = np.random.rand(L)
        ham = op.sum(J*op.zz(i, (i+1)%L) + h[i]*op.z(i) for i in range(L))
        mat2 = ham.to_matrix(basis, pauli=True)
        mat2 = qt.linalg.expm(-1j*mat2)
        
        mat = mat1 @ mat2
        self.assertTrue(np.allclose(mat, qt.generate.matrix.KIM_matrix(b, J, h, L)))
                   
if __name__ == "__main__":
   unittest.main()
