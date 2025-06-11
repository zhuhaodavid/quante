# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-30 19:59:42
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-11 23:24:57

import unittest
import numpy as np
import quante as qt

class TestSYK(unittest.TestCase):
    def test_syk4_dirac(self):
        L = 10
        Jmat = np.random.randn(L,L,L,L) + 1j * np.random.randn(L,L,L,L)
        qt.generate.matrix.sky_anti_symetrize(Jmat, hermitize=False)
        basis = qt.generate.basis.quspin_fermion_basis(L=L, Nf=L//2)
        builder = qt.generate.operas.fermion.builder()
        for i1, i2, j1, j2 in np.ndindex((L,)*4):
            builder += "++--", [i1, i2, j1, j2], Jmat[i1, i2, j1, j2]
        ham = builder.build()
        mat1 = ham.to_matrix(basis)/(2*L)**(3/2)
        mat2 = qt.generate.matrix.syk4_dirac(L, L//2, J=Jmat)
        np.testing.assert_allclose(mat1, mat2, rtol=1e-5, atol=1e-8)

           
if __name__ == "__main__":
    unittest.main()

