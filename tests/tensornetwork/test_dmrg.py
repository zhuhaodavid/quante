# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-03-10 19:18:49
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-13 20:41:32

import unittest
import numpy as np
import quante as qt
import quante.tensornetwork as qtn
from scipy.sparse.linalg import eigsh

class TestTN(unittest.TestCase):
    def test_dmrg_heisenberg(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L)
        H = qtn.MPO.from_oper(ham, pauli=False)
        eng1, vec1 = H.dmrg(outputlevel=0)
        eng2, vec2 = H.dmrg(Ms=[vec1], weight=[-eng1], outputlevel=0)
        eng3, vec3 = H.dmrg(Ms=[vec1, vec2], weight=[-eng1, -eng2], outputlevel=0)
        mat = H.to_matrix()
        eng = eigsh(mat, k=3, which='SA', return_eigenvectors=False)[::-1]
        print(eng)
        self.assertAlmostEqual(eng1, eng[0])
        self.assertAlmostEqual(eng2, eng[1])
        self.assertAlmostEqual(eng3, eng[2])
        self.assertAlmostEqual(eng1, vec1.measure(H))
        self.assertAlmostEqual(eng2, vec2.measure(H))
        self.assertAlmostEqual(eng3, vec3.measure(H))
    
    def test_dmrg_ProjMPS(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L)
        H = qtn.MPO.from_oper(ham, pauli=False)
        PH = qtn.projtt.ProjMPO(H, nsite=2)
        dmrg = qtn.DMRG(PH, outputlevel=0)
        eng1, vec1 = dmrg.run2()
        mat = H.to_matrix()
        eng = qt.linalg.eigvalsh(mat, k=1)
        self.assertAlmostEqual(eng1, eng[0])
        self.assertAlmostEqual(eng1, vec1.measure(H))
       




