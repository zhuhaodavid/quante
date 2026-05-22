# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-03-10 19:18:49
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-23 02:45:22

import unittest
import torch as tc
import numpy as np
import quante as qt
import quante.bridge.torch_utils as qtc
from scipy.sparse.linalg import eigsh

class TestTN(unittest.TestCase):
    def test_dmrg_heisenberg(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L)
        H = ham.to_mpo(pauli=False).to_torch('cpu')
        eng1, vec1 = ℋ.dmrg(outputlevel=0)
        eng2, vec2 = ℋ.dmrg(Ms=[vec1], weight=[-eng1], outputlevel=0)
        eng3, vec3 = ℋ.dmrg(Ms=[vec1, vec2], weight=[-eng1, -eng2], outputlevel=0)
        mat = ℋ.to_matrix().numpy()
        eng = eigsh(mat, k=3, which='SA', return_eigenvectors=False)[::-1]
        print(eng)
        self.assertAlmostEqual(eng1.item(), eng[0])
        self.assertAlmostEqual(eng2.item(), eng[1])
        self.assertAlmostEqual(eng3.item(), eng[2])
        self.assertAlmostEqual(eng1.item(), vec1.measure(ℋ).item())
        self.assertAlmostEqual(eng2.item(), vec2.measure(ℋ).item())
        self.assertAlmostEqual(eng3.item(), vec3.measure(ℋ).item())
    
    def test_dmrg_ProjMPS(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L)
        H = ham.to_mpo(pauli=False).to_torch('cpu')
        PH = qtc.networks.projtt.ProjMPO(H, nsite=2)
        dmrg = qtc.DMRG(PH, outputlevel=0)
        eng1, vec1 = dmrg.run2()
        mat = ℋ.to_matrix().numpy()
        eng = qt.linalg.eigvalsh(mat, k=1)
        self.assertAlmostEqual(eng1.item(), eng[0])
        self.assertAlmostEqual(eng1.item(), vec1.measure(ℋ).item())
       



