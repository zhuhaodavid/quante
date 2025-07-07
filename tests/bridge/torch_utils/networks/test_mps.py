# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 16:47:15
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-04 12:52:47


import unittest
import torch as tc
import numpy as np
import quante as qt
import quante.bridge.torch_utils as qtc

try:
    import tenpy
    tenpy_available = True
except ImportError:
    tenpy_available = False


class TestTN(unittest.TestCase):
    def test_canonicalize(self):
        L = 4
        ψ = qtc.MPS.from_random(L=L, bond_dim=3)
        vec = ψ.to_vector()
        ψ1 = ψ.copy()
        ψ1.canonicalize_()
        vec1 = ψ1.to_vector()
        self.assertTrue(tc.allclose(vec1, vec))

    def test_add(self):
        # 验证加法
        ψ1 = qtc.MPS.from_random(L=4, bond_dim=3)
        ψ2 = qtc.MPS.from_random(L=4, bond_dim=3)
        ψ3 = ψ1 + ψ2
        
        vec1 = ψ1.to_vector()
        vec2 = ψ2.to_vector()
        vec3 = ψ3.to_vector()
        self.assertTrue(tc.allclose(vec3, vec2+vec1))

    def test_apply(self):
        # 随机生成一个态
        N = 5
        ψ = qtc.MPS.from_random(L=N, bond_dim=10)
        vec1 = ψ.to_vector().numpy()
        
        # 考虑这个算符
        I = np.eye(2)
        lM = np.random.randn(4,4)

        # 严格作用
        vec2 = qt.linalg.kron(I, lM, I, I) @ vec1

        # 两体门
        ψ1 = ψ.copy()
        ψ1.apply_gate_(1, lM, trunc_para=(10,1e-5,1e-5))
        vec2p = ψ1.to_vector().numpy()
        self.assertTrue(np.allclose(vec2p, vec2))
       
        
        ψ1 = ψ.copy()
        ψ1.apply_gate_(1,lM, svd_alg='eig', trunc_para=(10,1e-5,1e-5), normalize=False)
        vec2p = ψ1.to_vector()
        self.assertTrue(np.allclose(vec2p.cpu().numpy(), vec2))

    def test_to_matrix(self):
        vec = tc.randn(2**5, dtype=tc.complex128)
        mps = qtc.MPS.from_vector(vec)
        vec2 = mps.to_vector()
        self.assertTrue(tc.allclose(vec, vec2))

    def test_dm(self):
        N = 10

        # gen MPO
        M1 = qtc.MPO.from_random(N, bond_dim=5)

        # gen MPS
        ψ = qtc.MPS.from_random(N, bond_dim=10)

        
        ψ1 = ψ.copy()
        ψ1.apply_mpo_(M1)
        vec1 = ψ1.to_vector()
        vec2 = M1.to_matrix() @ ψ.to_vector()
        self.assertTrue(tc.allclose(vec1, vec2))
        
        ψ1 = ψ.copy()
        ψ1.apply_mpo_naive_(M1)
        vec1 = ψ1.to_vector()
        vec2 = M1.to_matrix() @ ψ.to_vector()
        self.assertTrue(tc.allclose(vec1, vec2))
    
    def test_swapsite(self):
        vec = tc.randn(*[2]*8, dtype=tc.float64)
        psi = qtc.MPS.from_vector(vec.reshape(-1))
        psi.swapsite_(1,4)
        self.assertTrue(tc.allclose(psi.to_vector(), vec.swapaxes(1,4).reshape(-1)))
    
    # if tenpy is installed, test the conversion to tenpy MPS
    @unittest.skipIf(not tenpy_available, "tenpy is not installed")
    def test_to_tenpy(self):
        import tenpy.linalg.np_conserved as npc
        L = 10
        psi = qtc.MPS.from_random(L, bond_dim=10)
        bm_vec = psi.to_vector().numpy()
        tpsi = psi.to_tenpy()
        res = tpsi.get_B(0, form='B', label_p='0')
        for i in range(1,L):
            res = npc.tensordot(res, tpsi.get_B(i, form='B', label_p=f'{i}'), axes=('vR', 'vL'))
        vec = res.to_ndarray().reshape(-1)

        self.assertAlmostEqual(np.linalg.norm(vec - bm_vec), 0)


