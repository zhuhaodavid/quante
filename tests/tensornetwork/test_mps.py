# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 16:47:15
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 15:54:47


import unittest
import numpy as np
import quante as qt
import quante.tensornetwork as qtn

try:
    import tenpy
    tenpy_available = True
except ImportError:
    tenpy_available = False


class TestTN(unittest.TestCase):
    def test_canonicalize(self):
        L = 4
        ψ = qtn.MPS.from_random(L=L, bond_dim=3)
        vec = ψ.to_vector()
        ψ1 = ψ.copy()
        ψ1.canonicalize_()
        vec1 = ψ1.to_vector()
        self.assertTrue(np.allclose(vec1, vec))

    def test_add(self):
        # 验证加法
        ψ1 = qtn.MPS.from_random(L=4, bond_dim=3)
        ψ2 = qtn.MPS.from_random(L=4, bond_dim=3)
        ψ3 = ψ1 + ψ2
        
        vec1 = ψ1.to_vector()
        vec2 = ψ2.to_vector()
        vec3 = ψ3.to_vector()
        self.assertTrue(np.allclose(vec3, vec2+vec1))

    def test_inner_operator(self):
        ψ1 = qtn.MPS.from_random(L=4, bond_dim=3)
        ψ2 = qtn.MPS.from_random(L=4, bond_dim=3)

        self.assertTrue(np.allclose(ψ1 | ψ2, ψ1.inner(ψ2)))
        self.assertTrue(np.allclose(ψ1 | ψ1, ψ1.norm() ** 2))

    def test_mpo_matrix_element_operator(self):
        ψ1 = qtn.MPS.from_random(L=4, bond_dim=3)
        ψ2 = qtn.MPS.from_random(L=4, bond_dim=3)
        H = qtn.MPO.from_random(L=4, bond_dim=3)

        self.assertTrue(np.allclose(ψ1 | H | ψ2, H.mele(ψ1, ψ2)))
        self.assertTrue(np.allclose(ψ1 | H | ψ2, ψ1.to_vector().conj() @ H.to_matrix() @ ψ2.to_vector()))

    def test_bramps_repr(self):
        ψ = qtn.MPS.from_random(L=4, bond_dim=3)
        H = qtn.MPO.from_random(L=4, bond_dim=3)

        text = repr(ψ | H)
        self.assertIn("<bra|mpo", text)
        self.assertIn("open ket legs", text)

    def test_apply(self):
        # 随机生成一个态
        N = 5
        ψ = qtn.MPS.from_random(L=N, bond_dim=10)
        vec1 = ψ.to_vector()
        
        # 考虑这个算符
        I = np.eye(2)
        lM = np.random.randn(4,4)

        # 严格作用
        vec2 = qt.linalg.kron(I, lM, I, I) @ vec1

        # 两体门
        ψ1 = ψ.copy()
        ψ1.apply_gate_(1, lM, trunc_para=(10,1e-5,1e-5))
        vec2p = ψ1.to_vector()
        self.assertTrue(np.allclose(vec2p, vec2))
       
        
        ψ1 = ψ.copy()
        ψ1.apply_gate_(1,lM, svd_alg='eig', trunc_para=(10,1e-5,1e-5), normalize=False)
        vec2p = ψ1.to_vector()
        self.assertTrue(np.allclose(vec2p, vec2))

    def test_to_matrix(self):
        vec = (np.random.randn(2**5) + 1j * np.random.randn(2**5)).astype(np.complex128)
        mps = qtn.MPS.from_vector(vec)
        vec2 = mps.to_vector()
        self.assertTrue(np.allclose(vec, vec2))

    def test_dm(self):
        N = 10

        # gen MPO
        M1 = qtn.MPO.from_random(N, bond_dim=5)

        # gen MPS
        ψ = qtn.MPS.from_random(N, bond_dim=10)

        
        ψ1 = ψ.copy()
        ψ1.apply_mpo_(M1)
        vec1 = ψ1.to_vector()
        vec2 = M1.to_matrix() @ ψ.to_vector()
        self.assertTrue(np.allclose(vec1, vec2))
        
        ψ1 = ψ.copy()
        ψ1.apply_mpo_naive_(M1)
        vec1 = ψ1.to_vector()
        vec2 = M1.to_matrix() @ ψ.to_vector()
        self.assertTrue(np.allclose(vec1, vec2))

        for direction in ("right", "left"):
            ψ1 = ψ.copy()
            ψ1.apply_mpo_zip_up(M1, direction=direction)
            vec1 = ψ1.to_vector()
            self.assertTrue(np.allclose(vec1, vec2))
    
    def test_swapsite(self):
        vec = np.random.randn(*[2]*8).astype(np.float64)
        psi = qtn.MPS.from_vector(vec.reshape(-1))
        psi.swapsite_(1,4)
        self.assertTrue(np.allclose(psi.to_vector(), vec.swapaxes(1,4).reshape(-1)))
    
    # if tenpy is installed, test the conversion to tenpy MPS
    @unittest.skipIf(not tenpy_available, "tenpy is not installed")
    def test_to_tenpy(self):
        import tenpy.linalg.np_conserved as npc
        L = 10
        psi = qtn.MPS.from_random(L, bond_dim=10)
        bm_vec = psi.to_vector()
        tpsi = psi.to_tenpy()
        res = tpsi.get_B(0, form='B', label_p='0')
        for i in range(1,L):
            res = npc.tensordot(res, tpsi.get_B(i, form='B', label_p=f'{i}'), axes=('vR', 'vL'))
        vec = res.to_ndarray().reshape(-1)

        self.assertAlmostEqual(np.linalg.norm(vec - bm_vec), 0)



