# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 16:49:50
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-18 18:16:16

import unittest
import torch as tc
import numpy as np
import quante as qt
import quante.bridge.torch_utils as qtc

class TestTN(unittest.TestCase):
    def test_to_matrix(self):
        mat = tc.randn(2**6, 2**6, dtype=tc.complex128)
        mpo = qtc.MPO.from_matrix(mat)
        mat2 = mpo.to_matrix()
        self.assertTrue(tc.allclose(mat, mat2))
    
    def test_mpo_apply(self):
        # 验证 mpo 两体门的正确性

        # 随机一个态
        M = qtc.MPO.from_random(L=4, bond_dim=3)
        mat = M.to_matrix().numpy()

        # 随机一个两体门
        lmat = np.random.randn(4,4)
        rmat = np.random.randn(4,4)
        I = np.eye(2)

        lM = qt.linalg.kron(I, lmat, I)
        rM = qt.linalg.kron(I, rmat, I)

        # 严格作用
        mat1 = lM.T.conj() @ mat @ rM

        # apply_gate_ 作用
        M1 = M.copy()
        M1.apply_gate_(1, ((lmat.T.conj(), rmat), "topbottom"), svd_alg='eig', direction='right', trunc_para=(12, 1e-10, 1e-10))
        mat2 = M1.to_matrix().cpu().numpy()
        self.assertTrue(np.allclose(mat1, mat2))

        # apply_gate_ 作用
        M1 = M.copy()
        M1.apply_gate_(1, ((lmat.T.conj(), rmat), "topbottom"), svd_alg='svd', direction='right', trunc_para=(12, 1e-10, 1e-10))
        mat2 = M1.to_matrix().cpu().numpy()
        self.assertTrue(np.allclose(mat1, mat2))

    def test_swapsite(self):
        d = 5
        mat = tc.randn(*[2]*(2*d), dtype=tc.float64)
        mpo = qtc.MPO.from_matrix(mat.reshape(2**d, 2**d))
        mpo.swapsite_(1,4)
        self.assertTrue(tc.allclose(mpo.to_matrix() , mat.swapaxes(1,4).swapaxes(1+d,4+d).reshape(2**d, 2**d)))


if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_dm"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

