# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-01 00:36:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-27 17:52:31

import unittest
import scipy.sparse
import quante as qt
import numpy as np

class TestExpmMultiply(unittest.TestCase):
    def _gen_mat_state(self, L, dtype=np.complex128):
        # 拿到矩阵
        ham = qt.generate.operas.heisenberg_operator(L)
        basis = qt.generate.basis.spin_basis(L)
        state = qt.generate.state.random(basis.Ns, seed=42, dtype=dtype)
        return ham.to_matrix(basis, pauli=False, sparse=True), state

        # np.random.seed(42)
        # mat = np.random.randn(10, 10) #+ np.random.randn(100, 100)*1j
        # basis = np.random.randn(10) + np.random.randn(10)*1j
        # return mat, basis
    
    def test_expm_multiply_numba(self):
        from quante.linalg.evolve.nbfuc.expm_mul_core import _expm_multiply_simple, _expm_multiply_interval
        mat, state = self._gen_mat_state(16)
        state = state.astype(np.complex128)

        # 验证 exp(-1j*mat) @ state
        # 一次性作用:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state)
        res2, s = _expm_multiply_simple((-1j*mat), state, 1.)
        res3, s = _expm_multiply_simple(mat, state, -1j)
        self.assertTrue(np.allclose(res1, res2))
        self.assertTrue(np.allclose(res1, res3))
        
        # 方法0:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=10)
        res2, s1, s2 = _expm_multiply_interval((-1j*mat), state, 1., start=0, stop=10, num=10)
        res3, s1, s2 = _expm_multiply_interval(mat, state, -1j, start=0, stop=10, num=10)
        print(s1)
        self.assertTrue(np.allclose(res1, res2))
        self.assertTrue(np.allclose(res1, res3))
        
        # 方法1:
        L = 16
        ham = qt.generate.operas.heisenberg_operator(L)
        basis = qt.generate.basis.spin_basis(L)
        state = qt.generate.state.random(basis.Ns, dtype=np.complex128)
        mat = ham.to_matrix(basis, sparse=True, pauli=False)
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=19)
        res2, s1, s2 = _expm_multiply_interval((-1j*mat), state, 1., start=0, stop=10, num=19)
        res3, s1, s2 = _expm_multiply_interval(mat, state, -1j, start=0, stop=10, num=19)
        self.assertTrue(np.allclose(res1, res2))
        self.assertTrue(np.allclose(res1, res3))
        

        # 方法2:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=20)
        res2, s, _ = _expm_multiply_interval((-1j*mat), state, 1., start=0, stop=10, num=20)
        res3, s, _ = _expm_multiply_interval(mat, state, -1j, start=0, stop=10, num=20)
        print(s)
        self.assertTrue(np.allclose(res1, res2))
        self.assertTrue(np.allclose(res1, res3))
        
        
        # 验证 exp(mat) @ state，其中 mat 是实数矩阵
        
        mat, state = self._gen_mat_state(16, dtype=np.float64)
        # 一次性作用:
        res1 = scipy.sparse.linalg.expm_multiply((mat), state)
        res2, _ = _expm_multiply_simple(mat, state, 1.)
        self.assertTrue(np.allclose(res1, res2))
        
    
        # 方法0:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=10)
        res2, s, _ = _expm_multiply_interval(mat, state, 1., start=0, stop=10, num=10)
        print(s)
        self.assertTrue(np.allclose(res1, res2))
        
        
        # 方法1:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=19)
        res2, s, _ = _expm_multiply_interval(mat, state, 1., start=0, stop=10, num=19)
        print(s)
        self.assertTrue(np.allclose(res1, res2))
        

        # 方法2:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=20)
        res2, s, _ = _expm_multiply_interval(mat, state, 1., start=0, stop=10, num=20)
        print(s)
        self.assertTrue(np.allclose(res1, res2))


    def test_expm_multiply_torch(self):
        from quante.bridge.torch_utils.linalg.expm_multiply import _expm_multiply_simple, _expm_multiply_interval
        from quante.bridge.torch_utils.linalg.sparse import to_csr
        import torch as tc
        mat, state = self._gen_mat_state(16)
        # tcmat = tc.tensor(mat, device="cuda").to_sparse_csr()
        try:
            tcmat = to_csr(mat, device="cuda")
            tcstate = tc.tensor(state, device="cuda")
        except:
            tcmat = to_csr(mat, device="cpu")
            tcstate = tc.tensor(state, device="cpu")

        # 验证 exp(-1j*mat) @ state
        # 一次性作用:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state)
        res2, s = _expm_multiply_simple((-1j*tcmat), tcstate, 1.)
        res3, s = _expm_multiply_simple(tcmat, tcstate, -1j)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        self.assertTrue(np.allclose(res1, res3.cpu().numpy()))
        
        # 方法0:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=10)
        res2, s1, s2 = _expm_multiply_interval((-1j*tcmat), tcstate, 1., start=0, stop=10, num=10)
        res3, s1, s2 = _expm_multiply_interval(tcmat, tcstate, -1j, start=0, stop=10, num=10)
        # print(s1)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        self.assertTrue(np.allclose(res1, res3.cpu().numpy()))
        # print(np.linalg.norm(res4-res2.cpu().numpy()))
        
        # 方法1:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=19)
        res2, s1, s2 = _expm_multiply_interval((-1j*tcmat), tcstate, 1., start=0, stop=10, num=19)
        res3, s1, s2 = _expm_multiply_interval(tcmat, tcstate, -1j, start=0, stop=10, num=19)
        print(s1)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        self.assertTrue(np.allclose(res1, res3.cpu().numpy()))
        

        # 方法2:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=20)
        res2, s, _ = _expm_multiply_interval((-1j*tcmat), tcstate, 1., start=0, stop=10, num=20)
        res3, s, _ = _expm_multiply_interval(tcmat, tcstate, -1j, start=0, stop=10, num=20)
        print(s)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        self.assertTrue(np.allclose(res1, res3.cpu().numpy()))
        
        
        # 验证 exp(mat) @ state，其中 mat 是实数矩阵
        
        mat, state = self._gen_mat_state(16)
        try:
            tcmat = to_csr(mat, device="cuda")
            tcstate = tc.tensor(state, device="cuda")
        except:
            tcmat = to_csr(mat, device="cpu")
            tcstate = tc.tensor(state, device="cpu")
        
        # 一次性作用:
        res1 = scipy.sparse.linalg.expm_multiply((mat), state)
        res2, _ = _expm_multiply_simple(tcmat, tcstate, 1.)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        
    
        # 方法0:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=10)
        res2, s, _ = _expm_multiply_interval(tcmat, tcstate, 1., start=0, stop=10, num=10)
        print(s)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        
        
        # 方法1:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=19)
        res2, s, _ = _expm_multiply_interval(tcmat, tcstate, 1., start=0, stop=10, num=19)
        print(s)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        

        # 方法2:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=20)
        res2, s, _ = _expm_multiply_interval(tcmat, tcstate, 1., start=0, stop=10, num=20)
        print(s)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))


    def _gen_mat_state_matrix(self, L):
        # 拿到矩阵
        np.random.seed(42)
        mat = np.random.randn(L, L)
        basis = np.random.randn(L)
        return mat, basis
    
    def test_expm_multiply_numba_matrix(self):
        from quante.linalg.evolve.nbfuc.expm_mul_core import _expm_multiply_simple, _expm_multiply_interval
        mat, state = self._gen_mat_state_matrix(20)
        statec = state.astype(np.complex128)

        # 验证 exp(-1j*mat) @ state
        # 一次性作用:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state)
        res2, s = _expm_multiply_simple((-1j*mat), statec, 1.)
        res3, s = _expm_multiply_simple(mat, statec, -1j)
        self.assertTrue(np.allclose(res1, res2))
        self.assertTrue(np.allclose(res1, res3))
        
        # 方法0:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=10)
        res2, s1, s2 = _expm_multiply_interval((-1j*mat), statec, 1., start=0, stop=10, num=10)
        res3, s1, s2 = _expm_multiply_interval(mat, statec, -1j, start=0, stop=10, num=10)
        print(s1)
        self.assertTrue(np.allclose(res1, res2))
        self.assertTrue(np.allclose(res1, res3))
        
        # 方法1:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=19)
        res2, s1, s2 = _expm_multiply_interval((-1j*mat), statec, 1., start=0, stop=10, num=19)
        res3, s1, s2 = _expm_multiply_interval(mat, statec, -1j, start=0, stop=10, num=19)
        print(s1)
        self.assertTrue(np.allclose(res1, res2))
        self.assertTrue(np.allclose(res1, res3))
        

        # 方法2:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=20)
        res2, s, _ = _expm_multiply_interval((-1j*mat), statec, 1., start=0, stop=10, num=20)
        res3, s, _ = _expm_multiply_interval(mat, statec, -1j, start=0, stop=10, num=20)
        print(s)
        self.assertTrue(np.allclose(res1, res2))
        self.assertTrue(np.allclose(res1, res3))
        
        
        # 验证 exp(mat) @ state，其中 mat 是实数矩阵
        
        mat, state = self._gen_mat_state_matrix(20)
        # 一次性作用:
        res1 = scipy.sparse.linalg.expm_multiply((mat), state)
        res2, _ = _expm_multiply_simple(mat, state, 1.)
        self.assertTrue(np.allclose(res1, res2))
        
    
        # 方法0:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=10)
        res2, s, _ = _expm_multiply_interval(mat, state, 1., start=0, stop=10, num=10)
        print(s)
        self.assertTrue(np.allclose(res1, res2))
        
        
        # 方法1:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=19)
        res2, s, _ = _expm_multiply_interval(mat, state, 1., start=0, stop=10, num=19)
        print(s)
        self.assertTrue(np.allclose(res1, res2))
        

        # 方法2:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=20)
        res2, s, _ = _expm_multiply_interval(mat, state, 1., start=0, stop=10, num=20)
        print(s)
        self.assertTrue(np.allclose(res1, res2))


    def test_expm_multiply_torch_matrix(self):
        from quante.bridge.torch_utils.linalg.expm_multiply import _expm_multiply_simple, _expm_multiply_interval
        import torch as tc
        mat, state = self._gen_mat_state_matrix(20)
        try:
            tcmat = tc.tensor(mat, device="cuda")
            tcstate = tc.tensor(state, device="cuda")
        except:
            tcmat = tc.tensor(mat, device="cpu")
            tcstate = tc.tensor(state, device="cpu")

        # 验证 exp(-1j*mat) @ state
        # 一次性作用:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state)
        res2, s = _expm_multiply_simple((-1j*tcmat), tcstate.to(dtype=tc.complex128), 1.)
        res3, s = _expm_multiply_simple(tcmat, tcstate.to(dtype=tc.complex128), -1j)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        self.assertTrue(np.allclose(res1, res3.cpu().numpy()))
        
        # 方法0:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=10)
        res2, s1, s2 = _expm_multiply_interval((-1j*tcmat), tcstate.to(dtype=tc.complex128), 1., start=0, stop=10, num=10)
        res3, s1, s2 = _expm_multiply_interval(tcmat, tcstate.to(dtype=tc.complex128), -1j, start=0, stop=10, num=10)
        # print(s1)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        self.assertTrue(np.allclose(res1, res3.cpu().numpy()))
        # print(np.linalg.norm(res4-res2.cpu().numpy()))
        
        # 方法1:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=19)
        res2, s1, s2 = _expm_multiply_interval((-1j*tcmat), tcstate.to(dtype=tc.complex128), 1., start=0, stop=10, num=19)
        res3, s1, s2 = _expm_multiply_interval(tcmat, tcstate.to(dtype=tc.complex128), -1j, start=0, stop=10, num=19)
        print(s1)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        self.assertTrue(np.allclose(res1, res3.cpu().numpy()))
        

        # 方法2:
        res1 = scipy.sparse.linalg.expm_multiply((-1j*mat), state, start=0, stop=10, num=20)
        res2, s, _ = _expm_multiply_interval((-1j*tcmat), tcstate.to(dtype=tc.complex128), 1., start=0, stop=10, num=20)
        res3, s, _ = _expm_multiply_interval(tcmat, tcstate.to(dtype=tc.complex128), -1j, start=0, stop=10, num=20)
        print(s)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        self.assertTrue(np.allclose(res1, res3.cpu().numpy()))
        
        
        # 验证 exp(mat) @ state，其中 mat 是实数矩阵
        
        mat, state = self._gen_mat_state_matrix(20)
        try:
            tcmat = tc.tensor(mat, device="cuda")
            tcstate = tc.tensor(state, device="cuda")
        except:
            tcmat = tc.tensor(mat, device="cpu")
            tcstate = tc.tensor(state, device="cpu")
        
        # 一次性作用:
        res1 = scipy.sparse.linalg.expm_multiply((mat), state)
        res2, _ = _expm_multiply_simple(tcmat, tcstate, 1.)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        
    
        # 方法0:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=10)
        res2, s, _ = _expm_multiply_interval(tcmat, tcstate, 1., start=0, stop=10, num=10)
        print(s)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        
        
        # 方法1:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=19)
        res2, s, _ = _expm_multiply_interval(tcmat, tcstate, 1., start=0, stop=10, num=19)
        print(s)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))
        

        # 方法2:
        res1 = scipy.sparse.linalg.expm_multiply(mat, state, start=0, stop=10, num=20)
        res2, s, _ = _expm_multiply_interval(tcmat, tcstate, 1., start=0, stop=10, num=20)
        print(s)
        self.assertTrue(np.allclose(res1, res2.cpu().numpy()))


if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_expm_multiply_numba_matrix"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
