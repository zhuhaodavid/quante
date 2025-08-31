# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-01 00:36:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 13:58:54

import unittest

import quante as qt
import numpy as np

eps = 1e-10

try:
    import tenpy
    tenpy_installed = True
except ImportError:
    tenpy_installed = False

try:
    from quante.bridge.torch_utils import totc
    import torch as tc
    torch_installed = True
except ImportError:
    torch_installed = False

class TestTN(unittest.TestCase):
    
    @unittest.skipIf(not tenpy_installed, "tenpy is not installed")
    def test_lanczos_ground(self):
        from tenpy.linalg.krylov_based import LanczosEvolution, LanczosGroundState
        import tenpy.linalg.np_conserved as npc
        dim = 10
        chinfo = npc.ChargeInfo()  # the second argument is just a descriptive name
        legcharges1 = npc.LegCharge.from_trivial(dim, chinfo)
        legcharges2 = npc.LegCharge.from_trivial(dim, chinfo)

        H = qt.generate.matrix.random_matrix(dim, mtype='herm')
        psi0 = qt.generate.state.random(dim).reshape(-1)

        from quante.linalg.krylov.toy import lanczos_ground_state

        res00, vec = lanczos_ground_state(H.dot, psi0)

        H = npc.Array.from_ndarray(H,[legcharges1,legcharges1])
        psi0 = npc.Array.from_ndarray(psi0,[legcharges2])
        res0, res1, _ = LanczosGroundState(H, psi0, {}).run()
        self.assertAlmostEqual(res0, res00)
        self.assertTrue(np.allclose(vec, res1.to_ndarray()))
    
    @unittest.skipIf(not tenpy_installed, "tenpy is not installed")
    def test_lanczos_evolve(self):
        from tenpy.linalg.krylov_based import LanczosEvolution, LanczosGroundState
        import tenpy.linalg.np_conserved as npc
        chinfo = npc.ChargeInfo()  # the second argument is just a descriptive name
        legcharges1 = npc.LegCharge.from_trivial(10, chinfo)
        legcharges2 = npc.LegCharge.from_trivial(10, chinfo)

        H = qt.generate.matrix.random_matrix(10, mtype='herm')
        psi0 = qt.generate.state.random(10).reshape(-1)

        from quante.linalg.krylov.toy import lanczos_evolve_state

        vec = lanczos_evolve_state(H.dot, psi0, 0.1)

        H = npc.Array.from_ndarray(H,[legcharges1,legcharges1])
        psi0 = npc.Array.from_ndarray(psi0,[legcharges2])
        res0, res1 = LanczosEvolution(H, psi0, {}).run(0.1)
        self.assertTrue(np.allclose(vec, res0.to_ndarray()))


    def test_eigsolve_np_dense_real_arnoldi(self):
        from quante.linalg.krylov import eigsolve
        d = 100
        mat = np.random.randn(d, d)
        x0 = np.random.randn(d)
        k = 3
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=False)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat @ vec[i] - val[i] * vec[i]))


    def test_eigsolve_np_dense_complex_arnoldi(self):
        from quante.linalg.krylov import eigsolve
        d = 100
        mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        x0 = np.random.randn(d) + 1j * np.random.randn(d)
        k = 3
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=False)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat @ vec[i] - val[i] * vec[i]))

    def test_eigsolve_np_sparse_real_arnoldi(self):
        from quante.linalg.krylov import eigsolve
        mat = qt.generate.matrix.heisenberg_matrix(L=10, sparse=True)
        d = mat.shape[0]
        x0 = np.random.randn(d)
        k = 3
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=False)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat @ vec[i] - val[i] * vec[i]))
 
    def test_eigsolve_np_sparse_complex_arnoldi(self):
        from quante.linalg.krylov import eigsolve
        mat = qt.generate.matrix.heisenberg_matrix(L=10, j=(1j, 1, 1), sparse=True)
        d = mat.shape[0]
        x0 = np.random.randn(d) + 1j * np.random.randn(d)
        k = 3
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=False)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat @ vec[i] - val[i] * vec[i]))
    
    def test_eigsolve_np_dense_real_lanczos(self):
        from quante.linalg.krylov import eigsolve
        d = 100
        mat = np.random.randn(d, d)
        mat = (mat + mat.T) / 2
        x0 = np.random.randn(d)
        k = 3
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=True)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat @ vec[i] - val[i] * vec[i]))

    def test_eigsolve_np_dense_complex_lanczos(self):
        from quante.linalg.krylov import eigsolve
        d = 100
        mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        mat = (mat + mat.T.conj()) / 2
        x0 = np.random.randn(d) + 1j * np.random.randn(d)
        k = 3
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=True)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat @ vec[i] - val[i] * vec[i]))

    def test_eigsolve_np_sparse_real_lanczos(self):
        from quante.linalg.krylov import eigsolve
        mat = qt.generate.matrix.heisenberg_matrix(L=10, sparse=True)
        d = mat.shape[0]
        x0 = np.random.randn(d)
        k = 3
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=True)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat @ vec[i] - val[i] * vec[i]))

    def test_eigsolve_np_sparse_complex_lanczos(self):
        from quante.linalg.krylov import eigsolve
        mat = qt.generate.matrix.heisenberg_matrix(L=10, h=(0,1,0), sparse=True)
        d = mat.shape[0]
        x0 = np.random.randn(d) + 1j * np.random.randn(d)
        k = 3
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=True)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat @ vec[i] - val[i] * vec[i]))


    @unittest.skipIf(not torch_installed, "torch is not installed")
    def test_eigsolve_tc_dense_real_arnoldi(self):
        from quante.linalg.krylov import eigsolve
        d = 100
        mat = np.random.randn(d, d)
        x0 = np.random.randn(d)
        k = 3
        mat, x0 = totc(mat), totc(x0)
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=False)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat.numpy() @ vec[i].numpy() - val[i] * vec[i].numpy()))
        

    @unittest.skipIf(not torch_installed, "torch is not installed")
    def test_eigsolve_tc_dense_complex_arnoldi(self):
        from quante.linalg.krylov import eigsolve
        d = 100
        mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        x0 = np.random.randn(d) + 1j * np.random.randn(d)
        k = 3
        mat, x0 = totc(mat), totc(x0)
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=False)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat.numpy() @ vec[i].numpy() - val[i] * vec[i].numpy()))

    @unittest.skipIf(not torch_installed, "torch is not installed")
    def test_eigsolve_tc_sparse_real_arnoldi(self):
        from quante.linalg.krylov import eigsolve
        mat = qt.generate.matrix.heisenberg_matrix(L=10, sparse=True)
        d = mat.shape[0]
        x0 = np.random.randn(d)
        k = 3
        mat, x0 = totc(mat), totc(x0)
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=False)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, tc.linalg.norm(mat @ vec[i] - val[i] * vec[i]).item())
 
    @unittest.skipIf(not torch_installed, "torch is not installed")
    def test_eigsolve_tc_sparse_complex_arnoldi(self):
        from quante.linalg.krylov import eigsolve
        mat = qt.generate.matrix.heisenberg_matrix(L=10, j=(1j, 1, 1), sparse=True)
        d = mat.shape[0]
        x0 = np.random.randn(d) + 1j * np.random.randn(d)
        k = 3
        mat, x0 = totc(mat), totc(x0)
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=False)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, tc.linalg.norm(mat @ vec[i] - val[i] * vec[i]).item())
    
    @unittest.skipIf(not torch_installed, "torch is not installed")
    def test_eigsolve_tc_dense_real_lanczos(self):
        from quante.linalg.krylov import eigsolve
        d = 100
        mat = np.random.randn(d, d)
        mat = (mat + mat.T) / 2
        x0 = np.random.randn(d)
        k = 3
        mat, x0 = totc(mat), totc(x0)
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=True)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat.numpy() @ vec[i].numpy() - val[i] * vec[i].numpy()))

    @unittest.skipIf(not torch_installed, "torch is not installed")
    def test_eigsolve_tc_dense_complex_lanczos(self):
        from quante.linalg.krylov import eigsolve
        d = 100
        mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        mat = (mat + mat.T.conj()) / 2
        x0 = np.random.randn(d) + 1j * np.random.randn(d)
        k = 3
        mat, x0 = totc(mat), totc(x0)
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=True)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat.numpy() @ vec[i].numpy() - val[i] * vec[i].numpy()))

    @unittest.skipIf(not torch_installed, "torch is not installed")
    def test_eigsolve_tc_sparse_real_lanczos(self):
        from quante.linalg.krylov import eigsolve
        mat = qt.generate.matrix.heisenberg_matrix(L=10, sparse=True)
        d = mat.shape[0]
        x0 = np.random.randn(d)
        k = 3
        mat, x0 = totc(mat), totc(x0)
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=True)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, tc.linalg.norm(mat @ vec[i] - val[i] * vec[i]).item())

    @unittest.skipIf(not torch_installed, "torch is not installed")
    def test_eigsolve_tc_sparse_complex_lanczos(self):
        from quante.linalg.krylov import eigsolve
        mat = qt.generate.matrix.heisenberg_matrix(L=10, h=(0,1,0), sparse=True)
        d = mat.shape[0]
        x0 = np.random.randn(d) + 1j*np.random.randn(d)
        k = 3
        mat, x0 = totc(mat), totc(x0)
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=True)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, tc.linalg.norm(mat @ vec[i] - val[i] * vec[i]).item())



if __name__ == "__main__":
    unittest.main()
