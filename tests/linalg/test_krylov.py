# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-01 00:36:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 18:20:04

import unittest

import quante as qt
import numpy as np
from quante.linalg.krylov import eigsolve

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

orths = ['ModifiedGramSchmidt2', 'ModifiedGramSchmidt']

class TestKrylov(unittest.TestCase):
    
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
    
    def _main_eigsolve(self, mat, x0, orth, isherm):
        k = 3
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=isherm, orth=orth)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, np.linalg.norm(mat @ vec[i] - val[i] * vec[i]))

        mat, x0 = totc(mat), totc(x0)
        val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=isherm, orth=orth)
        val = val[:k]
        for i in range(k):
            self.assertGreater(eps, tc.linalg.norm(mat.to(tc.complex128) @ vec[i].to(tc.complex128) - val[i] * vec[i].to(tc.complex128)).item())
        
        if tc.cuda.is_available():
            mat, x0 = mat.cuda(), x0.cuda()
            val, vec, _ = eigsolve(mat, x0, howmany=3, isherm=isherm, orth=orth)
            val = val[:k]
            for i in range(k):
                self.assertGreater(eps, tc.linalg.norm(mat.to(tc.complex128) @ vec[i].to(tc.complex128) - val[i] * vec[i].to(tc.complex128)).item())
        

    def test_eigsolve_dense_arnoldi(self):
        d = 100
        for dtype in [np.float64, np.complex128]:
            for orth in orths:
                mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
                mat = mat.astype(dtype)
                x0 = np.random.randn(d) + 1j * np.random.randn(d)
                x0 = x0.astype(dtype)
                self._main_eigsolve(mat, x0, orth, isherm=False)
                

    def test_eigsolve_sparse_arnoldi(self):
        d = 100
        for dtype in [np.float64, np.complex128]:
            for orth in orths:
                mat = qt.generate.matrix.heisenberg_matrix(L=10, sparse=True)
                mat = mat.astype(dtype)
                d = mat.shape[0]
                x0 = np.random.randn(d) + 1j * np.random.randn(d)
                x0 = x0.astype(dtype)
                self._main_eigsolve(mat, x0, orth, isherm=False)


    def test_eigsolve_dense_lanczos(self):
        d = 100
        for dtype in [np.float64, np.complex128]:
            for orth in orths:
                mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
                mat = (mat + mat.T.conj()) / 2
                mat = mat.astype(dtype)
                x0 = np.random.randn(d) + 1j * np.random.randn(d)
                x0 = x0.astype(dtype)
                self._main_eigsolve(mat, x0, orth, isherm=True)
                

    def test_eigsolve_sparse_lanczos(self):
        d = 100
        for dtype in [np.float64, np.complex128]:
            for orth in orths:
                mat = qt.generate.matrix.heisenberg_matrix(L=10, sparse=True)
                mat = mat.astype(dtype)
                d = mat.shape[0]
                x0 = np.random.randn(d)
                x0 = x0.astype(dtype)
                self._main_eigsolve(mat, x0, orth, isherm=True)
