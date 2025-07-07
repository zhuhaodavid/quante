# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-01 00:36:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-07 11:57:56

import unittest

import quante.bridge.torch_utils as qtc
import quante as qt
import numpy as np

try:
    import tenpy
    tenpy_installed = True
except ImportError:
    tenpy_installed = False

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

        res00, vec = qt.linalg.krylov.lanczos_ground_state(H.dot, psi0)

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

        vec = qt.linalg.krylov.lanczos_evolve_state(H.dot, psi0, 0.1)

        H = npc.Array.from_ndarray(H,[legcharges1,legcharges1])
        psi0 = npc.Array.from_ndarray(psi0,[legcharges2])
        res0, res1 = LanczosEvolution(H, psi0, {}).run(0.1)
        self.assertTrue(np.allclose(vec, res0.to_ndarray()))

    # @unittest.skipIf(not tenpy_installed, "tenpy is not installed")
    # def test_lanczos_ground(self):
    #     from tenpy.linalg.krylov_based import LanczosEvolution, LanczosGroundState
    #     import tenpy.linalg.np_conserved as npc
    #     import torch as tc
    #     dim = 10
    #     chinfo = npc.ChargeInfo()  # the second argument is just a descriptive name
    #     legcharges1 = npc.LegCharge.from_trivial(dim, chinfo)
    #     legcharges2 = npc.LegCharge.from_trivial(dim, chinfo)

    #     H = qt.generate.matrix.random_matrix(dim, mtype='herm')
    #     H = tc.tensor(H)
    #     psi0 = qt.generate.state.random(dim).reshape(-1)
    #     psi0 = tc.tensor(psi0)

    #     res00, vec = qtc.linalg.krylov.lanczos_ground_state(H.matmul, psi0)

    #     H = npc.Array.from_ndarray(H,[legcharges1,legcharges1])
    #     psi0 = npc.Array.from_ndarray(psi0,[legcharges2])
    #     res0, res1, _ = LanczosGroundState(H, psi0, {}).run()
    #     self.assertAlmostEqual(res0, res00.item())
    #     self.assertTrue(np.allclose(vec.numpy(), res1.to_ndarray()))

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_dm"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
