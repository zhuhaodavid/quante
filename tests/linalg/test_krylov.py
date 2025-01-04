# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-01 00:36:26
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-01 00:39:24

import unittest

from tenpy.linalg.krylov_based import LanczosEvolution, LanczosGroundState
import tenpy.linalg.np_conserved as npc
import quante as qt
import numpy as np

class TestTN(unittest.TestCase):
    def test_lanczos_ground(self):
        chinfo = npc.ChargeInfo()  # the second argument is just a descriptive name
        legcharges1 = npc.LegCharge.from_trivial(10, chinfo)
        legcharges2 = npc.LegCharge.from_trivial(10, chinfo)

        H = qt.generate.matrix.random_matrix(10, type='herm')
        psi0 = qt.generate.state.random(10).reshape(-1)

        res00, vec = qt.linalg.krylov.lanczos_ground_state(H.dot, psi0)

        H = npc.Array.from_ndarray(H,[legcharges1,legcharges1])
        psi0 = npc.Array.from_ndarray(psi0,[legcharges2])
        res0, res1, _ = LanczosGroundState(H, psi0, {}).run()
        self.assertAlmostEqual(res0, res00)
        self.assertTrue(np.allclose(vec, res1.to_ndarray()))
    
    def test_lanczos_evolve(self):
        chinfo = npc.ChargeInfo()  # the second argument is just a descriptive name
        legcharges1 = npc.LegCharge.from_trivial(10, chinfo)
        legcharges2 = npc.LegCharge.from_trivial(10, chinfo)

        H = qt.generate.matrix.random_matrix(10, type='herm')
        psi0 = qt.generate.state.random(10).reshape(-1)

        vec = qt.linalg.krylov.lanczos_evolve_state(H.dot, psi0, 0.1)

        H = npc.Array.from_ndarray(H,[legcharges1,legcharges1])
        psi0 = npc.Array.from_ndarray(psi0,[legcharges2])
        res0, res1 = LanczosEvolution(H, psi0, {}).run(0.1)
        self.assertTrue(np.allclose(vec, res0.to_ndarray()))

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_dm"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
