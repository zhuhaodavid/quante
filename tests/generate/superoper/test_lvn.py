# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-01 00:36:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-23 21:21:03

import unittest
import quante as qt
import numpy as np

class TestLiouvillian(unittest.TestCase):
    def setUp(self):
        op = qt.generate.operas

        L = 10
        gamma_R = 1.0
        gamma_L = 0.5

        ham = op.heisenberg_operator(L=L)
        basis = qt.generate.basis.spin_basis(L=L,Nup=1)
        hammat = ham.to_matrix(basis=basis, sparse=True)

        Lindblad_R = [np.sqrt(gamma_R) * op.pm(i+1,i).to_matrix(basis=basis, sparse=True) for i in range(L-1)]
        Lindblad_L = [np.sqrt(gamma_L) * op.pm(i,i+1).to_matrix(basis=basis, sparse=True) for i in range(L-1)]
        particle_number = [op.n(i).to_matrix(basis=basis, sparse=True) for i in range(L)]

        lvn = qt.generate.superoper.Liouvillian(hammat, Lindblad_R + Lindblad_L)
        state = qt.generate.state.product_state(['up']+['dn']*(L-1), Nup=1)
        rhoinit = np.outer(state, state)

        res1 = qt.linalg.evolve_and_measure(
            lvn, rhoinit, [10, 20, 30, 40, 50], 
            measure=particle_number, 
            method='eig-cpu'
        )
        self.lvn = lvn
        self.rhoinit = rhoinit
        self.res1 = res1
        self.particle_number = particle_number

    def test_time_measurements(self):
        import torch as tc
        if tc.cuda.is_available():
            res2 = qt.linalg.evolve_and_measure(
                self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
                measure=self.particle_number, 
                method='eig-cuda:0'
            )
            self.assertTrue(np.allclose(self.res1, res2))

            res2 = qt.linalg.evolve_and_measure(
                self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
                measure=self.particle_number, 
                method='mul-cuda:0'
            )
            self.assertTrue(np.allclose(self.res1, res2))
        
        res2 = qt.linalg.evolve_and_measure(
            self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
            measure=self.particle_number, 
            method='mul-cpu'
        )
        self.assertTrue(np.allclose(self.res1, res2))
        
        res2 = qt.linalg.evolve_and_measure(
            self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
            measure=self.particle_number, 
            method='RK45'
        )
        self.assertTrue(np.allclose(self.res1, res2))

        self.lvn.default_mul = 'lo'
        res2 = qt.linalg.evolve_and_measure(
            self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
            measure=self.particle_number, 
            method='mul-cpu'
        )
        self.assertTrue(np.allclose(self.res1, res2))

        res2 = qt.linalg.evolve_and_measure(
            self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
            measure=self.particle_number, 
            method='RK45'
        )


if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_expm_multiply_numba_matrix"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
