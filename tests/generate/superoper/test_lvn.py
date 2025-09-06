# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-01 00:36:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-07 02:46:18

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

class TestmakeLiouvillianOper(unittest.TestCase):
    def test_make_LiouvillianOper(self):
        op = qt.generate.operas.spin

        L = 8
        J = 1.
        gamma_R = 1.0
        gamma_L = 0.7

        ham = op.builder()
        for i in range(L-1):
            ham += "+-", [i+1, i], -J
            ham += "+-", [i, i+1], -J
            ham += "xx", [i, i+1], -J
            ham += "yy", [i, i+1], J/2
        ham = ham.build()

        basis = qt.generate.basis.spin_basis(L=L)
        Lindblad_R = [np.sqrt(gamma_R) * op.pm(i+1,i) for i in range(L-1)]
        Lindblad_L = [np.sqrt(gamma_L) * op.pm(i,i+1) for i in range(L-1)]

        lo = Lindblad_R[0].to_matrix(basis, sparse=True)
        lvn = qt.generate.superoper.make_Liouvillian(ham, Lindblad_R + Lindblad_L, basis).matrix


        lvnOper = qt.generate.superoper.make_LiouvillianOper(L, ham, Lindblad_R + Lindblad_L)
        # print(lvnOper)
        basis = qt.generate.basis.spin_basis(L=2*L)
        lvn2 = lvnOper.to_matrix(basis, sparse=True)
        self.assertAlmostEqual(np.linalg.norm((lvn - lvn2).data), 0.)

        lvnOper = qt.generate.superoper.make_LiouvillianOper(L, ham, Lindblad_R + Lindblad_L, format='ladder')
        basis = qt.generate.basis.spin_basis(L=2*L)
        lvn3 = lvnOper.to_matrix(basis, sparse=True)

        lvn = lvn.reshape(*[2]*(4*L)).transpose([i if j == 0 else i+L for i in range(L) for j in range(2)] + [i if j == 0 else i+L for i in range(2*L,3*L) for j in range(2)]).reshape(2**(2*L), 2**(2*L))
        self.assertAlmostEqual(np.linalg.norm((lvn - lvn3).data), 0)

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_expm_multiply_numba_matrix"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
