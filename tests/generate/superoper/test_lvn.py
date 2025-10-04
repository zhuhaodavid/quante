# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-01 00:36:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-04 17:49:12

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
        hammat = ham.to_matrix(basis=basis, sparse=True, pauli=False)

        Lindblad_R = [np.sqrt(gamma_R) * op.pm(i+1,i).to_matrix(basis=basis, pauli=False, sparse=True) for i in range(L-1)]
        Lindblad_L = [np.sqrt(gamma_L) * op.pm(i,i+1).to_matrix(basis=basis, pauli=False, sparse=True) for i in range(L-1)]

        mats = [op.n(i).to_matrix(basis=basis, pauli=False, sparse=True) for i in range(L)]
        Ns = basis.Ns
        def particle_number(t, rho):
            return np.array([np.trace(rho.reshape((Ns,Ns)) @ m.toarray()) for m in mats])

        lvn = qt.generate.operas.super_oper.LiouvillianLinearOperator(hammat, Lindblad_R + Lindblad_L).to_matrix()
        state = qt.generate.state.product_state(['up']+['dn']*(L-1), Nup=1)
        rhoinit = np.outer(state, state)

        res1 = qt.linalg.evolve_and_measure(
            lvn, rhoinit, [10, 20, 30, 40, 50], 
            measure=particle_number, 
            method='eig-cpu',
            ttype='imag-time'
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
                method='eig-cuda:0',
                ttype='imag-time'
            )
            self.assertTrue(np.allclose(self.res1, res2))

            res2 = qt.linalg.evolve_and_measure(
                self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
                measure=self.particle_number, 
                method='mul-cuda:0',
                ttype='imag-time'
            )
            self.assertTrue(np.allclose(self.res1, res2))
        
        res2 = qt.linalg.evolve_and_measure(
            self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
            measure=self.particle_number, 
            method='mul-cpu',
            ttype='imag-time'
        )
        self.assertTrue(np.allclose(self.res1, res2))
        
        res2 = qt.linalg.evolve_and_measure(
            self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
            measure=self.particle_number, 
            method='RK45',
            ttype='imag-time'
        )
        self.assertTrue(np.allclose(self.res1, res2))

        self.lvn.default_mul = 'lo'
        res2 = qt.linalg.evolve_and_measure(
            self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
            measure=self.particle_number, 
            method='mul-cpu',
            ttype='imag-time'
        )
        self.assertTrue(np.allclose(self.res1, res2))

        res2 = qt.linalg.evolve_and_measure(
            self.lvn, self.rhoinit, [10, 20, 30, 40, 50], 
            measure=self.particle_number, 
            method='RK45',
            ttype='imag-time'
        )

try:
    import quspin
    quspin_available = True
except:
    quspin_available = False


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

        lo = Lindblad_R[0].to_matrix(basis, sparse=True, pauli=False)
        # lvn = qt.generate.matrix.superoper.make_Liouvillian(ham, Lindblad_R + Lindblad_L, basis, pauli=False).matrix
        # liou = qt.generate.operas.super_oper.LiouvilleOper(ham, Lindblad_R + Lindblad_L).to_matrix(basis=basis, sparse=True, pauli=False)


        # lvnOper = qt.generate.matrix.superoper.make_LiouvillianOper(L, ham, Lindblad_R + Lindblad_L)
        lvnOper = qt.generate.operas.super_oper.LiouvilleOper(L, ham, Lindblad_R + Lindblad_L)
        # print(lvnOper)
        basis = qt.generate.basis.spin_basis(L=2*L)
        lvn = lvnOper.to_matrix(basis, sparse=True, pauli=False)
        # self.assertAlmostEqual(np.linalg.norm((lvn - lvn2).data), 0.)

        # lvnOper = qt.generate.matrix.superoper.make_LiouvillianOper(L, ham, Lindblad_R + Lindblad_L, format='ladder')
        lvnOper = qt.generate.operas.super_oper.LiouvilleOper(L, ham, Lindblad_R + Lindblad_L, indx_order='snake')
        basis = qt.generate.basis.spin_basis(L=2*L)
        lvn3 = lvnOper.to_matrix(basis, sparse=True, pauli=False)

        lvn = lvn.reshape(*[2]*(4*L)).transpose([i if j == 0 else i+L for i in range(L) for j in range(2)] + [i if j == 0 else i+L for i in range(2*L,3*L) for j in range(2)]).reshape(2**(2*L), 2**(2*L))
        self.assertAlmostEqual(np.linalg.norm((lvn - lvn3).data), 0)
    

    @unittest.skipIf(not quspin_available, "quspin not available")
    def test_make_LiouvillianOper_kblock(self):
        import quante.bridge.quspin_utils as qs
        op = qt.generate.operas.spin

        L = 4
        gamma = 1.0

        ham = op.heisenberg_operator(L=L, cyclic=True)
        Lindblad = [gamma**0.5 * op.p(i) for i in range(L)]

        # lvn = qt.generate.matrix.superoper.make_LiouvillianOper(
        #     L, ham, Lindblad, format='ladder'
        # )
        lvn = qt.generate.operas.super_oper.LiouvilleOper(
            L, ham, Lindblad, indx_order='snake'
        )

        def sort(arr):
            idx = np.lexsort((arr.imag, np.round(arr.real, 10)))
            return arr[idx]

        basis = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True)
        mat1 = qs.hamiltonian(lvn, basis, dtype=np.complex128, sparse=False)
        engs1 = np.linalg.eigvals(mat1)
        engs1 = sort(engs1)

        res = []
        for kblock in range(L):
            basis = qs.spin_basis_2d(Lx=2, Ly=L, kyblock=kblock, pauli=True)
            mat1 = qs.hamiltonian(lvn, basis, dtype=np.complex128, sparse=False, check_symm=True)
            engs_block = np.linalg.eigvals(mat1)
            res.append(engs_block)
        engs2 = np.concatenate(res)
        engs2 = sort(engs2)
        self.assertTrue(np.allclose(engs1, engs2))

        res = []
        for pblock in [0,1]:
            basis = qs.spin_basis_2d(Lx=2, Ly=L, pyblock=pblock, pauli=True)
            mat1 = qs.hamiltonian(lvn, basis, dtype=np.complex128, sparse=False, check_symm=True)
            engs_block = np.linalg.eigvals(mat1)
            res.append(engs_block)
        engs2 = np.concatenate(res)
        engs2 = sort(engs2)
        self.assertTrue(np.allclose(engs1, engs2))



    

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_expm_multiply_numba_matrix"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
