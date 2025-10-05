# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-01 20:07:21
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-05 21:37:38

import unittest
from quante.generate.basis.spin_half.spin_super.basis import *
import quante.generate.operas.spin as op
import quante as qt
import scipy.sparse as sp

class TestLiouvilleBasis(unittest.TestCase):
    def setUp(self):
        self.L = 4
        J, Δ, gamma = 1., 0.5, 1.
        self.J, self.Δ, self.gamma = J, Δ, gamma
        self.ham = op.sum(J*(op.xx(i,i+1) + op.yy(i,i+1)) for i in range(self.L-1))
        self.jump_ops = [np.sqrt(gamma)*op.I(i) for i in range(self.L)]
        self.liou = qt.generate.operas.super_oper.Lindbladian(self.L, self.ham, self.jump_ops)

    def test_full_basis(self):
        import quante.bridge.quspin_utils as qs
        basis = qs.spin_super_basis(self.L, pauli=True)
        mat = qs.hamiltonian(self.liou, basis, dtype=np.complex128)
        P_sym = basis.sym_basis.get_proj(np.complex128)
        P_antisym = 1j*basis.asym_basis.get_proj(np.complex128)
        P = sp.hstack([P_sym, P_antisym], format='csr')
        mat = qs.hamiltonian(self.liou, basis, dtype=np.complex128)
        mat1 = np.real_if_close((P.conj().T @ mat @ P).toarray())
        self.assertEqual(mat1.dtype, np.float64)

        basis = BasisFull(self.L,None,None,'stacked')
        mat0 = self.liou.to_matrix(basis, pauli=True, sparse=False)
        self.assertEqual(mat0.dtype, np.float64)
        self.assertTrue(np.allclose(mat0, mat1))
        eng0 = np.linalg.eigvals(mat0)
        eng0 = qt.linalg.sortcomplex(eng0)
        self.eng0 = eng0

        P = basis.projection_matrix()
        mat2 = np.real_if_close((P.conj().T @ mat @ P).toarray())
        self.assertTrue(np.allclose(mat0, mat2))
        rho = qt.generate.matrix.random_matrix(2**self.L, mtype='rho')
        vec0 = np.real_if_close(basis.project(rho.reshape(-1,1)))
        vec1 = np.real_if_close(P.conj().T @ rho.reshape(-1,1))
        self.assertTrue(np.allclose(vec0, vec1))
        rho2 = basis.recover(vec0).reshape(*rho.shape)
        self.assertTrue(np.allclose(rho, rho2))
        self.rho = rho

    def test_Ndiff_basis(self):
        self.test_full_basis()
        eng1 = []
        for Ndiff in range(0, self.L+1):
            basis = BasisFull(self.L,Ndiff,None,'stacked')
            mat0 = self.liou.to_matrix(basis, pauli=True, sparse=False)
            self.assertEqual(mat0.dtype, np.float64)
            eng1.append(np.linalg.eigvals(mat0))
        eng1 = qt.linalg.sortcomplex(np.concatenate(eng1))
        self.assertTrue(np.allclose(self.eng0, eng1))

    def test_Nup_basis(self):
        self.test_full_basis()
        eng1 = []
        for nup in range(0, 2*self.L+1):
            basis = BasisFull(self.L,None,nup,'stacked')
            mat0 = self.liou.to_matrix(basis, pauli=True, sparse=False)
            self.assertEqual(mat0.dtype, np.float64)
            eng1.append(np.linalg.eigvals(mat0))
        eng1 = qt.linalg.sortcomplex(np.concatenate(eng1))
        self.assertTrue(np.allclose(self.eng0, eng1))

    def test_Ndiff_Nup_basis(self):
        self.test_full_basis()
        eng1 = []
        for ndiff in range(0, self.L+1):
            for nup in range(ndiff, 2*self.L+1, 2):
                basis = BasisFull(self.L,ndiff,nup,'stacked')
                mat0 = self.liou.to_matrix(basis, pauli=True, sparse=False)
                self.assertEqual(mat0.dtype, np.float64)
                eng1.append(np.linalg.eigvals(mat0))
        eng1 = qt.linalg.sortcomplex(np.concatenate(eng1))
        self.assertTrue(np.allclose(self.eng0, eng1))

    def test_Z21_basis(self):
        self.test_full_basis()
        Px = [i+j*self.L for j in [0,1] for i in range(self.L-1,-1,-1)]
        eng1 = []
        for block in [0,1]:
            basis = BasisZ21(self.L,None,None,'stacked', pblock=(Px, block))
            mat0 = self.liou.to_matrix(basis, pauli=True, sparse=False)
            self.assertEqual(mat0.dtype, np.float64)
            eng1.append(np.linalg.eigvals(mat0))
        eng1 = qt.linalg.sortcomplex(np.concatenate(eng1))
        self.assertTrue(np.allclose(self.eng0, eng1))

    def test_Z21_Ndiff_basis(self):
        self.test_full_basis()
        Px = [i+j*self.L for j in [0,1] for i in range(self.L-1,-1,-1)]
        eng1 = []
        for ndiff in range(0, self.L+1):
            for block in [0,1]:
                basis = BasisZ21(self.L,ndiff,None,'stacked', pblock=(Px, block))
                mat0 = self.liou.to_matrix(basis, pauli=True, sparse=False)
                self.assertEqual(mat0.dtype, np.float64)
                eng1.append(np.linalg.eigvals(mat0))
        eng1 = qt.linalg.sortcomplex(np.concatenate(eng1))
        self.assertTrue(np.allclose(self.eng0, eng1))

    def test_Z21_Nup_basis(self):
        self.test_full_basis()
        Px = [i+j*self.L for j in [0,1] for i in range(self.L-1,-1,-1)]
        eng1 = []
        for nup in range(0, 2*self.L+1):
            for block in [0,1]:
                basis = BasisZ21(self.L,None,nup,'stacked', pblock=(Px, block))
                mat0 = self.liou.to_matrix(basis, pauli=True, sparse=False)
                self.assertEqual(mat0.dtype, np.float64)
                eng1.append(np.linalg.eigvals(mat0))
        eng1 = qt.linalg.sortcomplex(np.concatenate(eng1))
        self.assertTrue(np.allclose(self.eng0, eng1))

    def test_Z21_Nup_Ndiff_basis(self):
        self.test_full_basis()
        Px = [i+j*self.L for j in [0,1] for i in range(self.L-1,-1,-1)]
        eng1 = []
        for ndiff in range(0, self.L+1):
            for nup in range(ndiff, 2*self.L+1, 2):
                for block in [0,1]:
                    basis = BasisZ21(self.L,ndiff,nup,'stacked', pblock=(Px, block))
                    mat0 = self.liou.to_matrix(basis, pauli=True, sparse=False)
                    self.assertEqual(mat0.dtype, np.float64)
                    eng1.append(np.linalg.eigvals(mat0))
        eng1 = qt.linalg.sortcomplex(np.concatenate(eng1))
        self.assertTrue(np.allclose(self.eng0, eng1))

    def test_Z21_block_compare(self):
        self.test_full_basis()
        for block in [0,1]:
            Px = [i+j*self.L for j in [0,1] for i in range(self.L-1,-1,-1)]
            basis = BasisZ21(self.L,None,None,'stacked', pblock=(Px, block))
            mat0 = self.liou.to_matrix(basis, pauli=True, sparse=False)

            from quante.bridge import quspin_utils as qs
            basis2 = qs.spin_super_basis(self.L, pauli=True, pblock=block)
            mat = qs.hamiltonian(self.liou, basis2, dtype=np.complex128)
            mat1 = basis2.realify(mat).toarray()
            self.assertTrue(np.allclose(mat0, mat1))

    def test_Z21_project_recover(self):
        self.test_full_basis()
        Px = [i+j*self.L for j in [0,1] for i in range(self.L-1,-1,-1)]
        basis0 = BasisZ21(self.L,None,None,'stacked', pblock=(Px, 0))
        basis1 = BasisZ21(self.L,None,None,'stacked', pblock=(Px, 1))

        rho = qt.generate.matrix.random_matrix(2**self.L, mtype='rho')
        vec00 = np.real_if_close(basis0.project(rho.reshape(-1,1)))
        vec01 = np.real_if_close(basis1.project(rho.reshape(-1,1)))
        rho0 = basis0.recover(vec00).reshape(*rho.shape)
        rho1 = basis1.recover(vec01).reshape(*rho.shape)
        self.assertTrue(np.allclose(rho, rho0 + rho1))

        P0 = basis0.projection_matrix()
        P1 = basis1.projection_matrix()
        vec10 = np.real_if_close(P0.conj().T @ rho.reshape(-1,1))
        vec11 = np.real_if_close(P1.conj().T @ rho.reshape(-1,1))
        self.assertTrue(np.allclose(vec00, vec10))
        self.assertTrue(np.allclose(vec01, vec11))

if __name__ == '__main__':
    unittest.main()
