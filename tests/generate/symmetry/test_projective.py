# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-08 17:12:39
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-27 17:52:03

import unittest
import quante as qt
import numpy as np


class TestSpinHalfProj(unittest.TestCase):

    def test_Nup(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        
        for Nup in range(L+1):
            basis = qt.generate.basis.spin_basis(L=L, Nup=Nup)
            mat1 = ham.to_matrix(basis, pauli=False)
            proj = basis.projection_matrix()
            self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))

    
    def test_kblock(self):
        L = 8
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        
        for k in range(L):
            ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
            basis = qt.generate.basis.spin_basis(L=L, kblock=k)
            mat1 = ham.to_matrix(basis, pauli=False)
            proj = basis.projection_matrix()
            self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))


    def test_pblock(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        for p in [-1,1]:
            ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
            basis = qt.generate.basis.spin_basis(L=L, pblock=p)
            mat1 = ham.to_matrix(basis, pauli=False)
            proj = basis.projection_matrix()
            self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))
         

    def test_zblock(self):
        L = 2
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=False)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        for p in [-1,1]:
            ham = qt.generate.operas.heisenberg_operator(L, cyclic=False)
            basis = qt.generate.basis.spin_basis(L=L, zblock=p)
            mat1 = ham.to_matrix(basis, pauli=False)
            proj = basis.projection_matrix()
            self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))

    def test_pzblock(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        for p in [-1,1]:
            ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
            basis = qt.generate.basis.spin_basis(L=L, pzblock=p)
            mat1 = ham.to_matrix(basis, pauli=False)
            proj = basis.projection_matrix()
            self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))


    def test_Nup_kblock(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        for Nup in range(L+1):
            for k in range(L):
                ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
                basis = qt.generate.basis.spin_basis(L=L, Nup=Nup, kblock=k)
                mat1 = ham.to_matrix(basis, pauli=False)
                proj = basis.projection_matrix()
                self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))
                
    def test_Nup_pblock(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        for Nup in range(L+1):
            for p in [-1, 1]:
                ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
                basis = qt.generate.basis.spin_basis(L=L, Nup=Nup, pblock=p)
                mat1 = ham.to_matrix(basis, pauli=False)
                proj = basis.projection_matrix()
                self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))
                

    def test_Nup_zblock(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        for Nup in range(L+1):
            for z in [-1, 1]:
                ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
                basis = qt.generate.basis.spin_basis(L=L, Nup=Nup, zblock=z)
                mat1 = ham.to_matrix(basis, pauli=False)
                proj = basis.projection_matrix()
                self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))

    def test_Nup_pzblock(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        for Nup in range(L+1):
            for pz in [-1, 1]:
                ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
                basis = qt.generate.basis.spin_basis(L=L, Nup=Nup, pzblock=pz)
                mat1 = ham.to_matrix(basis, pauli=False)
                proj = basis.projection_matrix()
                self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))

    def test_Nup_kblock_pblock(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        for Nup in range(L+1):
            for k in range(L//2+1):
                for p in [-1, 1]:
                    ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
                    basis = qt.generate.basis.spin_basis(L=L, Nup=Nup, kblock=k, pblock=p)
                    mat1 = ham.to_matrix(basis, pauli=False)
                    proj = basis.projection_matrix()
                    self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))


    def test_Nup_kblock_pblock_zblock(self):
        L = 10
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis = qt.generate.basis.spin_basis(L=L)
        mat0 = ham.to_matrix(basis, pauli=False)
        for k in range(L//2+1):
            for p in [-1, 1]:
                for z in [-1, 1]:
                    ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
                    basis = qt.generate.basis.spin_basis(L=L, Nup=L//2, kblock=k, pblock=p, zblock=z)
                    mat1 = ham.to_matrix(basis, pauli=False)
                    proj = basis.projection_matrix()
                    self.assertTrue(np.allclose(proj.conj().T @ mat0 @ proj, mat1))

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestSpinHalfProj("test_zblock"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)