# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-24 12:29:22
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-12 18:14:26

import unittest
import numpy as np
import quante as qt
import quante.bridge.quspin_utils as qs
op = qt.generate.operas
from quante.generate.basis import spin_basis_general
from quante.generate.basis.spin_half.spin_general.basis import *
from quante.generate.basis.spin_half.spin_general.basis_core import *

class TestGeneralBasis(unittest.TestCase):
    L, J, Δ, gamma = 4, 1.0, 0.5, 1.0

    @classmethod
    def setUpClass(cls):
        L, J, Δ, gamma = cls.L, cls.J, cls.Δ, cls.gamma
        # 几何置换
        Lx, Ly = L, 2
        N_2d = Lx * Ly
        s = np.arange(N_2d)
        x = s % Lx
        y = s // Lx
        cls.perm = (Lx - x - 1) + Lx * y
        cls.permx = cls.perm
        cls.permy = x + Lx * (Ly - y - 1)
        cls.zperm = -(np.arange(2*L) + 1)
        # Hamiltonian
        cls.ham = op.sum(J*(op.xx(i,i+1) + op.yy(i,i+1) + Δ*op.zz(i,i+1)) for i in range(L-1))
        # Lindblad (m 跳跃)
        cls.lind_m = [np.sqrt(gamma)*op.m(i) for i in range(L)]
        cls.liou_m = op.super_oper.Lindbladian(L, cls.ham, cls.lind_m, indx_order='stacked', flip=False)
        cls.liou_m_flip = op.super_oper.Lindbladian(L, cls.ham, cls.lind_m, indx_order='stacked', flip=True)
        # Lindblad (z 跳跃)
        cls.lind_z = [np.sqrt(gamma)*op.z(i) for i in range(L)]
        cls.liou_z = op.super_oper.Lindbladian(L, cls.ham, cls.lind_z, indx_order='stacked', flip=False)
        cls.liou_z_flip = op.super_oper.Lindbladian(L, cls.ham, cls.lind_z, indx_order='stacked', flip=True)

    @staticmethod
    def _sorted_eigs(mat):
        return qt.linalg.sortcomplex(np.linalg.eigvals(mat))

    def test_pblock_equivalence(self):
        for block in [0, 1]:
            with self.subTest(pblock=block):
                basis = BasisZ2N(2*self.L, None, None, None, pblock=(self.perm, block, 2))
                m0 = self.liou_m.to_matrix(basis, pauli=True, sparse=False)
                basis = spin_basis_general(2*self.L, pblock=(self.perm, block))
                m1 = self.liou_m.to_matrix(basis, pauli=True, sparse=False)
                self.assertTrue(np.allclose(m1, m0))
                basis_full = qs.spin_basis_2d(self.L, 2, pauli=True, pxblock=block)
                m2 = qs.hamiltonian(self.liou_m, basis=basis_full, dtype=np.complex128, sparse=False)
                self.assertTrue(np.allclose(self._sorted_eigs(m1), self._sorted_eigs(m2)))
                

    def test_flip_ndiff_equivalence_scalar(self):
        for ndiff in range(-self.L//2, self.L//2+1):
            with self.subTest(Ndiff=ndiff):
                basis = BasisZ2N(2*self.L, np.arange(self.L, 2*self.L), ndiff, None)
                m0 = self.liou_m.to_matrix(basis, pauli=True, sparse=False)
                basis = spin_basis_general(2*self.L, Ndiff=(np.arange(self.L, 2*self.L), ndiff))
                m1 = self.liou_m.to_matrix(basis, pauli=True, sparse=False)
                self.assertTrue(np.allclose(m1, m0))

                basis_flip = qs.spin_basis_2d(self.L, 2, pauli=True, Nup=self.L+ndiff)
                m2 = qs.hamiltonian(self.liou_m_flip, basis=basis_flip, dtype=np.complex128, sparse=False)
                basis = spin_basis_general(2*self.L, Nup=ndiff+self.L)
                m3 = self.liou_m_flip.to_matrix(basis, pauli=True, sparse=False)
                self.assertTrue(np.allclose(m2, m3))
                self.assertTrue(np.allclose(self._sorted_eigs(m1), self._sorted_eigs(m2)))

    def test_z_nup2_equivalence(self):
        for ndiff in range(-self.L//2, self.L//2+1):
            with self.subTest(Ndiff=ndiff):
                res = []
                for Nup in range(ndiff,2*self.L+1-ndiff,2):
                    basis = BasisZ2N(2*self.L, np.arange(self.L, 2*self.L), ndiff, Nup)
                    m0 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                    basis = spin_basis_general(2*self.L, Ndiff=(np.arange(self.L, 2*self.L), ndiff), Nup=Nup)
                    m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                    self.assertTrue(np.allclose(m1, m0))
                    res.append(self._sorted_eigs(m1))
                eng1 = qt.linalg.sortcomplex(np.hstack(res))
                basis_flip_z = qs.spin_basis_2d(self.L, 2, pauli=True, Nup=self.L+ndiff)
                m2 = qs.hamiltonian(self.liou_z_flip, basis=basis_flip_z, dtype=np.complex128, sparse=False)
                eng2 = self._sorted_eigs(m2)
                self.assertTrue(np.allclose(eng1, eng2))

    def test_flip_ndiff_pblock_equivalence_scalar(self):
        for ndiff in range(-self.L//2, self.L//2+1):
            for block in [0, 1]:
                with self.subTest(Ndiff=ndiff, pblock=block):
                    basis = BasisZ2N(2*self.L, np.arange(self.L, 2*self.L), ndiff, None, pblock=(self.perm, block, 2))
                    m0 = self.liou_m.to_matrix(basis, pauli=True, sparse=False)

                    basis = spin_basis_general(
                        2*self.L,
                        Ndiff=(np.arange(self.L, 2*self.L), ndiff), pblock=(self.perm, block)
                    )
                    m1 = self.liou_m.to_matrix(basis, pauli=True, sparse=False)

                    self.assertTrue(np.allclose(m1, m0))

                    m1e = self._sorted_eigs(m1)
                    basis_flip = qs.spin_basis_2d(
                        self.L, 2, pauli=True, Nup=self.L+ndiff, pxblock=block
                    )
                    m2 = qs.hamiltonian(self.liou_m_flip, basis=basis_flip, dtype=np.complex128, sparse=False)
                    basis_flip = spin_basis_general(2*self.L, Nup=ndiff+self.L, pblock=(self.perm, block))
                    m3 = self.liou_m_flip.to_matrix(basis_flip, pauli=True, sparse=False)
                    self.assertTrue(np.allclose(m2, m3))
                    # m2 = self.liou_m_flip.to_matrix(basis_flip, pauli=True, sparse=False)
                    m2e = self._sorted_eigs(m2)
                    self.assertTrue(np.allclose(m1e, m2e))

    def test_z_nup2_pblock_equivalence(self):
        for ndiff in range(-self.L//2, self.L//2+1):
            for block in [0, 1]:
                with self.subTest(Ndiff=ndiff, pblock=block):
                    res = []
                    for Nup in range(ndiff,2*self.L+1-ndiff,2):
                        basis = BasisZ2N(2*self.L, np.arange(self.L, 2*self.L), ndiff, Nup, pblock=(self.perm, block, 2))
                        m0 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                        basis = spin_basis_general(2*self.L, Ndiff=(np.arange(self.L, 2*self.L), ndiff), Nup=Nup, pblock=(self.perm, block))
                        m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                        self.assertTrue(np.allclose(m1, m0))

                        res.append(self._sorted_eigs(m1))
                    eng1 = qt.linalg.sortcomplex(np.hstack(res))
                    basis_flip_z = qs.spin_basis_2d(self.L, 2, pauli=True, Nup=self.L+ndiff, pxblock=block)
                    m2 = qs.hamiltonian(self.liou_z_flip, basis=basis_flip_z, dtype=np.complex128, sparse=False)
                    eng2 = self._sorted_eigs(m2)
                    self.assertTrue(np.allclose(eng1, eng2))

    def test_p_z_blocks(self):
        for pb in [0, 1]:
            for zb in [0, 1]:
                with self.subTest(pblock=pb, zblock=zb):
                    basis_super = qs.spin_basis_2d(
                        self.L, 2, pauli=True, pxblock=pb, zblock=zb
                    )
                    m_ref = qs.hamiltonian(self.liou_z, basis=basis_super, dtype=np.complex128, sparse=False)
                    # m_ref = self.liou_z.to_matrix(basis_super, pauli=True, sparse=False)
                    e_ref = self._sorted_eigs(m_ref)
                    basis = BasisZ2N(2*self.L, np.arange(self.L, 2*self.L), None, None, pblock=(self.perm, pb, 2), zblock=(self.zperm, zb, 2))
                    m0 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                    basis = spin_basis_general(
                        2*self.L, pblock=(self.perm, pb), zblock=(self.zperm, zb)
                    )
                    m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                    self.assertTrue(np.allclose(m1, m0))
                    self.assertTrue(np.allclose(self._sorted_eigs(m1), e_ref))
                    # basis2 = BasisZ2N(
                    #     2*self.L, None, None,None,
                    #     pblock=(self.perm, pb), zblock=(self.zperm, zb)
                    # )
                    # m2 = self.liou_z.to_matrix(basis2, pauli=True, sparse=False)
                    # self.assertTrue(np.allclose(self._sorted_eigs(m2), e_ref))

    def test_flip_p_z_blocks(self):
        for ndiff in range(0, self.L//2+1):
            Ndiff = 0 if ndiff == 0 else [ndiff, -ndiff]
            Nup = self.L if ndiff == 0 else [self.L-ndiff, self.L+ndiff]
            for pb in [0, 1]:
                for zb in [0, 1]:
                    with self.subTest(ndiff=ndiff, pblock=pb, zblock=zb):
                        basis = BasisZ2N(
                            2*self.L, np.arange(self.L,2*self.L),
                            Ndiff, None, pblock=(self.perm, pb, 2), zblock=(self.zperm, zb, 2)
                        )
                        m0 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)

                        basis = spin_basis_general(
                            2*self.L, Ndiff=(np.arange(self.L,2*self.L), Ndiff), pblock=(self.perm, pb), zblock=(self.zperm, zb)
                        )
                        m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                        self.assertTrue(np.allclose(m1, m0))

                        e1 = self._sorted_eigs(m1)
                        basis_flip = qs.spin_basis_2d(
                            self.L, 2, pauli=True, Nup=Nup, pxblock=pb, zblock=zb
                        )
                        m2 = qs.hamiltonian(self.liou_z_flip, basis=basis_flip, dtype=np.complex128, sparse=False)
                        basis_flip = spin_basis_general(
                            2*self.L, Nup=Nup, pblock=(self.perm, pb), zblock=(self.zperm, zb)
                        )
                        m3 = self.liou_z_flip.to_matrix(basis_flip, pauli=True, sparse=False)
                        self.assertTrue(np.allclose(m2, m3))
                        # m2 = self.liou_z_flip.to_matrix(basis_flip, pauli=True, sparse=False)
                        e2 = self._sorted_eigs(m2)
                        self.assertTrue(np.allclose(e1, e2))
                        

    def test_z_nup2_p_z_block_equivalence(self):
        for ndiff in range(0, self.L//2+1):
            for pb in [0, 1]:
                for zb in [0, 1]:
                    with self.subTest(Ndiff=ndiff, pblock=pb, zblock=zb):
                        res = []
                        for Nup in range(ndiff,(2*self.L-ndiff)//2+2,2):
                            basis = BasisZ2N(2*self.L, np.arange(self.L, 2*self.L), list(set([ndiff,-ndiff])), list(set([Nup, 2*self.L-Nup])), pblock=(self.perm, pb, 2), zblock=(self.zperm, zb, 2))
                            m0 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                            basis = spin_basis_general(2*self.L, Ndiff=(np.arange(self.L, 2*self.L), list(set([ndiff,-ndiff]))), Nup=list(set([Nup, 2*self.L-Nup])), pblock=(self.perm, pb), zblock=(self.zperm, zb))
                            m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                            self.assertTrue(np.allclose(m1, m0))
                            res.append(self._sorted_eigs(m1))
                        eng1 = qt.linalg.sortcomplex(np.hstack(res))
                        basis_flip = qs.spin_basis_2d(
                            self.L, 2, pauli=True, Nup=list(set([self.L-ndiff,self.L+ndiff])), pxblock=pb, zblock=zb
                        )
                        m2 = qs.hamiltonian(self.liou_z_flip, basis=basis_flip, dtype=np.complex128, sparse=False)
                        eng2 = self._sorted_eigs(m2)
                        self.assertTrue(np.allclose(eng1, eng2))



    def test_px_py_z_blocks(self):
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                for b2 in [0, 1]:
                    with self.subTest(px=b0, py=b1, z=b2):
                        basis = BasisZ2N(2*self.L,None,None,None,
                            pxblock=(self.permx, b0, 2),
                            pyblock=(self.permy, b1, 2),
                            zblock=(self.zperm, b2, 2)
                        )
                        m0 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                        basis_gen = spin_basis_general(
                            2*self.L,
                            pxblock=(self.permx, b0),
                            pyblock=(self.permy, b1),
                            zblock=(self.zperm, b2)
                        )
                        m1 = self.liou_z.to_matrix(basis_gen, pauli=True, sparse=False)
                        self.assertTrue(np.allclose(m0,m1))
                        e1 = self._sorted_eigs(m1)
                        basis2 = qs.spin_basis_2d(
                            Lx=self.L, Ly=2, pauli=True,
                            pxblock=b0, pyblock=b1, zblock=b2
                        )
                        m2 = qs.hamiltonian(self.liou_z, basis=basis2, dtype=np.complex128, sparse=False)
                        # m2 = self.liou_z.to_matrix(basis2, pauli=True, sparse=False)
                        e2 = self._sorted_eigs(m2)
                        self.assertTrue(np.allclose(e1, e2))


    def test_flip_px_py_z_blocks(self):
        for ndiff in range(0, self.L//2+1):
            Ndiff = 0 if ndiff == 0 else [ndiff, -ndiff]
            Nup = self.L if ndiff == 0 else [self.L-ndiff, self.L+ndiff]
            for b0 in [0, 1]:
                for b1 in [0, 1]:
                    for b2 in [0, 1]:
                        with self.subTest(ndiff=ndiff, px=b0, py=b1, z=b2):
                            basis = BasisZ2N(2*self.L, np.arange(self.L,2*self.L), Ndiff, None,
                                pxblock=(self.permx, b0, 2),
                                pyblock=(self.permy, b1, 2),
                                zblock=(self.zperm, b2, 2)
                            )
                            m0 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                            basis = spin_basis_general(
                                2*self.L, Ndiff=(np.arange(self.L,2*self.L), Ndiff),
                                pxblock=(self.permx, b0),
                                pyblock=(self.permy, b1),
                                zblock=(self.zperm, b2)
                            )
                            m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                            self.assertTrue(np.allclose(m0,m1))
                            e1 = self._sorted_eigs(m1)
                            basis_flip = qs.spin_basis_2d(
                                Lx=self.L, Ly=2, Nup=Nup, pauli=True,
                                pxblock=b0, pzyblock=b1, zblock=b2
                            )
                            m2 = qs.hamiltonian(self.liou_z_flip, basis=basis_flip, dtype=np.complex128, sparse=False)
                            basis_flip = spin_basis_general(
                                2*self.L, Nup=Nup,
                                pxblock=(self.permx, b0),
                                pzyblock=(-(self.permy+1), b1),
                                zblock=(self.zperm, b2)
                            )
                            m3 = self.liou_z_flip.to_matrix(basis_flip, pauli=True, sparse=False)
                            self.assertTrue(np.allclose(m2, m3))
                            # m2 = self.liou_z_flip.to_matrix(basis_flip, pauli=True, sparse=False)
                            e2 = self._sorted_eigs(m2)
                            self.assertTrue(np.allclose(e1, e2))


    def test_z_nup2_px_py_z_block_equivalence(self):
        for ndiff in range(0, self.L//2+1):
            for b0 in [0, 1]:
                for b1 in [0, 1]:
                    for b2 in [0, 1]:
                        with self.subTest(ndiff=ndiff, px=b0, py=b1, z=b2):
                            res = []
                            for Nup in range(ndiff,(2*self.L-ndiff)//2+2,2):
                                basis = BasisZ2N(2*self.L, np.arange(self.L, 2*self.L),
                                                 list(set([ndiff,-ndiff])), list(set([Nup, 2*self.L-Nup])),
                                    pxblock=(self.permx, b0, 2),
                                    pyblock=(self.permy, b1, 2),
                                    zblock=(self.zperm, b2, 2)
                                                 )
                                m0 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                                basis = spin_basis_general(
                                    2*self.L, Ndiff=(np.arange(self.L, 2*self.L), list(set([ndiff,-ndiff]))), 
                                    Nup=list(set([Nup, 2*self.L-Nup])), 
                                    pxblock=(self.permx, b0),
                                    pyblock=(self.permy, b1),
                                    zblock=(self.zperm, b2)
                                )
                                m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                                self.assertTrue(np.allclose(m0,m1))
                                res.append(self._sorted_eigs(m1))
                            eng1 = qt.linalg.sortcomplex(np.hstack(res))
                            basis_flip = qs.spin_basis_2d(
                                Lx=self.L, Ly=2, Nup=list(set([self.L-ndiff, self.L+ndiff])), pauli=True,
                                pxblock=b0, pzyblock=b1, zblock=b2
                            )
                            m2 = qs.hamiltonian(self.liou_z_flip, basis=basis_flip, dtype=np.complex128, sparse=False)
                            eng2 = self._sorted_eigs(m2)
                            self.assertTrue(np.allclose(eng1, eng2))


    def test_ZNN(self):
        L = 5
        kblock = 1
        ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        basis1 = qt.generate.basis.spin_basis(L, kblock=kblock)
        mat1 = ham.to_matrix(basis1, pauli=True)

        basis2 = qs.spin_basis(L=L, pauli=True, kblock=kblock)
        mat2 = qs.hamiltonian(ham, basis2, sparse=False, dtype=np.complex128)
        self.assertTrue(np.allclose(mat1, mat2))

        Lx = L
        Ly = 1
        N_2d = Lx * Ly  # total number of sites
        s = np.arange(N_2d)  # sites [0,1,2,..]
        x = s % Lx  # x positions for sites
        y = s // Lx  # y positions for sites
        T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
        basis3 = BasisZNN(L, None, None, None, kxblock=(T_x, kblock, Lx))
        self.assertTrue(np.allclose(basis3.s_list, basis1.s_list))
        self.assertTrue(np.allclose(basis3.R_list, basis1.other_params['R_list']))

        mat3 = ham.to_matrix(basis3, pauli=True)
        self.assertTrue(np.allclose(mat1, mat3))

        zblock = 0
        basis1 = qt.generate.basis.spin_basis(L, zblock=(-1)**zblock)
        mat1 = ham.to_matrix(basis1, pauli=True)
        Z = - (np.arange(N_2d) + 1)
        basis3 = BasisZNN(L, None, None, None, zblock=(Z, zblock, 2))
        mat3 = ham.to_matrix(basis3, pauli=True)


        basis2 = qs.spin_basis(L=L, pauli=True, kblock=kblock, zblock=(-1)**zblock)
        mat2 = qs.hamiltonian(ham, basis2, sparse=False, dtype=np.complex128)
        basis3 = BasisZNN(L, None, None, None, kxblock=(T_x, kblock, Lx), zblock=(Z, zblock, 2))
        mat3 = ham.to_matrix(basis3, pauli=True)
        self.assertTrue(np.allclose(mat2, mat3))

        basis2 = qs.spin_basis_2d(Lx=Lx, Ly=Ly, pauli=True, kxblock=kblock, zblock=zblock)
        mat2 = qs.hamiltonian(ham, basis2, sparse=False, dtype=np.complex128)
        basis3 = qt.generate.basis.spin_basis_2d(Lx=Lx, Ly=Ly, kxblock=kblock, zblock=zblock)
        mat3 = ham.to_matrix(basis3, pauli=True)
        self.assertTrue(np.allclose(mat2, mat3))


        Lx, Ly = 3, 4
        N_2d = Lx * Ly
        ham = qt.generate.operas.heisenberg_operator(2, cyclic=False, j=(1,1,0))
        ham = ham.translate(direction='x', tol=Lx, Lx=Lx, Ly=Ly).translate(direction='y', tol=Ly, Lx=Lx, Ly=Ly)

        for kxblock in range(Lx):
            for kyblock in range(Ly):
                basis2 = qs.spin_basis_2d(Lx=Lx, Ly=Ly, pauli=True, kxblock=kxblock, kyblock=kyblock)
                mat2 = qs.hamiltonian(ham, basis2, sparse=False, dtype=np.complex128)
                basis3 = qt.generate.basis.spin_basis_2d(Lx=Lx, Ly=Ly, kxblock=kxblock, kyblock=kyblock)
                mat3 = ham.to_matrix(basis3, pauli=True)
                self.assertTrue(np.allclose(mat2, mat3))
        
        Lx, Ly = 3, 5
        N_2d = Lx * Ly
        ham = qt.generate.operas.heisenberg_operator(2, cyclic=False, j=(1,1,0))
        ham = ham.translate(direction='x', tol=Lx, Lx=Lx, Ly=Ly).translate(direction='y', tol=Ly, Lx=Lx, Ly=Ly)

        for kxblock in range(Lx):
            for kyblock in range(Ly):
                basis2 = qs.spin_basis_2d(Lx=Lx, Ly=Ly, pauli=True, Nup=4, kxblock=kxblock, kyblock=kyblock)
                mat2 = qs.hamiltonian(ham, basis2, sparse=False, dtype=np.complex128)
                basis3 = qt.generate.basis.spin_basis_2d(Lx=Lx, Ly=Ly, Nup=4, kxblock=kxblock, kyblock=kyblock)
                mat3 = ham.to_matrix(basis3, pauli=True)
                self.assertTrue(np.allclose(mat2, mat3))
        
    def test_ZNN_proj(self):
        Lx, Ly = 3, 4
        for kxblock in range(Lx):
            for kyblock in range(Ly):
                basis = qs.spin_basis_2d(Lx=Lx, Ly=Ly, pauli=True, kxblock=kxblock, kyblock=kyblock)
                P1 = basis.get_proj(np.complex128)
                basis = qt.generate.basis.spin_basis_2d(Lx=Lx, Ly=Ly, kxblock=kxblock, kyblock=kyblock)
                P2 = basis.projection_matrix()
                self.assertTrue(np.allclose((P1-P2).data, 0))

        Lx, Ly = 3, 4
        N_2d = Lx * Ly
        for kxblock in range(Lx):
            for kyblock in range(Ly):
                basis = qs.spin_basis_2d(Lx=Lx, Ly=Ly, pauli=True, kxblock=kxblock, kyblock=kyblock)
                state = np.random.rand(2**N_2d) + 1j*np.random.rand(2**N_2d)
                state1 = basis.project_to(state, sparse=False)
                basis = qt.generate.basis.spin_basis_2d(Lx=Lx, Ly=Ly, kxblock=kxblock, kyblock=kyblock)
                state2 = basis.project(state).reshape(-1)
                self.assertTrue(np.allclose(state1, state2))

        Lx, Ly = 3, 4
        N_2d = Lx * Ly
        for kxblock in range(Lx):
            for kyblock in range(Ly):
                basis = qs.spin_basis_2d(Lx=Lx, Ly=Ly, pauli=True, kxblock=kxblock, kyblock=kyblock)
                state = np.random.rand(basis.Ns) + 1j*np.random.rand(basis.Ns)
                state1 = basis.project_from(state, sparse=False)

                basis = qt.generate.basis.spin_basis_2d(Lx=Lx, Ly=Ly, kxblock=kxblock, kyblock=kyblock)
                state2 = basis.recover(state).reshape(-1)
                self.assertTrue(np.allclose(state1, state2))
    



if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestGeneralBasis('test_ZNN_proj'))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)