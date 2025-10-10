# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-24 12:29:22
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-02 22:17:01

import unittest
import numpy as np
import quante as qt
import quante.bridge.quspin_utils as qs
op = qt.generate.operas
from quante.generate.basis import spin_basis_general
from quante.generate.basis.spin_half.spin_general.basis import *
from quante.generate.basis.spin_half.spin_general.basis_core import *

class TestLiouvilleSuperBasis(unittest.TestCase):
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
                basis = spin_basis_general(2*self.L, pblock=(self.perm, block))
                m1 = self.liou_m.to_matrix(basis, pauli=True, sparse=False)
                basis_full = qs.spin_basis_2d(self.L, 2, pauli=True, pxblock=block)
                m2 = qs.hamiltonian(self.liou_m, basis=basis_full, dtype=np.complex128, sparse=False)
                self.assertTrue(np.allclose(self._sorted_eigs(m1), self._sorted_eigs(m2)))

    def test_flip_ndiff_equivalence_scalar(self):
        for ndiff in range(-self.L//2, self.L//2+1):
            with self.subTest(Ndiff=ndiff):
                basis = spin_basis_general(2*self.L, Ndiff=(np.arange(self.L, 2*self.L), ndiff))
                m1 = self.liou_m.to_matrix(basis, pauli=True, sparse=False)
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
                    basis = spin_basis_general(2*self.L, Ndiff=(np.arange(self.L, 2*self.L), ndiff), Nup=Nup)
                    m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
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
                    basis = spin_basis_general(
                        2*self.L,
                        Ndiff=(np.arange(self.L, 2*self.L), ndiff), pblock=(self.perm, block)
                    )
                    m1 = self.liou_m.to_matrix(basis, pauli=True, sparse=False)
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
                    # BasisNdiffZ2N
                    basis2 = BasisZ2N(
                        2*self.L, flipset=np.arange(self.L,2*self.L),
                        Ndiff=ndiff, Nup2=None,pblock=(self.perm, block)
                    )
                    m3 = self.liou_m.to_matrix(basis2, pauli=True, sparse=False)
                    self.assertTrue(np.allclose(self._sorted_eigs(m3), m2e))

    def test_z_nup2_pblock_equivalence(self):
        for ndiff in range(-self.L//2, self.L//2+1):
            for block in [0, 1]:
                with self.subTest(Ndiff=ndiff, pblock=block):
                    res = []
                    for Nup in range(ndiff,2*self.L+1-ndiff,2):
                        basis = spin_basis_general(2*self.L, Ndiff=(np.arange(self.L, 2*self.L), ndiff), Nup=Nup, pblock=(self.perm, block))
                        m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
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
                    basis = spin_basis_general(
                        2*self.L, pblock=(self.perm, pb), zblock=(self.zperm, zb)
                    )
                    m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                    self.assertTrue(np.allclose(self._sorted_eigs(m1), e_ref))
                    basis2 = BasisZ2N(
                        2*self.L, None, None,None,
                        pblock=(self.perm, pb), zblock=(self.zperm, zb)
                    )
                    m2 = self.liou_z.to_matrix(basis2, pauli=True, sparse=False)
                    self.assertTrue(np.allclose(self._sorted_eigs(m2), e_ref))

    def test_flip_p_z_blocks(self):
        for ndiff in range(0, self.L//2+1):
            Ndiff = 0 if ndiff == 0 else [ndiff, -ndiff]
            Nup = self.L if ndiff == 0 else [self.L-ndiff, self.L+ndiff]
            for pb in [0, 1]:
                for zb in [0, 1]:
                    with self.subTest(ndiff=ndiff, pblock=pb, zblock=zb):
                        basis = spin_basis_general(
                            2*self.L, Ndiff=(np.arange(self.L,2*self.L), Ndiff), pblock=(self.perm, pb), zblock=(self.zperm, zb)
                        )
                        m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
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
                        basis2 = BasisZ2N(
                            2*self.L, flipset=np.arange(self.L,2*self.L),
                            Ndiff=Ndiff, Nup2=None, pblock=(self.perm, pb), zblock=(self.zperm, zb)
                        )
                        m3 = self.liou_z.to_matrix(basis2, pauli=True, sparse=False)
                        self.assertTrue(np.allclose(self._sorted_eigs(m3), e2))

    def test_z_nup2_p_z_block_equivalence(self):
        for ndiff in range(0, self.L//2+1):
            for pb in [0, 1]:
                for zb in [0, 1]:
                    with self.subTest(Ndiff=ndiff, pblock=pb, zblock=zb):
                        res = []
                        for Nup in range(ndiff,(2*self.L-ndiff)//2+2,2):
                            basis = spin_basis_general(2*self.L, Ndiff=(np.arange(self.L, 2*self.L), list(set([ndiff,-ndiff]))), Nup=list(set([Nup, 2*self.L-Nup])), pblock=(self.perm, pb), zblock=(self.zperm, zb))
                            m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
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
                        basis_gen = spin_basis_general(
                            2*self.L,
                            pxblock=(self.permx, b0),
                            pyblock=(self.permy, b1),
                            zblock=(self.zperm, b2)
                        )
                        m1 = self.liou_z.to_matrix(basis_gen, pauli=True, sparse=False)
                        e1 = self._sorted_eigs(m1)
                        basis2 = qs.spin_basis_2d(
                            Lx=self.L, Ly=2, pauli=True,
                            pxblock=b0, pyblock=b1, zblock=b2
                        )
                        m2 = qs.hamiltonian(self.liou_z, basis=basis2, dtype=np.complex128, sparse=False)
                        # m2 = self.liou_z.to_matrix(basis2, pauli=True, sparse=False)
                        e2 = self._sorted_eigs(m2)
                        self.assertTrue(np.allclose(e1, e2))
                        basis3 = BasisZ2N(
                            2*self.L, None, None,None,
                            pxblock=(self.permx, b0),
                            pyblock=(self.permy, b1),
                            zblock=(self.zperm, b2)
                        )
                        m3 = self.liou_z.to_matrix(basis3, pauli=True, sparse=False)
                        self.assertTrue(np.allclose(self._sorted_eigs(m3), e2))

    def test_flip_px_py_z_blocks(self):
        for ndiff in range(0, self.L//2+1):
            Ndiff = 0 if ndiff == 0 else [ndiff, -ndiff]
            Nup = self.L if ndiff == 0 else [self.L-ndiff, self.L+ndiff]
            for b0 in [0, 1]:
                for b1 in [0, 1]:
                    for b2 in [0, 1]:
                        with self.subTest(ndiff=ndiff, px=b0, py=b1, z=b2):
                            basis = spin_basis_general(
                                2*self.L, Ndiff=(np.arange(self.L,2*self.L), Ndiff),
                                pxblock=(self.permx, b0),
                                pyblock=(self.permy, b1),
                                zblock=(self.zperm, b2)
                            )
                            m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
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
                            basis2 = BasisZ2N(
                                2*self.L, flipset=np.arange(self.L,2*self.L),
                                Ndiff=Ndiff,Nup2=None,
                                pxblock=(self.permx, b0),
                                pyblock=(self.permy, b1),
                                zblock=(self.zperm, b2)
                            )
                            m3 = self.liou_z.to_matrix(basis2, pauli=True, sparse=False)
                            self.assertTrue(np.allclose(self._sorted_eigs(m3), e2))

    def test_z_nup2_px_py_z_block_equivalence(self):
        for ndiff in range(0, self.L//2+1):
            for b0 in [0, 1]:
                for b1 in [0, 1]:
                    for b2 in [0, 1]:
                        with self.subTest(ndiff=ndiff, px=b0, py=b1, z=b2):
                            res = []
                            for Nup in range(ndiff,(2*self.L-ndiff)//2+2,2):
                                basis = spin_basis_general(
                                    2*self.L, Ndiff=(np.arange(self.L, 2*self.L), list(set([ndiff,-ndiff]))), 
                                    Nup=list(set([Nup, 2*self.L-Nup])), 
                                    pxblock=(self.permx, b0),
                                    pyblock=(self.permy, b1),
                                    zblock=(self.zperm, b2)
                                )
                                m1 = self.liou_z.to_matrix(basis, pauli=True, sparse=False)
                                res.append(self._sorted_eigs(m1))
                            eng1 = qt.linalg.sortcomplex(np.hstack(res))
                            basis_flip = qs.spin_basis_2d(
                                Lx=self.L, Ly=2, Nup=list(set([self.L-ndiff, self.L+ndiff])), pauli=True,
                                pxblock=b0, pzyblock=b1, zblock=b2
                            )
                            m2 = qs.hamiltonian(self.liou_z_flip, basis=basis_flip, dtype=np.complex128, sparse=False)
                            eng2 = self._sorted_eigs(m2)
                            self.assertTrue(np.allclose(eng1, eng2))


if __name__ == "__main__":
    unittest.main()

