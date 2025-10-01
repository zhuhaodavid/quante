# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-24 14:05:46
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-30 20:22:24

import unittest
import numpy as np
import quante as qt
import quante.generate.operas.spin as op
import quante.bridge.quspin_utils as qs
from quante.generate.basis.spin_half.bitsoperation import count_tot_down

class TestLiouvillianDecomposition(unittest.TestCase):
    L = 4
    J = 1.0
    Δ = 0.5
    gamma = 1.0

    @staticmethod
    def _ham(L, J, Δ):
        return op.sum(J * (op.xx(i, i+1) + op.yy(i, i+1) + Δ * op.zz(i, i+1)) for i in range(L-1))

    @staticmethod
    def _lind_ops(L, gamma):
        return [np.sqrt(gamma) * op.m(i) for i in range(L)]

    @staticmethod
    def _iLiou_antisym(L, J, Δ):
        return (
            op.sum(J*(op.xx(i,i+1)+op.yy(i,i+1)+Δ*op.zz(i,i+1)) for i in range(L-1))
            - op.sum(J*(op.xx(i,i+1)+op.yy(i,i+1)+Δ*op.zz(i,i+1)) for i in range(L, 2*L-1))
        )

    @staticmethod
    def _Liou_sym(L, gamma):
        return (
            op.sum(gamma*op.m(i)*op.m(i+L) for i in range(L))
            - 0.5*op.sum(gamma*op.p(i)*op.m(i) for i in range(L))
            - 0.5*op.sum(gamma*op.p(i)*op.m(i) for i in range(L, 2*L))
        )

    @staticmethod
    def _iLiou_flip_antisym(L, J, Δ):
        # 与 _iLiou_antisym 形式相同，这里保留语义区分
        return TestLiouvillianDecomposition._iLiou_antisym(L, J, Δ)

    @staticmethod
    def _Liou_flip_sym(L, gamma):
        return (
            op.sum(gamma*op.m(i)*op.p(i+L) for i in range(L))
            - 0.5*op.sum(gamma*op.p(i)*op.m(i) for i in range(L))
            - 0.5*op.sum(gamma*op.m(i)*op.p(i) for i in range(L, 2*L))
        )

    @staticmethod
    def _op_list_from_antisym(iLiou_antisym):
        return [[opnm, j, i] for opnm, coef in iLiou_antisym.to_quspin(pauli=True) for i, *j in coef]

    def test_full_liouvillian_equivalence(self):
        L, J, Δ, gamma = self.L, self.J, self.Δ, self.gamma
        iLiou_antisym = self._iLiou_antisym(L, J, Δ)
        Liou_sym = self._Liou_sym(L, gamma)
        Liou = Liou_sym - 1j * iLiou_antisym
        basis_full = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True)
        Lioumat = qs.hamiltonian(Liou, basis=basis_full, dtype=np.complex128).toarray()
        Lioumat_real = np.real_if_close(Lioumat)
        eng_original = qt.linalg.sortcomplex(np.linalg.eigvals(Lioumat))
        self.assertEqual(Lioumat_real.dtype, np.complex128)
        ham = self._ham(L, J, Δ)
        lind_ops = self._lind_ops(L, gamma)
        mat = qs.liouvillian(L, ham, lind_ops, basis_full).toarray()
        self.assertTrue(np.allclose(mat, Lioumat))
        # 保存供后续测试复用（缓存)
        self._cache = {
            "basis_full": basis_full,
            "Liou": Liou,
            "Lioumat": Lioumat,
            "eng_original": eng_original,
            "iLiou_antisym": iLiou_antisym,
            "Liou_sym": Liou_sym
        }

    def test_sym_antisym_real_transform(self):
        if not hasattr(self, "_cache"):
            self.test_full_liouvillian_equivalence()
        L, Lioumat, eng_original = self.L, self._cache["Lioumat"], self._cache["eng_original"]
        basis_full = self._cache["basis_full"]
        basis_sym = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, pyblock=0)
        basis_antisym = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, pyblock=1)
        P_sym = basis_sym.get_proj(np.complex128).toarray()
        P_antisym = 1j * basis_antisym.get_proj(np.complex128).toarray()
        P = np.hstack((P_sym, P_antisym))
        Lioumat_real = np.real_if_close(P.conj().T @ Lioumat @ P)
        eng_after = qt.linalg.sortcomplex(np.linalg.eigvals(Lioumat_real))
        self.assertEqual(Lioumat_real.dtype, np.float64)
        self.assertTrue(np.allclose(eng_original, eng_after))
        # 分块构造
        Liou_sym = self._cache["Liou_sym"]
        iLiou_antisym = self._cache["iLiou_antisym"]
        mat00 = qs.hamiltonian(Liou_sym, basis=basis_sym, dtype=np.float64).toarray()
        mat11 = qs.hamiltonian(Liou_sym, basis=basis_antisym, dtype=np.float64).toarray()
        op_list = self._op_list_from_antisym(iLiou_antisym)
        v_out_full = basis_full.inplace_Op(P_sym, op_list, np.complex128)
        mat10 = np.real_if_close(-1j * P_antisym.T.conj().dot(v_out_full))
        v_out_full = basis_full.inplace_Op(P_antisym, op_list, np.complex128)
        mat01 = np.real_if_close(-1j * P_sym.T.conj().dot(v_out_full))
        mat = np.block([[mat00, mat01], [mat10, mat11]])
        self.assertTrue(np.allclose(Lioumat_real, mat))

    def test_sym_antisym_blocks_op_shift_sector(self):
        if not hasattr(self, "_cache"):
            self.test_full_liouvillian_equivalence()
        L = self.L
        iLiou_antisym = self._cache["iLiou_antisym"]
        Liou_sym = self._cache["Liou_sym"]
        basis_sym = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, pyblock=0)
        basis_antisym = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, pyblock=1)
        P_sym = basis_sym.get_proj(np.complex128).toarray()
        P_antisym = 1j * basis_antisym.get_proj(np.complex128).toarray()
        P = np.hstack((P_sym, P_antisym))
        Lioumat = self._cache["Lioumat"]
        Lioumat_real = np.real_if_close(P.conj().T @ Lioumat @ P)
        mat00 = qs.hamiltonian(Liou_sym, basis=basis_sym, dtype=np.float64).toarray()
        mat11 = qs.hamiltonian(Liou_sym, basis=basis_antisym, dtype=np.float64).toarray()
        op_list = self._op_list_from_antisym(iLiou_antisym)
        v_in = np.eye(basis_antisym.Ns, dtype=np.float64)
        mat01 = np.zeros((basis_sym.Ns, basis_antisym.Ns), dtype=np.float64)
        basis_sym.Op_shift_sector(basis_antisym, op_list, v_in=v_in, v_out=mat01, dtype=np.float64)
        v_in = np.eye(basis_sym.Ns, dtype=np.float64)
        mat10 = np.zeros((basis_antisym.Ns, basis_sym.Ns), dtype=np.float64)
        basis_antisym.Op_shift_sector(basis_sym, op_list, v_in=v_in, v_out=mat10, dtype=np.float64)
        mat10 = -mat10.real
        mat = np.block([[mat00, mat01], [mat10, mat11]])
        self.assertTrue(np.allclose(Lioumat_real, mat))

    def test_super_basis_equivalence(self):
        L, J, Δ, gamma = self.L, self.J, self.Δ, self.gamma
        ham = self._ham(L, J, Δ)
        lind_ops = self._lind_ops(L, gamma)
        basis_super = qs.spin_super_basis(L, pauli=True)
        mat1 = qs.liouvillian(L, ham, lind_ops, basis_super).toarray()
        eng1 = qt.linalg.sortcomplex(np.linalg.eigvals(mat1))
        self.assertEqual(mat1.dtype, np.float64)
        basis_super_snake = qs.spin_super_basis(L, pauli=True, indx_order='snake')
        mat2 = qs.liouvillian(L, ham, lind_ops, basis_super_snake, indx_order='snake').toarray()
        eng2 = qt.linalg.sortcomplex(np.linalg.eigvals(mat2))
        self.assertTrue(np.allclose(eng1, eng2))
        basis_px = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, pxblock=0)
        mat_px = qs.liouvillian(L, ham, lind_ops, basis_px).toarray()
        eng_px = qt.linalg.sortcomplex(np.linalg.eigvals(mat_px))
        basis_super_p = qs.spin_super_basis(L, pauli=True, pblock=0)
        mat_p = qs.liouvillian(L, ham, lind_ops, basis_super_p).toarray()
        eng_p = qt.linalg.sortcomplex(np.linalg.eigvals(mat_p))
        self.assertTrue(np.allclose(eng_px, eng_p))

    def test_flip_liouvillian(self):
        if not hasattr(self, "_cache"):
            self.test_full_liouvillian_equivalence()
        L, J, Δ, gamma = self.L, self.J, self.Δ, self.gamma
        iLiou_flip_antisym = self._iLiou_flip_antisym(L, J, Δ)
        Liou_flip_sym = self._Liou_flip_sym(L, gamma)
        Liou_flip = Liou_flip_sym - 1j * iLiou_flip_antisym
        basis_full = self._cache["basis_full"]
        Lioumat_flip = np.real_if_close(qs.hamiltonian(Liou_flip, basis=basis_full, dtype=np.complex128).toarray())
        eng_flip = qt.linalg.sortcomplex(np.linalg.eigvals(Lioumat_flip))
        eng_original = self._cache["eng_original"]
        self.assertTrue(np.allclose(eng_original, eng_flip))
        basis_Nup = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, Nup=L)
        Lioumat_flip_Nup = np.real_if_close(qs.hamiltonian(Liou_flip, basis=basis_Nup, dtype=np.complex128).toarray())
        eng_Nup = qt.linalg.sortcomplex(np.linalg.eigvals(Lioumat_flip_Nup))
        self.assertTrue(all(any(np.isclose(e, e0) for e0 in eng_flip) for e in eng_Nup))
        ham = self._ham(L, J, Δ)
        lind_ops = self._lind_ops(L, gamma)
        mat_full = qs.liouvillian(L, ham, lind_ops, basis_full, flip=True).toarray()
        self.assertTrue(np.allclose(Lioumat_flip, mat_full))
        mat_Nup = qs.liouvillian(L, ham, lind_ops, basis_Nup, flip=True).toarray()
        self.assertTrue(np.allclose(Lioumat_flip_Nup, mat_Nup))
        # 缓存
        self._cache.update({
            "Liou_flip": Liou_flip,
            "Lioumat_flip": Lioumat_flip,
            "Liou_flip_sym": Liou_flip_sym,
            "iLiou_flip_antisym": iLiou_flip_antisym
        })

    def test_flip_half_filling_sym_antisym_blocks(self):
        if "Liou_flip" not in getattr(self, "_cache", {}):
            self.test_flip_liouvillian()
        L = self.L
        Lioumat_flip = self._cache["Lioumat_flip"]
        Liou_flip_sym = self._cache["Liou_flip_sym"]
        iLiou_flip_antisym = self._cache["iLiou_flip_antisym"]
        basis_Nup_sym = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, Nup=L, pzyblock=0)
        basis_Nup_antisym = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, Nup=L, pzyblock=1)
        P_sym = basis_Nup_sym.get_proj(np.complex128).toarray()
        P_antisym = 1j * basis_Nup_antisym.get_proj(np.complex128).toarray()
        P = np.hstack((P_sym, P_antisym))
        Liou_flip_real = np.real_if_close(P.conj().T @ Lioumat_flip @ P)
        eng_from_block = qt.linalg.sortcomplex(np.linalg.eigvals(Liou_flip_real))
        eng_ref = qt.linalg.sortcomplex(np.linalg.eigvals(np.real_if_close(
            qs.hamiltonian(self._cache["Liou_flip"], basis=qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, Nup=L), dtype=np.complex128).toarray()
        )))
        self.assertEqual(Liou_flip_real.dtype, np.float64)
        self.assertTrue(np.allclose(eng_from_block, eng_ref))
        mat00 = qs.hamiltonian(Liou_flip_sym, basis=basis_Nup_sym, dtype=np.float64).toarray()
        mat11 = qs.hamiltonian(Liou_flip_sym, basis=basis_Nup_antisym, dtype=np.float64).toarray()
        op_list = self._op_list_from_antisym(iLiou_flip_antisym)
        basis_full = self._cache["basis_full"]
        v_out_full = basis_full.inplace_Op(P_sym, op_list, np.complex128)
        mat10 = np.real_if_close(-1j * P_antisym.T.conj().dot(v_out_full))
        v_out_full = basis_full.inplace_Op(P_antisym, op_list, np.complex128)
        mat01 = np.real_if_close(-1j * P_sym.T.conj().dot(v_out_full))
        self.assertTrue(np.allclose(np.block([[mat00, mat01], [mat10, mat11]]), Liou_flip_real))
        # Op_shift_sector
        v_in = np.eye(basis_Nup_antisym.Ns, dtype=np.float64)
        blk01 = np.zeros((basis_Nup_sym.Ns, basis_Nup_antisym.Ns), dtype=np.float64)
        basis_Nup_sym.Op_shift_sector(basis_Nup_antisym, op_list, v_in=v_in, v_out=blk01, dtype=np.float64)
        v_in = np.eye(basis_Nup_sym.Ns, dtype=np.float64)
        blk10 = np.zeros((basis_Nup_antisym.Ns, basis_Nup_sym.Ns), dtype=np.float64)
        basis_Nup_antisym.Op_shift_sector(basis_Nup_sym, op_list, v_in=v_in, v_out=blk10, dtype=np.float64)
        blk10 = -blk10.real
        self.assertTrue(np.allclose(np.block([[mat00, blk01], [blk10, mat11]]), Liou_flip_real))
        ham = self._ham(L, self.J, self.Δ)
        lind_ops = self._lind_ops(L, self.gamma)
        basis_real = qs.spin_super_basis(L, pauli=True, Nup=L, flip=True)
        mat_liou = qs.liouvillian(L, ham, lind_ops, basis_real, flip=True).toarray()
        self.assertTrue(np.allclose(Liou_flip_real, mat_liou))

    def test_non_half_filling_blocks(self):
        if "Liou_flip" not in getattr(self, "_cache", {}):
            self.test_flip_liouvillian()
        L = self.L
        Lioumat_flip = self._cache["Lioumat_flip"]
        Ndiff = 1
        basis_Nup_m = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, Nup=L-Ndiff)
        basis_Nup_p = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, Nup=L+Ndiff)
        P_m = basis_Nup_m.get_proj(np.complex128).toarray()
        P_p = basis_Nup_p.get_proj(np.complex128).toarray()
        P = np.hstack((P_m, P_p))
        Liou_sub = P.conj().T @ Lioumat_flip @ P
        mat01 = np.real_if_close(Liou_sub[basis_Nup_m.Ns:, :basis_Nup_m.Ns])
        mat10 = np.real_if_close(Liou_sub[:basis_Nup_m.Ns, basis_Nup_m.Ns:])
        self.assertTrue(np.allclose(mat01, 0.0))
        self.assertTrue(np.allclose(mat10, 0.0))
        basis_sym = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, Nup=[L-Ndiff, L+Ndiff], pzyblock=0)
        basis_antisym = qs.spin_basis_2d(Lx=L, Ly=2, pauli=True, Nup=[L-Ndiff, L+Ndiff], pzyblock=1)
        P_sym = P.conj().T @ basis_sym.get_proj(np.complex128).toarray()
        P_antisym = 1j * P.conj().T @ basis_antisym.get_proj(np.complex128).toarray()
        P_real = np.hstack((P_sym, P_antisym))
        Liou_real = np.real_if_close(P_real.conj().T @ Liou_sub @ P_real)
        self.assertEqual(Liou_real.dtype, np.float64)
        ham = self._ham(L, self.J, self.Δ)
        lind_ops = self._lind_ops(L, self.gamma)
        basis_super = qs.spin_super_basis(L, pauli=True, Nup=[L-Ndiff, L+Ndiff], flip=True)
        mat_ref = qs.liouvillian(L, ham, lind_ops, basis_super, flip=True).toarray()
        self.assertTrue(np.allclose(Liou_real, mat_ref))
        sign_list = (-1)**(np.vectorize(count_tot_down)(basis_antisym._basis) == L-Ndiff)
        diag = np.vectorize(lambda x: (x >> L) == (x & ((1 << L) - 1)))(basis_sym._basis)
        P_sym2 = P.conj().T @ basis_sym.get_proj(np.complex128).toarray()
        P_antisym2 = 1j * P.conj().T @ (sign_list * basis_antisym.get_proj(np.complex128).toarray())
        P_real2 = np.hstack((P_sym2, P_antisym2))
        Liou_real2 = np.real_if_close(P_real2.conj().T @ Liou_sub @ P_real2)
        self.assertEqual(Liou_real2.dtype, np.float64)
        n0 = basis_sym.Ns
        mat01b = Liou_real2[:n0, n0:]
        mat10b = Liou_real2[n0:, :n0]
        mat00b = Liou_real2[:n0, :n0]
        mat11b = Liou_real2[n0:, n0:]
        self.assertTrue(np.allclose(mat01b, mat01b.T))
        self.assertTrue(np.allclose(mat10b, mat10b.T))
        self.assertTrue(np.allclose(mat01b, -mat10b))
        self.assertTrue(np.allclose(mat00b, mat11b))
        # 避免未使用变量告警
        _ = (diag is not None)

if __name__ == "__main__":
    unittest.main()


