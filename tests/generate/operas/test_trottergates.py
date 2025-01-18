# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-28 21:21:02
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-18 16:54:25

import unittest

import quante as qt
import quante.torch_utils as qtc
op = qt.generate.operas
spin_basis = qt.generate.basis.spin_basis

import numpy as np
import torch as tc

class TestTN(unittest.TestCase):
    def test_oddeven_decomp(self):
        # 验证奇偶拆分 --- 精确对角化

        L = 10
        tau = 1.  # 不应该依赖 tau

        hz = [ 0.1889499 , -0.08278537,  0.41204622, -0.39731249,  0.38652398,  0.32772005, -0.2914097 , -0.37077033,  0.33506667,  0.21862272,  0.36220858,  0.01096741,  0.08961132, -0.40967383]

        builder = op.SpinOperBuilder()
        for i in range(0, L-1, 2):
            builder += 1., 'z', i, 'z', i+1
        for i in range(1, L-1):
            builder += 0.5, 'x', i
            builder += hz[i]/2, 'z', i
        builder += 1., 'x', 0
        builder += hz[0], 'z', 0
        if L % 2 == 0:
            builder += 1., 'x', L-1
            builder += hz[L-1], 'z', L-1
        ham = builder.build()

        basis = spin_basis(L=L)
        mat = ham.to_matrix(basis, pauli=1)
        evals, evecs = qt.linalg.eigh(mat)
        evolve_operator = ((evecs * np.exp(- 1j * tau * evals)) 
                           @ evecs.conjugate().transpose())

        builder = op.SpinOperBuilder()
        for i in range(1, L-1, 2):
            builder += 1., 'z', i, 'z', i+1
        for i in range(1, L-1):
            builder += 0.5, 'x', i
            builder += hz[i]/2, 'z', i
        if L % 2 == 1:
            builder += 1., 'x', L-1
            builder += hz[L-1], 'z', L-1
        ham = builder.build()

        basis = spin_basis(L=L)
        mat = ham.to_matrix(basis, pauli=1)

        evals, evecs = qt.linalg.eigh(mat)
        evolve_operator = (evecs * np.exp(- 1j * tau * evals)) @ evecs.conjugate().transpose() @ evolve_operator

        ham = op.sum(op.zz(i,i+1) for i in range(L-1)) +\
            op.sum(op.x(i) for i in range(L)) +\
            op.sum(hz[i] * op.z(i) for i in range(L) )

        U_tau = qtc.MPO.from_eye(L)  # 生成单位矩阵

        gates = ham.gate2_decomposition(L, tau=1., form='brick', pauli=1)

        for pos_cur, gate in zip(*gates):
            U_tau.apply_gate_(pos_cur, gate)

        evolve_operator2 = U_tau.to_matrix()

        self.assertTrue(np.allclose(evolve_operator, evolve_operator2.numpy()))

    def test_trotter(self):
        L = 10
        tau = 0.1

        hz = [ 0.1889499 , -0.08278537,  0.41204622, -0.39731249,  0.38652398,  0.32772005, -0.2914097 , -0.37077033,  0.33506667,  0.21862272,  0.36220858,  0.01096741,  0.08961132, -0.40967383]

        ham = op.sum(op.zz(i,i+1) for i in range(L-1)) +\
            op.sum(op.x(i) for i in range(L)) +\
            op.sum(hz[i] * op.z(i) for i in range(L) )

        gates = ham.trotter_gates(L, tau=tau, order='4_opt', pauli=1)

        U_tau = qtc.MPO.from_eye(L)  # 生成单位矩阵
        pos_list, ugate_list = gates
        i = 1
        for pos, gate in zip(pos_list, ugate_list):
            direction =  'right' if pos < pos_list[i] else "left"
            U_tau.apply_gate_(pos, gate, direction=direction, svd_alg='qr')
            i = (i + 1) % len(pos_list)

        evolve_operator2 = U_tau.to_matrix()

        basis = spin_basis(L=L)
        mat = ham.to_matrix(basis, pauli=1)
        evolve_operator = qt.linalg.expm(mat, -1j*tau)
        
        
        self.assertAlmostEqual(np.linalg.norm(evolve_operator - evolve_operator2.numpy()), 7.577970076253801e-06)

       
if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_dm"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
