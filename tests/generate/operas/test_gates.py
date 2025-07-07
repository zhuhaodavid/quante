# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-09 19:48:55
# @Last Modified by:   hzhu
# @Last Modified time: 2025-02-02 15:32:35

import unittest


class TestOperas(unittest.TestCase):
    
    def test_gate2_decomposition(self):
        
        import quante as qt
        import numpy as np
        import torch as tc
        import quante.bridge.torch_utils as qtc
        op = qt.generate.operas
        
        # 定义哈密顿量
        L = 10
        tau = 1
        
        jx = [ 0.32725316, -0.96568997,  2.13891087, -1.16919487,  1.2650014 ,
        0.38177732,  1.07316627,  0.77324926, -0.92001495,  0.34511413]
        jy = [ 1.12830153, -0.19211697, -0.94967577,  0.31803172, -1.10886015,
       -1.27451578,  0.94486262,  1.22393333,  0.56887909, -0.25333099]
        jz = [-0.82562608,  0.23959898,  0.38654536,  1.20739345,  0.4576763 ,
       -2.43487163,  0.63990931,  1.08554791,  0.05690342,  0.04692738]
        hx = [ 1.16852895, -0.34021327,  0.76397712,  0.69193161, -0.9026171 ,
       -1.38188541,  1.01474971,  0.56736074,  0.53625989, -2.47836713]
        hy = [-0.62327138, -0.59679828,  1.25496805,  0.59914667, -0.60396922,
       -0.13742651, -1.6067878 ,  1.07789373,  1.33066415, -1.71387809]
        hz = [-0.13803836,  0.7377294 , -1.02106295, -0.12854978, -0.01749027,
       -1.12325084,  1.22301713,  0.16891778,  0.71240084,  1.24899503]
        
        ham = op.sum(jx[i] * op.xx(i,i+1) + jy[i] * op.yy(i,i+1) + jz[i] * op.zz(i,i+1) for i in range(L-1)) + op.sum( hx[i] * op.x(i) + hy[i] * op.y(i) + hz[i] * op.z(i) for i in range(L) )
        
        forms = ["brick", "ladder"]
        res = [18.053909613025848, 20.79898251992355]
        
        for form in forms:
            gates = ham.gate2_decomposition(L, tau=tau, form=form, pauli=False)
            U_tau = qtc.MPO.from_eye(L, [2]*L)
            for i in range(len(gates[0])):
                local_evolve = tc.tensor(gates[1][i]).reshape(2,2,2,2)
                U_tau.apply_gate_(gates[0][i], local_evolve, svd_alg='svd')
            
            basis = qt.generate.basis.spin_basis(L=L)
            mat = ham.to_matrix(basis, sparse=False)
            evolve_operator = qt.linalg.expm( -1j*tau*mat)
            diff = np.linalg.norm(evolve_operator - U_tau.to_matrix().numpy())
            # print(diff, diff2)
            self.assertAlmostEqual(diff, res[forms.index(form)])
    
    
    def test_trotter_gates(self):
        import quante as qt
        import numpy as np
        import torch as tc
        import quante.bridge.torch_utils as qtc
        op = qt.generate.operas
        
        # 定义哈密顿量
        L = 10
        tau = 1
        
        jx = [ 0.32725316, -0.96568997,  2.13891087, -1.16919487,  1.2650014 ,
        0.38177732,  1.07316627,  0.77324926, -0.92001495,  0.34511413]
        jy = [ 1.12830153, -0.19211697, -0.94967577,  0.31803172, -1.10886015,
       -1.27451578,  0.94486262,  1.22393333,  0.56887909, -0.25333099]
        jz = [-0.82562608,  0.23959898,  0.38654536,  1.20739345,  0.4576763 ,
       -2.43487163,  0.63990931,  1.08554791,  0.05690342,  0.04692738]
        hx = [ 1.16852895, -0.34021327,  0.76397712,  0.69193161, -0.9026171 ,
       -1.38188541,  1.01474971,  0.56736074,  0.53625989, -2.47836713]
        hy = [-0.62327138, -0.59679828,  1.25496805,  0.59914667, -0.60396922,
       -0.13742651, -1.6067878 ,  1.07789373,  1.33066415, -1.71387809]
        hz = [-0.13803836,  0.7377294 , -1.02106295, -0.12854978, -0.01749027,
       -1.12325084,  1.22301713,  0.16891778,  0.71240084,  1.24899503]
        
        ham = op.sum(jx[i] * op.xx(i,i+1) + jy[i] * op.yy(i,i+1) + jz[i] * op.zz(i,i+1) for i in range(L-1)) + op.sum( hx[i] * op.x(i) + hy[i] * op.y(i) + hz[i] * op.z(i) for i in range(L) )
        
        orders = ["1", '2', '4', '4_opt']
        res = [18.053909613025848, 9.227232495242529, 1.591992259253265, 0.08184911108516048] # 0.08184903933355117
        # res = [18.053909613025848, 9.227232495242529, 1.591992259253265, 0.08184903933355117]
        
        # 生成门
        for i in range(len(orders)):
            gates = ham.trotter_gates(L, tau=tau, order=orders[i], evolve_type='time', pauli=False)
            
            # 使用案例
            U_tau = qtc.MPO.from_eye(L, [2]*L)
            for pos_cur, gate in zip(*gates):
                local_evolve = tc.tensor(gate).reshape(2,2,2,2)
                U_tau.apply_gate_(pos_cur, local_evolve, svd_alg='eig')
            
            # 验证程序
            basis = qt.generate.basis.spin_basis(L=L)
            mat = ham.to_matrix(basis, sparse=False)
            evolve_operator = qt.linalg.expm( -1j*tau*mat)
            diff = np.linalg.norm(evolve_operator - U_tau.to_matrix().numpy())
            self.assertAlmostEqual(diff, res[i])
            
        # test N_step = 3
        res = [20.877620270868896, 14.737143777596243, 3.517077361878826, 0.1316388963604957]
        # 生成门
        for i in range(len(orders)):
            gates = ham.trotter_gates(L, tau=tau, order=orders[i], evolve_type='time', pauli=False, N_step=3)
            
            # 使用案例
            U_tau = qtc.MPO.from_eye(L, [2]*L)
            for pos_cur, gate in zip(*gates):
                local_evolve = tc.tensor(gate).reshape(2,2,2,2)
                U_tau.apply_gate_(pos_cur, local_evolve, svd_alg='eig')
            
            # 验证程序
            basis = qt.generate.basis.spin_basis(L=L)
            mat = ham.to_matrix(basis, sparse=False)
            evolve_operator = qt.linalg.expm( -1j*tau*3*mat)
            diff = np.linalg.norm(evolve_operator - U_tau.to_matrix().numpy())
            self.assertAlmostEqual(diff, res[i])

if __name__ == "__main__":
   unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestOperas("test_gate2_decomposition"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)