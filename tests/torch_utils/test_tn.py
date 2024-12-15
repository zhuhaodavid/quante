# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-28 21:21:02
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-24 22:58:30


import unittest

from quante.torch_utils import tensor_network
import numpy as np
import torch as tc

from quante.torch_utils.utils import convert_to_torch

class TestTN(unittest.TestCase):
    def test_canonicalize(self):
        # 验证正交化
        N = 4
        linkdims = [1] + [3] * (N-1) + [1]
        ψs1 = [np.random.randn(linkdims[i],2,linkdims[i+1]) + 1j * np.random.randn(linkdims[i],2,linkdims[i+1]) for i in range(N)]

        ψs1 = convert_to_torch(ψs1)
        ψ = tensor_network.MPS(ψs1)
        vec = ψ.to_matrix()

        ψ1 = ψ.copy()
        ψ1.canonicalize_()
        vec1 = ψ1.to_matrix()
        self.assertTrue(tc.allclose(vec1, vec))
        
    def test_add(self):
        # 验证加法
        N = 4
        linkdims = [1] + [3] * (N-1) + [1]
        ψ1 = [np.random.randn(linkdims[i],2,linkdims[i+1]) + 1j * np.random.randn(linkdims[i],2,linkdims[i+1]) for i in range(N)]
        ψ1 = tensor_network.MPS(convert_to_torch(ψ1))
        ψ2 = [np.random.randn(linkdims[i],2,linkdims[i+1]) + 1j * np.random.randn(linkdims[i],2,linkdims[i+1]) for i in range(N)]
        ψ2 = tensor_network.MPS(convert_to_torch(ψ2))

        ψ3 = ψ1 + ψ2

        vec1 = ψ1.to_matrix()
        vec2 = ψ2.to_matrix()
        vec3 = ψ3.to_matrix()
        self.assertTrue(tc.allclose(vec3, vec2+vec1))
        
    def test_apply(self):
        
        import quante.generate.operas as op
        from quante.torch_utils.utils import convert_to_torch
        
        device = tc.device("cpu")
        dtype = tc.complex128

        # 随机生成一个态
        N = 5
        linkdims = [1] + [10] * (N-1) + [1]
        ψs = [np.random.randn(linkdims[i],2,linkdims[i+1]) + 1j * np.random.randn(linkdims[i],2,linkdims[i+1]) for i in range(N)]
        vec1 = tensor_network.full_contract(convert_to_torch(ψs)).numpy()
        
        # 考虑这个算符
        M = op.xx(1,2) + op.yy(1,2)
        from quante.generate import spin_basis
        basis = spin_basis(L=N)
        M = M.to_matrix(basis)

        # 严格作用
        vec2 = M @ vec1

        # 作用前态是相同
        ψs = tensor_network.MPS(convert_to_torch(ψs, device=device, dtype=dtype))
        vec1p = ψs.to_matrix()
        self.assertTrue(np.allclose(vec1p.cpu().numpy(), vec1))

        # 两体门
        lM = op.xx(0,1) + op.yy(0,1)
        basis = spin_basis(L=2)
        lM = lM.to_matrix(basis)
        lM = convert_to_torch([lM.reshape(2,2,2,2)], device=device, dtype=dtype)[0]
    
        # 作用到 MPS 上
        ψs1 = ψs.copy()
        ψs1.apply_gate_(1,lM, trunc_para=(10,1e-5,1e-5))
        vec2p = ψs1.to_matrix()
        self.assertTrue(np.allclose(vec2p.cpu().numpy(), vec2))
        
        
        ψs1 = ψs.copy()
        ψs1.apply_gate_(1,lM, svd_alg='svd', trunc_para=(10,1e-5,1e-5), normalize=False)
        vec2p = ψs1.to_matrix()
        self.assertTrue(np.allclose(vec2p.cpu().numpy(), vec2))
        
        # # 正交作用
        ψs2 = ψs.copy()
        ψs2.canonicalize_()

        ψs2.apply_gate_(1,lM, unitary_gate=True, normalize=False)
        vec3p = ψs2.to_matrix()
        self.assertTrue(np.allclose(vec3p.cpu().numpy(), vec2))

    
    def test_mpo_apply(self):
        # 验证 mpo 两体门的正确性
        import quante.generate.operas as op
        from quante.torch_utils.utils import convert_to_torch
        
        device = tc.device("cpu")
        dtype = tc.complex128

        # 随机一个态
        N = 4
        linkdims = [1] + [3] * (N-1) + [1]
        Ws1 = [np.random.randn(linkdims[i],2,2,linkdims[i+1]) + 1j * np.random.randn(linkdims[i],2,2,linkdims[i+1]) for i in range(N)]
        vec1 = tensor_network.full_contract(convert_to_torch(Ws1)).numpy()

        # 随机一个两体门
        import quante.linalg as qla
        M1 = np.random.randn(4,4)
        eye = np.eye(2)
        lM = qla.kron(eye, M1, eye)

        M2 = np.random.randn(4,4)
        eye = np.eye(2)
        rM = qla.kron(eye, M2, eye)

        # 严格作用
        vec2 = lM.T.conj() @ vec1 @ rM

        # 初始态相同
        lM = convert_to_torch([M1.reshape(2,2,2,2)], device=device, dtype=dtype)[0]
        rM = convert_to_torch([M2.reshape(2,2,2,2)], device=device, dtype=dtype)[0]
        ψs = tensor_network.MPO(convert_to_torch(Ws1, device=device, dtype=dtype))
        vec1p = ψs.to_matrix()
        self.assertTrue(np.allclose(vec1p.cpu().numpy(), vec1))

        # apply_gate_ 作用
        ψs1 = ψs.copy()
        ψs1.apply_gate_(1, ((lM.conj().permute([2,3,0,1]), rM), "topbottom"), svd_alg='eig', direction='right', trunc_para=(12, 1e-10, 1e-10))
        vec2p = ψs1.to_matrix()
        self.assertTrue(np.allclose(vec2p.cpu().numpy(), vec2))


        ψs1 = ψs.copy()
        ψs1.canonicalize_()
        
        ψs1.apply_gate_(1, ((lM.conj().permute([2,3,0,1]), rM), "topbottom"), unitary_gate=True, normalize=True, svd_alg='svd')
        vec3p = ψs1.to_matrix()
        self.assertTrue(np.allclose(vec3p.cpu().numpy(), vec2))
        
    def test_oddeven_decomp(self):
        import quante.generate.operas as op
        from quante.generate import spin_basis
        import quante.linalg as qla
        # 验证奇偶拆分 --- 精确对角化

        L = 10
        tau = 1.  # 不应该依赖 tau

        hz = [ 0.1889499 , -0.08278537,  0.41204622, -0.39731249,  0.38652398,  0.32772005, -0.2914097 , -0.37077033,  0.33506667,  0.21862272,  0.36220858,  0.01096741,  0.08961132, -0.40967383]

        ham = op.sum(op.zz(i,i+1) for i in range(0,L-1,2)) + op.x(0) +\
            op.sum(op.x(i)/2 for i in range(1,L-1)) +\
            op.sum(hz[i]/2 * op.z(i) for i in range(1,L-1) ) + hz[0] * op.z(0)

        if L % 2 == 0:
            ham += op.x(L-1)
            ham += hz[L-1] * op.z(L-1)

        basis = spin_basis(L=L)
        mat = ham.to_matrix(basis, pauli=1)

        evals, evecs = qla.eigh(mat)
        evolve_operator = (evecs * np.exp(- 1j * tau * evals)) @ evecs.conjugate().transpose()

        ham = op.sum(op.zz(i,i+1) for i in range(1,L-1,2)) +\
            op.sum(op.x(i)/2 for i in range(1,L-1)) +\
            op.sum(hz[i]/2 * op.z(i) for i in range(1,L-1) )

        if L % 2 == 1:
            ham += op.x(L-1)
            ham += hz[L-1] * op.z(L-1)

        basis = spin_basis(L=L)
        mat = ham.to_matrix(basis, pauli=1)

        evals, evecs = qla.eigh(mat)
        evolve_operator = (evecs * np.exp(- 1j * tau * evals)) @ evecs.conjugate().transpose() @ evolve_operator

        ham = op.sum(op.zz(i,i+1) for i in range(L-1)) +\
            op.sum(op.x(i) for i in range(L)) +\
            op.sum(hz[i] * op.z(i) for i in range(L) )

        U_tau = tensor_network.MPO(tensor_network.mpo_eye(L, [2]*L))  # 生成单位矩阵

        gates = ham.gate2_decomposition(L, tau=1., form='brick', pauli=1)

        for pos_cur, gate in zip(*gates):
            local_evolve = tc.tensor(gate).reshape(2,2,2,2)
            U_tau.apply_gate_(pos_cur, local_evolve)

        evolve_operator2 = U_tau.to_matrix()

        self.assertTrue(np.allclose(evolve_operator, evolve_operator2.numpy()))

    def test_trotter(self):
        import quante.generate.operas as op
        from quante.generate import spin_basis
        import quante.linalg as qla
        L = 10
        tau = 0.1

        hz = [ 0.1889499 , -0.08278537,  0.41204622, -0.39731249,  0.38652398,  0.32772005, -0.2914097 , -0.37077033,  0.33506667,  0.21862272,  0.36220858,  0.01096741,  0.08961132, -0.40967383]

        ham = op.sum(op.zz(i,i+1) for i in range(L-1)) +\
            op.sum(op.x(i) for i in range(L)) +\
            op.sum(hz[i] * op.z(i) for i in range(L) )

        gates = ham.trotter_gates(L, tau=tau, order='4_opt', pauli=1)

        U_tau = tensor_network.MPO(tensor_network.mpo_eye(L, [2]*L))  # 生成单位矩阵
        pos_list, ugate_list = gates
        i = 1
        for pos, gate in zip(pos_list, ugate_list):
            direction =  'right' if pos < pos_list[i] else "left"
            local_evolve = tc.tensor(gate).reshape(2,2,2,2)
            U_tau.apply_gate_(pos, local_evolve, direction=direction, svd_alg='qr')
            i = (i + 1) % len(pos_list)

        # U_tau.apply_gates_()
        evolve_operator2 = U_tau.to_matrix()

        basis = spin_basis(L=L)
        mat = ham.to_matrix(basis, pauli=1)
        evolve_operator = qla.expm(mat, -1j*tau)
        
        
        self.assertAlmostEqual(np.linalg.norm(evolve_operator - evolve_operator2.numpy()), 7.577970076253801e-06)

    def test_dm(self):
        N = 10

        # gen MPO
        linkdims = [1] + [5] * (N-1) + [1]
        Ws = [tc.randn(linkdims[i],2,2,linkdims[i+1], dtype=tc.complex128) + 1j * tc.randn(linkdims[i],2,2,linkdims[i+1], dtype=tc.complex128) for i in range(N)]

        # gen MPO
        linkdims = [1] + [5] * (N-1) + [1]
        Ws2 = [tc.randn(linkdims[i],2,2,linkdims[i+1], dtype=tc.complex128) + 1j * tc.randn(linkdims[i],2,2,linkdims[i+1], dtype=tc.complex128) for i in range(N)]

        # gen MPS
        linkdims = [1] + [10] * (N-1) + [1]
        ψs = [tc.randn(linkdims[i],2,linkdims[i+1], dtype=tc.complex128) + 1j * tc.randn(linkdims[i],2,linkdims[i+1], dtype=tc.complex128) for i in range(N)]

        # import quante.tensor.densitymatrix_method as dmm
        
        # res = dmm.dm_apply_mpo_on_mps2([i.numpy() for i in Ws2], [ψ.numpy() for ψ in ψs], 0, 1e-10)
        
        

        # import sys
        # sys.path.append(r"D:\OneDrive\备份\PyLib_backup\v1.0\quante")
        
        # from torch_utils import tensor_class as mtn
        Ws = tensor_network.MPO(Ws)
        ψs = tensor_network.MPS(ψs)
        
        ψs.canonicalize_()
        ψs.lognm *= 0.
        Ws.canonicalize_()
        Ws.lognm *= 0.
        
        vec1 = Ws.to_matrix() @ ψs.to_matrix()
        
        ψs.apply_mpo_(Ws)
        vec2 = ψs.to_matrix()
        self.assertTrue(tc.allclose(vec1, vec2))
        
        # Ws2 = mtn.MPO(Ws2)
        Ws2 = tensor_network.MPO(Ws2)
        Ws2.canonicalize_()
        Ws2.lognm *= 0.
        
        mat1 = Ws.to_matrix() @ Ws2.to_matrix()
        err = Ws2.apply_mpo_(Ws)
        mat2 = Ws2.to_matrix()
        self.assertTrue(tc.allclose(mat1, mat2))
        
        
        
        vec1 = Ws.to_matrix() @ ψs.to_matrix()
        
        ψs.apply_mpo_naive_(Ws)
        vec2 = ψs.to_matrix()
        self.assertTrue(tc.allclose(vec1, vec2))
        
        # Ws2 = mtn.MPO(Ws2)
        Ws2.canonicalize_()
        Ws2.lognm *= 0.
        
        mat1 = Ws.to_matrix() @ Ws2.to_matrix()
        Ws2.apply_mpo_naive_(Ws)
        mat2 = Ws2.to_matrix()
        self.assertTrue(tc.allclose(mat1, mat2))

    def test_dmrg(self):
        import quante as qt
        op = qt.generate.operas
        import quante.torch_utils.tensor_network as tensor_network
        import numpy as np
        import torch as tc

        L = 100
        ham = qt.generate.operas.heisenberg_operator(L, j=(1, 1, 1))
        ham = ham.expandxy(pauli=False)
        H = ham.to_mpo(L, pauli=False, backend='torch')
        psi0 = tensor_network.MPS.random(L, linkdims=1, dtype=tc.float64)
        nsweeps = 4
        chi_max = [100] * nsweeps
        trunc_cut = [1E-10] * nsweeps
        eng, _ = H.dmrg(psi0, nsweeps, chi_max, trunc_cut)
        self.assertAlmostEqual(eng.item(), -44.127739870478734)

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_apply"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)