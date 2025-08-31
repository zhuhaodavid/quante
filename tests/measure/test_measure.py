# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-16 23:25:52
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 16:55:33

import quante as qt
import numpy as np
import scipy.sparse as sps
import unittest
import torch as tc
from quante.bridge.torch_utils import totc

class TestQuantity(unittest.TestCase):
    def test_entanglement_spectrum(self):
        # Create a simple 2-qubit Bell state |00> + |11>
        state = np.zeros(4, dtype=np.complex128)
        state[0] = 1/np.sqrt(2)
        state[3] = 1/np.sqrt(2)
        # No basis provided, left_number=1 (split between qubits)
        spectrum = qt.measure.entanglement_spectrum(state, left_number=1)
        # The entanglement spectrum should be [1/sqrt(2), 1/sqrt(2)]
        expected = np.array([[1/np.sqrt(2), 1/np.sqrt(2)]])
        # Only the first two singular values are nonzero
        np.testing.assert_allclose(spectrum[:2], expected, rtol=1e-7, atol=1e-8)
        # The rest should be zero
        np.testing.assert_allclose(spectrum[2:], 0, atol=1e-8)
    
    def test_unfolding(self):
        op = qt.generate.operas.spin

        L = 6
        gs = [0.43656223, 0.07111102, 0.11462402, 0.37942382, 0.07833752, 0.12742619]

        builder = op.builder()
        for i in range(L-1):
            builder += 'xx', [i, i+1], 1.
            builder += 'yy', [i, i+1], 1.
            builder += 'zz', [i, i+1], 1.
            builder += 'z', [i,], gs[i]
        ham = builder.build()

        basis = qt.generate.basis.spin_basis(L=L, Nup=L//2)
        mat = ham.to_matrix(basis=basis)
        eng = np.linalg.eigvalsh(mat)

        eng_unfold = qt.measure.unfolding_diff(eng)

        res = np.array([1.25435517, 0.46919587, 1.58384313, 
                        .43021752,1.58826541,0.46563927,1.22822096,
                        1.02044134,0.96412631,1.03219716,0.97022907])

        np.testing.assert_allclose(res, eng_unfold, atol=1e-8)
    
class TestExpect(unittest.TestCase):
    def test_dense_state(self):
        d = 100
        for dtype1 in [np.float64, np.complex128]:
            for dtype2 in [np.float64, np.complex128]:
                mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
                mat = mat.astype(dtype1)
                state = np.random.randn(d) + 1j * np.random.randn(d)
                state = state.astype(dtype2)

                res = qt.measure.expect(mat, state)
                res1 = state.conj() @ (mat @ state)
                self.assertAlmostEqual(res, res1)
                
                device = 'cpu'
                mat = totc(mat, device=device)
                state = totc(state, device=device)
                res = qt.measure.expect(mat, state)
                res1 = (state.to(tc.complex128).conj() @ (mat.to(tc.complex128) @ state.to(tc.complex128))).cpu().numpy()
                self.assertAlmostEqual(res, res1)

                if tc.cuda.is_available():
                    device = tc.cuda.device(0)
                    mat = totc(mat, device=device)
                    state = totc(state, device=device)
                    res = qt.measure.expect(mat, state)
                    res1 = (state.to(tc.complex128).conj() @ (mat.to(tc.complex128) @ state.to(tc.complex128))).cpu().numpy()
                    self.assertAlmostEqual(res, res1)
        

    def test_csr_state(self):
        d = 100
        for dtype1 in [np.float64, np.complex128]:
            for dtype2 in [np.float64, np.complex128]:
                for f in ['csr', 'csc', 'coo']:
                    mat = sps.rand(d,d, format=f, density=0.8) + 1j * sps.rand(d,d, format=f, density=0.8)
                    mat = mat.astype(dtype1)
                    state = np.random.randn(d) + 1j * np.random.randn(d)
                    state = state.astype(dtype2)
                    
                    res = qt.measure.expect(mat, state)
                    res1 = state.conj() @ (mat.toarray() @ state)
                    self.assertAlmostEqual(res, res1)
                
                    device = 'cpu'
                    mat = totc(mat, device=device)
                    state = totc(state, device=device)
                    res = qt.measure.expect(mat, state)
                    res1 = (state.to(tc.complex128).conj() @ (mat.to(tc.complex128) @ state.to(tc.complex128))).cpu().numpy()
                    self.assertAlmostEqual(res, res1)

                    if tc.cuda.is_available():
                        device = tc.cuda.device(0)
                        mat = totc(mat, device=device)
                        state = totc(state, device=device)
                        res = qt.measure.expect(mat, state)
                        res1 = (state.to(tc.complex128).conj() @ (mat.to(tc.complex128) @ state.to(tc.complex128))).cpu().numpy()
                        self.assertAlmostEqual(res, res1)
            
    
    def test_dia_state(self):
        d = 100
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.measure.expect(mat, state)
        res1 = state.conj() @ (mat.toarray() @ state)
        self.assertAlmostEqual(res, res1)
    

    def test_dense_multistate(self):
        d = 100
        n = 101
        for dtype1 in [np.float64, np.complex128]:
            for dtype2 in [np.float64, np.complex128]:
                mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
                mat = mat.astype(dtype1)
                states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
                states = states.astype(dtype2)
                
                res = qt.measure.expect(mat, states)
                res1 = (states.conj().T @ mat @ states).diagonal()
                self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

                device = 'cpu'
                mat = totc(mat, device=device)
                states = totc(states, device=device)
                res = qt.measure.expect(mat, states)
                res1 = (states.to(tc.complex128).conj().T @ mat.to(tc.complex128) @ states.to(tc.complex128)).diagonal().cpu().numpy()
                self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

                if tc.cuda.is_available():
                    device = tc.cuda.device(0)
                    mat = totc(mat, device=device)
                    states = totc(states, device=device)
                    res = qt.measure.expect(mat, states)
                    res1 = (states.to(tc.complex128).conj().T @ mat.to(tc.complex128) @ states.to(tc.complex128)).diagonal().cpu().numpy()
                    self.assertAlmostEqual(np.linalg.norm(res - res1), 0)


    def test_csr_multistate(self):
        d = 100
        n = 101
        for dtype1 in [np.float64, np.complex128]:
            for dtype2 in [np.float64, np.complex128]:
                for f in ['csr', 'csc', 'coo']:
                    mat = sps.rand(d,d, format=f, density=0.8) + 1j * sps.rand(d,d, format=f, density=0.8)
                    mat = mat.astype(dtype1)
                    states = np.random.randn(d,n)
                    states = states.astype(dtype2)
                    
                    res = qt.measure.expect(mat, states)
                    res1 = (states.conj().T @ mat @ states).diagonal()
                    self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

                    device = 'cpu'
                    mat = totc(mat, device=device)
                    states = totc(states, device=device)
                    res = qt.measure.expect(mat, states)
                    res1 = (states.to(tc.complex128).conj().T @ mat.to(tc.complex128) @ states.to(tc.complex128)).diagonal().cpu().numpy()
                    self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

                    if tc.cuda.is_available():
                        device = tc.cuda.device(0)
                        mat = totc(mat, device=device)
                        states = totc(states, device=device)
                        res = qt.measure.expect(mat, states)
                        res1 = (states.to(tc.complex128).conj().T @ mat.to(tc.complex128) @ states.to(tc.complex128)).diagonal().cpu().numpy()
                        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)


    def test_dia_multistate(self):
        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)
   
    def test_dense_dm(self):
        d = 100
        n = 100
        for dtype1 in [np.float64, np.complex128]:
            for dtype2 in [np.float64, np.complex128]:
                mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
                mat = mat.astype(dtype1)
                states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
                states = states.astype(dtype2)
                res = qt.measure.expect(mat, states, isdm=True)
                res1 = (mat @ states).trace()
                self.assertAlmostEqual(res, res1)

                device = 'cpu'
                mat = tc.tensor(mat, device=device)
                states = tc.tensor(states, device=device)
                res = qt.measure.expect(mat, states, isdm=True)
                res1 = (mat.to(tc.complex128) @ states.to(tc.complex128)).trace().cpu().numpy()
                self.assertAlmostEqual(res, res1)

                if tc.cuda.is_available():
                    device = tc.cuda.device(0)
                    mat = tc.tensor(mat, device=device)
                    states = tc.tensor(states, device=device)
                    res = qt.measure.expect(mat, states, isdm=True)
                    res1 = (mat.to(tc.complex128) @ states.to(tc.complex128)).trace().cpu().numpy()
                    self.assertAlmostEqual(res, res1)

    def test_csr_dm(self):
        d = 100
        n = 100
        for dtype1 in [np.float64, np.complex128]:
            for dtype2 in [np.float64, np.complex128]:
                for f in ['csr', 'csc', 'coo']:
                    mat = sps.rand(d,d, format=f, density=0.8) + 1j * sps.rand(d,d, format=f, density=0.8)
                    mat = mat.astype(dtype1)
                    states = np.random.randn(d,n)
                    states = states.astype(dtype2)
                    
                    res = qt.measure.expect(mat, states, isdm=True)
                    res1 = (mat @ states).trace()
                    self.assertAlmostEqual(res, res1)

                    device = 'cpu'
                    mat = totc(mat, device=device)
                    states = totc(states, device=device)
                    res = qt.measure.expect(mat, states, isdm=True)
                    res1 = (mat.to(tc.complex128) @ states.to(tc.complex128)).trace().cpu().numpy()
                    self.assertAlmostEqual(res, res1)

                    if tc.cuda.is_available():
                        device = tc.cuda.device(0)
                        mat = totc(mat, device=device)
                        states = totc(states, device=device)
                        res = qt.measure.expect(mat, states, isdm=True)
                        res1 = (mat.to(tc.complex128) @ states.to(tc.complex128)).trace().cpu().numpy()
                        self.assertAlmostEqual(res, res1)


    def test_dia_dm(self):
        d = 100
        n = 100
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

    def test_dense_multidm(self):
        d = 10
        n = 7
        for dtype1 in [np.float64, np.complex128]:
            for dtype2 in [np.float64, np.complex128]:
                mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
                mat = mat.astype(dtype1)
                states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
                states = states.astype(dtype2)

                res = qt.measure.expect(mat, states, isdm=True)
                res1 = np.real_if_close([np.trace(states[:,:,i] @ mat) for i in range(n)])
                self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

                device = 'cpu'
                mat = totc(mat, device=device)
                states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
                states = totc(states, device=device)
                res = qt.measure.expect(mat, states, isdm=True)
                res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.cpu().numpy()) for i in range(n)])
                self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

                if tc.cuda.is_available():
                    device = tc.cuda.device(0)
                    mat = totc(mat, device=device)
                    states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
                    states = totc(states, device=device)
                    res = qt.measure.expect(mat, states, isdm=True)
                    res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.cpu().numpy()) for i in range(n)])
                    self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

    def test_csr_multidm(self):
        d = 10
        n = 7
        for dtype1 in [np.float64, np.complex128]:
            for dtype2 in [np.float64, np.complex128]:
                for f in ['csr', 'csc', 'coo']:
                    mat = sps.rand(d,d, format=f, density=0.8) + 1j * sps.rand(d,d, format=f, density=0.8)
                    mat = mat.astype(dtype1)
                    states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
                    states = states.astype(dtype2)

                    res = qt.measure.expect(mat, states, isdm=True)
                    res1 = np.real_if_close([np.trace(states[:,:,i] @ mat) for i in range(n)])
                    self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

                    device = 'cpu'
                    mat = totc(mat, device=device)
                    states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
                    states = totc(states, device=device)
                    res = qt.measure.expect(mat, states, isdm=True)
                    res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.to_dense().cpu().numpy()) for i in range(n)])
                    self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

                    if tc.cuda.is_available():
                        device = tc.cuda.device(0)
                        mat = totc(mat, device=device)
                        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
                        states = totc(states, device=device)
                        res = qt.measure.expect(mat, states, isdm=True)
                        res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.to_dense().cpu().numpy()) for i in range(n)])
                        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

    def test_dia_multidm(self):
        d = 10
        n = 7
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = np.real_if_close([(mat @ states[:,:,i]).trace() for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)
