# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-16 23:25:52
# @Last Modified by:   dzwang
# @Last Modified time: 2025-06-18 13:46:01

import quante as qt
import numpy as np
import scipy.sparse as sps
import unittest
import torch as tc
import quante.torch_utils as qtc

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
    
    def test_expect(self):
        # Test the expect function with different types of inputs
        d = 100
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        mat += mat.conj().T
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.measure.expect(mat, state)
        res1 = state.conj() @ (mat @ state)
        self.assertAlmostEqual(res, res1)
        if tc.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        mat = tc.tensor(mat, device=device)
        state = tc.tensor(state, device=device)
        res = qt.measure.expect(mat, state)
        res1 = (state.conj() @ (mat @ state)).cpu().numpy()
        self.assertAlmostEqual(res, res1)

        mat = qtc.totc(mat)
        res = qt.measure.expect(mat, state)
        res1 = (state.conj() @ (mat @ state)).item()
        self.assertAlmostEqual(res, res1)

        d = 100
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.measure.expect(mat, state)
        res1 = state.conj() @ (mat.toarray() @ state)
        self.assertAlmostEqual(res, res1)

        d = 100
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.measure.expect(mat, state)
        res1 = state.conj() @ (mat.toarray() @ state)
        self.assertAlmostEqual(res, res1)

        
        # # Test with non-Hermitian matrices
        d = 100
        n = 101
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        mat = tc.tensor(mat, device=device)
        states = tc.tensor(states, device=device)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal().cpu().numpy()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        mat = qtc.totc(mat)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ (mat @ states)).diagonal().cpu().numpy()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)



        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) , ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        states = np.random.randn(d,n)
        res = qt.measure.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        # dm

        d = 100
        n = 100
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        mat = tc.tensor(mat, device=device)
        states = tc.tensor(states, device=device)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace().cpu().numpy()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        mat = tc.tensor(mat, device=device)
        states = tc.tensor(states, device=device)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = (mat.to(dtype=tc.complex128) @ states).trace().cpu().numpy()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        mat = tc.tensor(mat, device=device)
        states = tc.tensor(states, device=device)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = (mat @ states.to(dtype=tc.complex128)).trace().cpu().numpy()
        self.assertAlmostEqual(res, res1)


        d = 100
        n = 100
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.measure.expect(mat, states, isdm=False)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        mat = tc.tensor(mat, device=device)
        states = tc.tensor(states, device=device)
        res = qt.measure.expect(mat, states, isdm=False)
        res1 = (states.conj().T @ mat.to(dtype=tc.complex128) @ states).diagonal().cpu().numpy()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = np.real_if_close([np.trace(states[:,:,i] @ mat) for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = np.real_if_close([(mat @ states[:,:,i]).trace() for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = np.real_if_close([(mat @ states[:,:,i]).trace() for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        mat = qtc.totc(mat, device=device)
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        states = qtc.totc(states, device=device)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.cpu().numpy()) for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = np.random.randn(d,d) 
        mat = qtc.totc(mat, device=device)
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        states = qtc.totc(states, device=device)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.cpu().numpy()) for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        mat = qtc.totc(mat, device=device)
        states = np.random.randn(d,d,n)
        states = qtc.totc(states, device=device)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.cpu().numpy()) for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        mat = qtc.totc(mat, device=device)
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        states = qtc.totc(states, device=device)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = np.real_if_close([(mat @ states[:,:,i]).trace().item() for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = sps.rand(d,d)
        mat = qtc.totc(mat, device=device)
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        states = qtc.totc(states, device=device)
        res = qt.measure.expect(mat, states, isdm=True)
        res1 = np.real_if_close([(mat.to(tc.complex128) @ states[:,:,i]).trace().item() for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)



#todo write more tests


if __name__ == '__main__':
    unittest.main()

