# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-17 17:47:03
# @Last Modified by:   hzhu
# @Last Modified time: 2025-05-17 21:29:20

import quante as qt
import numpy as np
import scipy.sparse as sps
import unittest
import torch as tc
import quante.torch_utils as qtc

class TestQuantity(unittest.TestCase):
    def test_expect(self):
        # Test the expect function with different types of inputs
        d = 100
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        mat += mat.conj().T
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.linalg.expect(mat, state)
        res1 = state.conj() @ (mat @ state)
        self.assertAlmostEqual(res, res1)

        mat = tc.tensor(mat, device='cuda')
        state = tc.tensor(state, device='cuda')
        res = qt.linalg.expect(mat, state)
        res1 = (state.conj() @ (mat @ state)).cpu().numpy()
        self.assertAlmostEqual(res, res1)

        mat = qtc.totc(mat)
        res = qt.linalg.expect(mat, state)
        res1 = (state.conj() @ (mat @ state)).item()
        self.assertAlmostEqual(res, res1)

        d = 100
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.linalg.expect(mat, state)
        res1 = state.conj() @ (mat.toarray() @ state)
        self.assertAlmostEqual(res, res1)

        d = 100
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.linalg.expect(mat, state)
        res1 = state.conj() @ (mat.toarray() @ state)
        self.assertAlmostEqual(res, res1)

        
        # # Test with non-Hermitian matrices
        d = 100
        n = 101
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        mat = tc.tensor(mat, device='cuda')
        states = tc.tensor(states, device='cuda')
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal().cpu().numpy()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        mat = qtc.totc(mat)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ (mat @ states)).diagonal().cpu().numpy()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)



        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) , ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        states = np.random.randn(d,n)
        res = qt.linalg.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        # dm

        d = 100
        n = 100
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        mat = tc.tensor(mat, device='cuda')
        states = tc.tensor(states, device='cuda')
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace().cpu().numpy()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        mat = tc.tensor(mat, device='cuda')
        states = tc.tensor(states, device='cuda')
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = (mat.to(dtype=tc.complex128) @ states).trace().cpu().numpy()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n)
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        mat = tc.tensor(mat, device='cuda')
        states = tc.tensor(states, device='cuda')
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = (mat @ states.to(dtype=tc.complex128)).trace().cpu().numpy()
        self.assertAlmostEqual(res, res1)


        d = 100
        n = 100
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.linalg.expect(mat, states, isdm=False)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        mat = tc.tensor(mat, device='cuda')
        states = tc.tensor(states, device='cuda')
        res = qt.linalg.expect(mat, states, isdm=False)
        res1 = (states.conj().T @ mat.to(dtype=tc.complex128) @ states).diagonal().cpu().numpy()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = np.real_if_close([np.trace(states[:,:,i] @ mat) for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = np.real_if_close([(mat @ states[:,:,i]).trace() for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = np.real_if_close([(mat @ states[:,:,i]).trace() for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        mat = qtc.totc(mat, device='cuda')
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        states = qtc.totc(states, device='cuda')
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.cpu().numpy()) for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = np.random.randn(d,d) 
        mat = qtc.totc(mat, device='cuda')
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        states = qtc.totc(states, device='cuda')
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.cpu().numpy()) for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        mat = qtc.totc(mat, device='cuda')
        states = np.random.randn(d,d,n)
        states = qtc.totc(states, device='cuda')
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = np.real_if_close([np.trace(states[:,:,i].cpu().numpy() @ mat.cpu().numpy()) for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        mat = qtc.totc(mat, device='cuda')
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        states = qtc.totc(states, device='cuda')
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = np.real_if_close([(mat @ states[:,:,i]).trace().item() for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 10
        n = 7
        mat = sps.rand(d,d)
        mat = qtc.totc(mat, device='cuda')
        states = np.random.randn(d,d,n) + 1j * np.random.randn(d,d,n)
        states = qtc.totc(states, device='cuda')
        res = qt.linalg.expect(mat, states, isdm=True)
        res1 = np.real_if_close([(mat.to(tc.complex128) @ states[:,:,i]).trace().item() for i in range(n)])
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)




if __name__ == '__main__':
    unittest.main()
