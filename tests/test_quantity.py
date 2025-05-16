# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-16 23:25:52
# @Last Modified by:   hzhu
# @Last Modified time: 2025-05-16 23:51:24

import quante as qt
import numpy as np
import scipy.sparse as sps
import unittest

class TestQuantity(unittest.TestCase):
    def test_expect(self):
        # Test the expect function with different types of inputs
        d = 100
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.quantity.expect(mat, state)
        res1 = state.conj() @ (mat @ state)
        self.assertAlmostEqual(res, res1)

        d = 100
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.quantity.expect(mat, state)
        res1 = state.conj() @ (mat.toarray() @ state)
        self.assertAlmostEqual(res, res1)

        d = 100
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        state = np.random.randn(d) + 1j * np.random.randn(d)
        res = qt.quantity.expect(mat, state)
        res1 = state.conj() @ (mat.toarray() @ state)
        self.assertAlmostEqual(res, res1)
        
        # Test with non-Hermitian matrices
        d = 100
        n = 101
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.quantity.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.quantity.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n)
        res = qt.quantity.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = np.random.randn(d,d)
        states = np.random.randn(d,n)
        res = qt.quantity.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)



        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.quantity.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) , ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.quantity.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n)
        res = qt.quantity.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.dia_array(((np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n)
        res = qt.quantity.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        d = 100
        n = 101
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        states = np.random.randn(d,n)
        res = qt.quantity.expect(mat, states)
        res1 = (states.conj().T @ mat @ states).diagonal()
        self.assertAlmostEqual(np.linalg.norm(res - res1), 0)

        # dm

        d = 100
        n = 100
        mat = np.random.randn(d,d) + 1j * np.random.randn(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.quantity.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = sps.rand(d,d) + 1j * sps.rand(d,d)
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.quantity.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)

        d = 100
        n = 100
        mat = sps.dia_array(((np.random.randn(d) + 1j * np.random.randn(d), ), (0,)), shape=(d,d))
        states = np.random.randn(d,n) + 1j * np.random.randn(d,n)
        res = qt.quantity.expect(mat, states, isdm=True)
        res1 = (mat @ states).trace()
        self.assertAlmostEqual(res, res1)
        

if __name__ == '__main__':
    unittest.main()

