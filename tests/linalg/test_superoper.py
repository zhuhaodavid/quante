# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-14 22:23:16
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-30 18:56:56


import unittest
import quante as qt
import numpy as np
from quante.generate.matrix import SuperOperator

class TestSuperOper(unittest.TestCase):
    def setUp(self):
        d = 2
        ndX, ndY = np.random.rand(d,d) + 1j*np.random.rand(d,d), np.random.rand(d,d) + 1j*np.random.rand(d,d)
        self.ndmat = np.random.rand(d,d) + 1j*np.random.rand(d,d)
        self.so = SuperOperator([ndX, ndY], order='c')
        self.sof = SuperOperator([ndX, ndY], order='f')
        self.res1 = sum([i @ self.ndmat @ i.conj().T for i in [ndX, ndY]])

    def test_time_measurements(self):
        super_oper = self.so.vectorize()
        res1 = (super_oper @ self.ndmat.reshape(-1)).reshape(2,2)
        self.assertTrue(np.allclose(res1, self.res1))

        super_oper = self.sof.vectorize()
        res1 = (super_oper @ self.ndmat.reshape(-1, order='f')).reshape(2,2, order='f')
        self.assertTrue(np.allclose(res1, self.res1))

        # choi form
        choi_oper = self.so.choi_form()
        res1 = np.einsum("ijkj->ik", (choi_oper @ np.kron(np.eye(2), self.ndmat)).reshape(2,2,2,2))
        self.assertTrue(np.allclose(res1, self.res1))
        
        choi_oper = self.sof.choi_form()
        res1 = np.einsum("ijkj->ik", (choi_oper @ np.kron(self.ndmat.T, np.eye(2))).reshape(2,2,2,2, order='f'))
        self.assertTrue(np.allclose(res1, self.res1))
        
        # kraus form
        kraus_opers = self.so.kraus_form()
        res1 = np.zeros((2,2), dtype=complex)
        for kraus_oper in kraus_opers:
            res1 += kraus_oper @ self.ndmat @ kraus_oper.conj().T
        self.assertTrue(np.allclose(res1, self.res1))
        
        kraus_opers = self.sof.kraus_form()
        res1 = np.zeros((2,2), dtype=complex)
        for kraus_oper in kraus_opers:
            res1 += kraus_oper @ self.ndmat @ kraus_oper.conj().T
        self.assertTrue(np.allclose(res1, self.res1))
        
        # stinespring form
        A, B = self.so.stinespring_form()
        res1 = np.einsum('ijik->jk', (A @ self.ndmat @ B).reshape(2,2,2,2))
        self.assertTrue(np.allclose(res1, self.res1))
        
        A, B = self.sof.stinespring_form()
        res1 = np.einsum('ijik->jk', (A @ self.ndmat @ B).reshape(2,2,2,2, order='f'))
        self.assertTrue(np.allclose(res1, self.res1))
        
        

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_expm_multiply_numba_matrix"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
       


if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_expm_multiply_numba_matrix"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
