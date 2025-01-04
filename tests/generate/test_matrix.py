# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-30 19:59:42
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-30 20:34:33

import unittest
import numpy as np
import quante as qt

_PAULI_MAT = {
    "X": np.array([[0.,1.],[1.,0.]]),
    "Y": np.array([[0.,-1.j],[1.j,0.]]),
    "Z": np.array([[1.,0.],[0.,-1.]]),
    "P": np.array([[0.,1.],[0.,0.]]),
    "M": np.array([[0.,0.],[1.,0.]]),
    "I": np.array([[1.,0.],[0.,1.]]),
    "x": np.array([[0.,0.5],[0.5,0.]]),
    "y": np.array([[0.,-0.5j],[0.5j,0.]]),
    "z": np.array([[0.5,0.],[0.,-0.5]]),
    "p": np.array([[0.,1.],[0.,0.]]),
    "m": np.array([[0.,0.],[1.,0.]]),
    "i": np.array([[1.,0.],[0.,1.]]),
}


class TestMatrix(unittest.TestCase):
    
    def test_pauli_matrix(self):
        for key, value in _PAULI_MAT.items():
            self.assertTrue(np.allclose(qt.generate.matrix.pauli_matrix(key), value))
        
        for key1, value1 in _PAULI_MAT.items():
            for key2, value2 in _PAULI_MAT.items():
                self.assertTrue(np.allclose(qt.generate.matrix.pauli_matrix(key1+key2), np.kron(value1, value2)))
        
        for key1, value1 in _PAULI_MAT.items():
            for key2, value2 in _PAULI_MAT.items():
                for key3, value3 in _PAULI_MAT.items():
                    self.assertTrue(np.allclose(qt.generate.matrix.pauli_matrix(key1+key2+key3), np.kron(np.kron(value1, value2), value3)))
        
        self.assertTrue(np.allclose(qt.generate.matrix.pauli_matrix("5*xy+7*zy"), 5*np.kron(_PAULI_MAT['x'], _PAULI_MAT['y']) + 7*np.kron(_PAULI_MAT['z'], _PAULI_MAT['y'])))
    

if __name__ == "__main__":
   unittest.main()
