# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-30 19:59:42
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-06 11:17:54

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
    

class TestKIM(unittest.TestCase):
    
    def test_KIM_matrix(self):
        L = 10
        import quante as qt
        op = qt.generate.operas
        basis = qt.generate.spin_basis(L)
        
        b = 1.
        ham = op.sum(b * op.x(i) for i in range(L))
        mat1 = ham.to_matrix(basis, pauli=True)
        mat1 = qt.linalg.expm(-1j*mat1)
        
        J = 1.
        h = np.random.rand(L)
        ham = op.sum(J*op.zz(i, (i+1)%L) + h[i]*op.z(i) for i in range(L))
        mat2 = ham.to_matrix(basis, pauli=True)
        mat2 = qt.linalg.expm(-1j*mat2)
        
        mat = mat1 @ mat2
        self.assertTrue(np.allclose(mat, qt.generate.matrix.KIM_matrix(b, J, h, L)))
           
class TestSYK(unittest.TestCase):
    
    def test_syk4_dirac(self):
        L = 10
        Jmat = qt.generate.matrix._syk4_dirac_Jmat(L, J=1.0)
        basis = qt.generate.basis.quspin_fermion_basis(L=L, Nf=L//2)
        builder = qt.generate.operas.fermion.builder()
        for i1 in range(L):
            for i2 in range(L):
                for j1 in range(L):
                    for j2 in range(L):
                        builder += "++--", [i1, i2, j1, j2], Jmat[i1 * L + i2, j1 * L + j2]
        ham = builder.build()
        mat1 = ham.to_matrix(basis)
        mat2 = qt.generate.matrix.syk4_dirac(L, J=Jmat, Nf=L//2)
        self.assertTrue(np.allclose(mat1, mat2))

           
if __name__ == "__main__":
   unittest.main()
