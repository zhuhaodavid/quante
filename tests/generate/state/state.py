# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 17:18:57
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 18:02:04


import unittest
import numpy as np
import quante as qt

class TestMatrix(unittest.TestCase):
    def test_product_state(self):
        L = 6
        basis = qt.generate.basis.spin_basis(L=L)
        for j,s in enumerate(basis.s_list):
            a = ["up" if i == '0' else "dn" for i in np.binary_repr(s, width=L)]
            b = qt.generate.state.product_state(a)
            c = qt.generate.state.onehot(j, basis.Ns)
            self.assertTrue(np.allclose(b, c)) 
        
    def test_product_state_Nup(self):
        L = 6
        for Nup in range(0, L + 1):
            basis = qt.generate.basis.spin_basis(L=L, Nup=Nup)
            for j,s in enumerate(basis.s_list):
                a = ["up" if i == '0' else "dn" for i in np.binary_repr(s, width=L)]
                b = qt.generate.state.product_state(a, Nup=Nup)
                c = qt.generate.state.onehot(j, basis.Ns)
                self.assertTrue(np.allclose(b, c)) 
     
    def test_neel(self):
        for Nup in [None, 3]:
            a = qt.generate.state.neel(6, Nup=Nup)
            b = qt.generate.state.product_state(["up", "dn", "up", "dn", "up", "dn"], Nup=Nup)
            self.assertTrue(np.allclose(a, b))

            a = qt.generate.state.neel(6, down_first=True, Nup=Nup)
            b = qt.generate.state.product_state(["dn", "up", "dn", "up", "dn", "up"], Nup=Nup)
            self.assertTrue(np.allclose(a, b))

            a = qt.generate.state.neel(5, Nup=Nup)
            b = qt.generate.state.product_state(["up", "dn", "up", "dn", "up"], Nup=Nup)
            self.assertTrue(np.allclose(a, b))

        for Nup in [None, 2]:
            a = qt.generate.state.neel(5, down_first=True, Nup=Nup)
            b = qt.generate.state.product_state(["dn", "up", "dn", "up", "dn"], Nup=Nup)
            self.assertTrue(np.allclose(a, b))
    
if __name__ == '__main__':
    unittest.main()

            