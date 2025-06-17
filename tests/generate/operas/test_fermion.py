# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-04-22 14:13:21
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 11:19:15

import unittest
import quante as qt
op = qt.generate.operas

class TestSlaterState(unittest.TestCase):
    def test_BdG_ham(self):
        L = 8
        ham = qt.generate.operas.spin.builder()
        for i in range(L-1):
            ham += "xx", [i, i+1], 1.0
        for i in range(L):
            ham += "z", [i], 1.0
        ham = ham.build()

        ham = ham.jw_transfer()

        BdG, coef_I = ham.BdG_ham()

        ham2 = qt.generate.operas.fermion.builder()
        for i in range(L):
            for j in range(L):
                ham2 += "+-", [i,j], BdG[i,j]/2
                ham2 += "++", [i,j], BdG[i,L+j]/2
                ham2 += "--", [i,j], BdG[L+i,j]/2
                ham2 += "-+", [i,j], BdG[L+i,L+j]/2
                ham2 += "I", [0], coef_I
        ham2 = ham2.build()
        ham2 = ham2.normal_ordering()

        self.assertTrue((ham2 -  ham).data == {})

