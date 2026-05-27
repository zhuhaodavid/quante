# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-19 20:28:48
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-22 14:22:53


import unittest
import numpy as np
import quante.tensornetwork as qtn

tf = qtn.tensor_operations

class TestTN(unittest.TestCase):
    def test_noise_proj_left(self):
        dtype = np.float64
        lproj = np.random.randn(5,4,5).astype(dtype)
        mid = np.random.randn(4,2,2,4).astype(dtype)
        phi = np.random.randn(5,2,2,6).astype(dtype)
        res = tf._noise_proj_left(lproj, mid, phi)
        # res2 = tc.einsum("adg,dcfe,gfh->ache", lproj, mid, psii)
        res2 = np.einsum("adg,dcfe,gfbi->acbie", lproj, mid, phi)
        a,c,b,i,e = res2.shape
        self.assertTrue(np.allclose(res, res2.reshape(a*c,-1)))
    
    def test_noise_proj_right(self):
        dtype = np.float64
        rproj = np.random.randn(6,4,6).astype(dtype)
        mid = np.random.randn(4,2,2,4).astype(dtype)
        phi = np.random.randn(5,2,2,6).astype(dtype)
        res = tf._noise_proj_right(phi, mid, rproj)
        res2 = np.einsum("agfh,dcfe,beh->cbagd", phi, mid, rproj)
        b,c,d,a,g = res2.shape
        self.assertTrue(np.allclose(res, res2.reshape(b*c,-1)))

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_dm"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)


