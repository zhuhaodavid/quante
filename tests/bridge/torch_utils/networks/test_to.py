# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-19 20:28:48
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-22 14:22:53


import unittest
import torch as tc
import quante.bridge.torch_utils as qtc
import quante.bridge.torch_utils.networks.tensor_operations as tf

class TestTN(unittest.TestCase):
    def test_noise_proj_left(self):
        dtype = tc.float64
        lproj = tc.randn(5,4,5, dtype=dtype)
        mid = tc.randn(4,2,2,4, dtype=dtype)
        phi = tc.randn(5,2,2,6, dtype=dtype)
        res = tf._noise_proj_left(lproj, mid, phi)
        # res2 = tc.einsum("adg,dcfe,gfh->ache", lproj, mid, psii)
        res2 = tc.einsum("adg,dcfe,gfbi->acbie", lproj, mid, phi)
        a,c,b,i,e = res2.shape
        self.assertTrue(tc.allclose(res, res2.reshape(a*c,-1)))
    
    def test_noise_proj_right(self):
        dtype = tc.float64
        rproj = tc.randn(6,4,6, dtype=dtype)
        mid = tc.randn(4,2,2,4, dtype=dtype)
        phi = tc.randn(5,2,2,6, dtype=dtype)
        res = tf._noise_proj_right(phi, mid, rproj)
        res2 = tc.einsum("agfh,dcfe,beh->cbagd", phi, mid, rproj)
        b,c,d,a,g = res2.shape
        self.assertTrue(tc.allclose(res, res2.reshape(b*c,-1)))

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestTN("test_dm"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

