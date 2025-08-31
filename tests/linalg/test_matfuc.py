# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-31 14:01:04
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 14:23:24


import unittest

import numpy as np
import scipy.sparse as sp
from quante.linalg.matops import isherm
try:
    import torch as tc
    from quante.bridge.torch_utils import totc
    has_tc = True
except ImportError:
    has_tc = False

class Testisherm(unittest.TestCase):
    
    def test_np(self):
        d = 100
        mat = np.random.randn(d, d)
        self.assertFalse(isherm(mat))
        mat = mat + mat.T
        self.assertTrue(isherm(mat))
        mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
        self.assertFalse(isherm(mat))
        mat = mat + mat.T.conj()
        self.assertTrue(isherm(mat))

    def test_sp(self):
        d = 100
        mat = sp.random(d, d, density=0.8)
        self.assertFalse(isherm(mat))
        mat = mat + mat.T
        self.assertTrue(isherm(mat))
        mat = sp.random(d, d, density=0.8, dtype=np.complex128)
        self.assertFalse(isherm(mat))
        mat = mat + mat.T.conj()
        self.assertTrue(isherm(mat))

    @unittest.skipUnless(has_tc, "Torch not available")
    def test_tc(self):
        d = 100
        mat = tc.randn(d, d, dtype=tc.float64)
        self.assertFalse(isherm(mat))
        mat = mat + mat.T
        self.assertTrue(isherm(mat))
        mat = tc.randn(d, d, dtype=tc.complex128)
        self.assertFalse(isherm(mat))
        mat = mat + mat.T.conj()
        self.assertTrue(isherm(mat))
    
    @unittest.skipUnless(has_tc, "Torch not available")
    def test_sp(self):
        d = 100
        mat = sp.random(d, d, density=0.8)
        mat = totc(mat)
        self.assertFalse(isherm(mat))
        mat = sp.random(d, d, density=0.8)
        mat = mat + mat.T
        mat = totc(mat)
        self.assertTrue(isherm(mat))
        mat = sp.random(d, d, density=0.8, dtype=np.complex128)
        mat = totc(mat)
        self.assertFalse(isherm(mat))
        mat = sp.random(d, d, density=0.8, dtype=np.complex128)
        mat = mat + mat.T.conj()
        mat = totc(mat)
        self.assertTrue(isherm(mat))

if __name__ == '__main__':
    unittest.main()