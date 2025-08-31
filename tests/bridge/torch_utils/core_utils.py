# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-31 14:58:14
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 15:34:32

import unittest
import scipy.sparse as sp
from quante.bridge.torch_utils import totc
from quante.bridge.torch_utils import tonp
import numpy as np
import torch as tc

class TestCoreUtils(unittest.TestCase):
    def test_tonp_sparse(self):
        d = 100

        npmat = sp.random(d, d, density=0.8, format='csr')
        mat = totc(npmat)
        mat = tonp(mat)
        self.assertEqual((mat-npmat).nnz, 0)

        npmat = sp.random(d, d, density=0.8, format='coo')
        mat = totc(npmat)
        mat = tonp(mat)
        self.assertEqual((mat-npmat).nnz, 0)

        npmat = sp.random(d, d, density=0.8, format='csc')
        mat = totc(npmat)
        mat = tonp(mat)
        self.assertEqual((mat-npmat).nnz, 0)

        npmat = sp.random(d, d, density=0.8, format='dia')
        try:
            mat = totc(npmat)
        except NotImplementedError:
            pass


if __name__ == "__main__":
    unittest.main()
