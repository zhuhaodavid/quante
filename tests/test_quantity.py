# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-16 23:25:52
# @Last Modified by:   hzhu
# @Last Modified time: 2025-05-22 18:12:12

import quante as qt
import numpy as np
import scipy.sparse as sps
import unittest
import torch as tc
import quante.torch_utils as qtc

class TestQuantity(unittest.TestCase):
    def test_entanglement_spectrum(self):
        # Create a simple 2-qubit Bell state |00> + |11>
        state = np.zeros(4, dtype=np.complex128)
        state[0] = 1/np.sqrt(2)
        state[3] = 1/np.sqrt(2)
        # No basis provided, left_number=1 (split between qubits)
        spectrum = qt.quantity.entanglement_spectrum(state, left_number=1)
        # The entanglement spectrum should be [1/sqrt(2), 1/sqrt(2)]
        expected = np.array([[1/np.sqrt(2), 1/np.sqrt(2)]])
        # Only the first two singular values are nonzero
        np.testing.assert_allclose(spectrum[:2], expected, rtol=1e-7, atol=1e-8)
        # The rest should be zero
        np.testing.assert_allclose(spectrum[2:], 0, atol=1e-8)
    

#todo write more tests


if __name__ == '__main__':
    unittest.main()