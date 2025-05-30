# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-16 23:25:52
# @Last Modified by:   hzhu
# @Last Modified time: 2025-05-29 22:15:09

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
    
    def test_unfolding(self):
        op = qt.generate.operas.spin

        L = 6
        gs = [0.43656223, 0.07111102, 0.11462402, 0.37942382, 0.07833752, 0.12742619]

        builder = op.builder()
        for i in range(L-1):
            builder += 'xx', [i, i+1], 1.
            builder += 'yy', [i, i+1], 1.
            builder += 'zz', [i, i+1], 1.
            builder += 'z', [i,], gs[i]
        ham = builder.build()

        basis = qt.generate.basis.spin_basis(L=L, Nup=L//2)
        mat = ham.to_matrix(basis=basis)
        eng = np.linalg.eigvalsh(mat)

        eng_unfold = qt.quantity.unfolding_diff(eng)

        res = np.array([1.25435517, 0.46919587, 1.58384313, 
                        .43021752,1.58826541,0.46563927,1.22822096,
                        1.02044134,0.96412631,1.03219716,0.97022907])

        np.testing.assert_allclose(res, eng_unfold, atol=1e-8)
    

#todo write more tests


if __name__ == '__main__':
    unittest.main()

