# -*- coding: utf-8 -*-

import unittest

import numpy as np
import scipy.linalg

import quante.tensornetwork.opensystem as qto
from quante.generate.matrix import pauli_matrix


class TestOpenSystemSystem(unittest.TestCase):
    def test_as_matrix_array(self):
        mat = pauli_matrix("x")
        self.assertTrue(np.allclose(qto.as_matrix(mat), mat))

    def test_liouvillian_from_ham_action(self):
        ham = pauli_matrix("z")
        rho = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        L = qto.liouvillian_from_ham(ham)
        got = (L @ rho.reshape(-1)).reshape(2, 2)
        ref = -1j * (ham @ rho - rho @ ham)
        self.assertTrue(np.allclose(got, ref))

    def test_liouvillian_from_ham_with_jump(self):
        ham = np.zeros((2, 2), dtype=np.complex128)
        jump = pauli_matrix("m")
        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        L = qto.liouvillian_from_ham(ham, [jump])
        got = (L @ rho.reshape(-1)).reshape(2, 2)
        ref = jump @ rho @ jump.conj().T
        ref -= 0.5 * ((jump.conj().T @ jump) @ rho + rho @ (jump.conj().T @ jump))
        self.assertTrue(np.allclose(got, ref))

    def test_system_liouvillian(self):
        ham = pauli_matrix("x")
        system = qto.System(ham)
        self.assertEqual(system.dim, 2)
        self.assertEqual(system.liouvillian().shape, (4, 4))

    def test_system_propagator(self):
        ham = pauli_matrix("z")
        system = qto.System(ham)
        dt = 0.17
        ref = scipy.linalg.expm(qto.liouvillian_from_ham(ham) * dt)
        self.assertTrue(np.allclose(system.propagator(dt), ref))

    def test_system_half_propagator(self):
        ham = pauli_matrix("z")
        system = qto.System(ham)
        dt = 0.17
        self.assertTrue(np.allclose(system.half_propagator(dt), system.propagator(dt / 2)))

    def test_system_rejects_ambiguous_input(self):
        ham = pauli_matrix("z")
        with self.assertRaises(ValueError):
            qto.System()
        with self.assertRaises(ValueError):
            qto.System(ham, lindbladian=np.eye(4))


if __name__ == "__main__":
    unittest.main()

