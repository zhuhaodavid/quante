# -*- coding: utf-8 -*-

import unittest

import numpy as np

import quante.tensornetwork.opensystem as qto
from quante.generate.matrix import pauli_matrix


class TestOpenSystemResultProcess(unittest.TestCase):
    def test_tempo_result_measure_none(self):
        states = np.array([np.eye(2), 0.5 * np.eye(2)])
        result = qto.TempoResult([0.0, 1.0], states)
        self.assertTrue(np.allclose(result.measure(), states))

    def test_tempo_result_measure_matrix(self):
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        rho1 = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = qto.TempoResult([0.0, 1.0], [rho0, rho1])
        times, vals = result.measure(pauli_matrix("z"), real=True)
        self.assertTrue(np.allclose(times, [0.0, 1.0]))
        self.assertTrue(np.allclose(vals, [0.5, -0.5]))

    def test_tempo_result_measure_callable(self):
        result = qto.TempoResult([0.0, 1.0], [np.eye(2), 2 * np.eye(2)])
        _, vals = result.measure(lambda t, rho: t + np.trace(rho), real=True)
        self.assertTrue(np.allclose(vals, [2.0, 5.0]))

    def test_tempo_result_state(self):
        result = qto.TempoResult([0.0], [np.eye(2)])
        self.assertTrue(np.allclose(result.state(), np.eye(2)))

    def test_process_tensor_storage(self):
        pt = qto.ProcessTensor(times=[0.0, 1.0], initial_tensor=np.array([1.0]))
        mpo = np.zeros((1, 2, 2, 3))
        cap = np.ones((3,))
        pt.set_mpo_tensor(0, mpo)
        pt.set_cap_tensor(0, cap)
        self.assertIs(pt.get_mpo_tensor(0), mpo)
        self.assertIs(pt.get_cap_tensor(0), cap)
        self.assertEqual(pt.bond_dims(), [(1, 3)])

    def test_pt_tempo_placeholder(self):
        with self.assertRaises(NotImplementedError):
            qto.pt_tempo_compute(None, None, None, None)


if __name__ == "__main__":
    unittest.main()

