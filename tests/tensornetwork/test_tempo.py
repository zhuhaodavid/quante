# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-31 00:00:00
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-31 00:00:00

import unittest

import numpy as np

import quante as qt
import quante.tensornetwork.opensystem as qto
from quante.generate.matrix import pauli_matrix
from quante.tensornetwork import MPS


class TestTempoEngine(unittest.TestCase):
    def test_tempo_uses_evolve_engine_step_and_run(self):
        sigma_x = pauli_matrix("X")
        sigma_z = pauli_matrix("Z")
        up = pauli_matrix("u")
        rho0 = up @ up.conj().T

        system = qt.generate.LiouvillianDynamics(ham=0.5 * sigma_x)
        bath = qto.Bath(
            0.5 * sigma_z,
            qto.PowerLawSpectralDensity(alpha=0.1, zeta=1.0, cutoff=3.0),
        )
        params = qto.TempoParams(
            dt=0.1,
            tcut=0.2,
            epsrel=1e-3,
            trunc_cut=1e-8,
        )
        engine = qto.TempoEngine(system, bath, params, rho0, np.array([0.0, 0.1, 0.2]))

        self.assertFalse(engine.finished)
        self.assertIsInstance(engine.step(), MPS)
        self.assertTrue(np.allclose(engine.density_matrix(), rho0))
        self.assertAlmostEqual(engine.cur_time, 0.0)

        tail = engine.run(progressbar=False)
        self.assertEqual(tail.shape, (2, 2, 2))
        self.assertTrue(engine.finished)
        self.assertEqual(len(engine.truncation_errors), 2)
        self.assertIs(engine.cur_state, engine.mps)
        self.assertTrue(np.allclose(np.trace(engine.density_matrix()), 1.0))
        with self.assertRaises(StopIteration):
            engine.step()


if __name__ == "__main__":
    unittest.main()
