# -*- coding: utf-8 -*-

import unittest

import numpy as np

import quante.tensornetwork.opensystem as qto
from quante.generate.matrix import pauli_matrix
from quante.tensornetwork.networks import MPS, MPO


class TestOpenSystemTempo(unittest.TestCase):
    def test_tempo_params_memory_steps(self):
        params = qto.TempoParams(dt=0.1, tcut=0.31)
        self.assertEqual(params.memory_steps(), 4)

    def test_tempo_engine_build_influence(self):
        sx = pauli_matrix("x")
        sz = pauli_matrix("z")
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        bath = qto.Bath(sz, lambda t: 0.0)
        params = qto.TempoParams(dt=0.1, tcut=0.2)
        engine = qto.TempoEngine(qto.System(sx), bath, rho0, [0.0], params, progressbar=False)
        self.assertEqual(engine.infl_coeffs.shape, (3,))
        self.assertIsNone(engine._compress_memory())

    def test_tempo_engine_run_initial_and_step(self):
        ham = pauli_matrix("z")
        rho0 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        bath = qto.Bath(pauli_matrix("z"), lambda t: 0.0)
        params = qto.TempoParams(dt=0.1, tcut=0.1)
        engine = qto.TempoEngine(qto.System(ham), bath, rho0, [0.0, 0.1], params, progressbar=False)
        self.assertTrue(np.allclose(engine.run(), rho0))
        rho1 = engine.run()
        ref = (qto.System(ham).propagator(0.1) @ rho0.reshape(-1)).reshape(2, 2)
        self.assertTrue(np.allclose(rho1, ref))

    def test_tempo_engine_measure_states(self):
        ham = np.zeros((2, 2), dtype=np.complex128)
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        bath = qto.Bath(pauli_matrix("z"), lambda t: 0.0)
        params = qto.TempoParams(dt=0.1, tcut=0.1)
        engine = qto.TempoEngine(qto.System(ham), bath, rho0, [0.0, 0.1], params, progressbar=False)
        states = engine.measure(progressbar=False)
        self.assertEqual(states.shape, (2, 2, 2))
        self.assertTrue(np.allclose(states[0], rho0))

    def test_tempo_engine_measure_matrix(self):
        ham = np.zeros((2, 2), dtype=np.complex128)
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        bath = qto.Bath(pauli_matrix("z"), lambda t: 0.0)
        params = qto.TempoParams(dt=0.1, tcut=0.1)
        engine = qto.TempoEngine(qto.System(ham), bath, rho0, [0.0, 0.1], params, progressbar=False)
        times, vals = engine.measure(pauli_matrix("z"), real=True, progressbar=False)
        self.assertTrue(np.allclose(times, [0.0, 0.1]))
        self.assertTrue(np.allclose(vals, [0.5, 0.5]))

    def test_tempo_engine_result(self):
        ham = np.zeros((2, 2), dtype=np.complex128)
        rho0 = np.eye(2) / 2
        bath = qto.Bath(pauli_matrix("z"), lambda t: 0.0)
        params = qto.TempoParams(dt=0.1, tcut=0.1)
        engine = qto.TempoEngine(qto.System(ham), bath, rho0, [0.0, 0.1], params, progressbar=False)
        result = engine.result()
        self.assertIsInstance(result, qto.TempoResult)
        self.assertEqual(result.states.shape, (2, 2, 2))

    def test_tempo_compute_result_and_measure(self):
        ham = np.zeros((2, 2), dtype=np.complex128)
        rho0 = np.eye(2) / 2
        bath = qto.Bath(pauli_matrix("z"), lambda t: 0.0)
        params = qto.TempoParams(dt=0.1, tcut=0.1)
        result = qto.tempo_compute(qto.System(ham), bath, rho0, [0.0, 0.1], params, progressbar=False)
        self.assertIsInstance(result, qto.TempoResult)
        times, vals = qto.tempo_compute(
            qto.System(ham),
            bath,
            rho0,
            [0.0, 0.1],
            params,
            measure=pauli_matrix("z"),
            real=True,
            progressbar=False,
        )
        self.assertTrue(np.allclose(times, [0.0, 0.1]))
        self.assertTrue(np.allclose(vals, [0.0, 0.0]))

    def test_mps_backend_initializes_adt_and_influence_mpo(self):
        ham = np.zeros((2, 2), dtype=np.complex128)
        rho0 = np.eye(2) / 2
        bath = qto.Bath(pauli_matrix("z"), lambda t: 0.0)
        params = qto.TempoParams(dt=0.1, tcut=0.2)
        engine = qto.TempoEngine(
            qto.System(ham),
            bath,
            rho0,
            [0.0, 0.1],
            params,
            backend="mps",
            progressbar=False,
        )
        self.assertIsInstance(engine.adt, MPS)
        self.assertEqual(len(engine.adt), 1)
        self.assertEqual(len(engine.influence_mpo), params.memory_steps() + 1)

    def test_mps_backend_select_step_mpo(self):
        ham = np.zeros((2, 2), dtype=np.complex128)
        rho0 = np.eye(2) / 2
        bath = qto.Bath(pauli_matrix("z"), lambda t: 0.0)
        params = qto.TempoParams(dt=0.1, tcut=0.2)
        engine = qto.TempoEngine(
            qto.System(ham),
            bath,
            rho0,
            [0.0, 0.1],
            params,
            backend="mps",
            progressbar=False,
        )
        prop, _ = engine._mps_propagators(0)
        engine._apply_first_half_step(prop)
        mpo = engine._select_step_mpo(1)
        self.assertIsInstance(mpo, MPO)
        self.assertEqual(len(mpo), len(engine.adt))

    def test_mps_backend_matches_reference_tiny_case(self):
        sx = pauli_matrix("x")
        sz = pauli_matrix("z")
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        bath = qto.Bath(0.5 * sz, qto.PowerLawSD(0.1, 1.0, 5.0))
        params = qto.TempoParams(dt=0.1, tcut=0.2, epsrel=1e-8)
        ts = np.arange(0.0, 0.5 + 0.05, 0.1)
        system = qto.System(0.5 * sx)
        vals = {}
        for backend in ["reference", "mps"]:
            result = qto.tempo_compute(
                system,
                bath,
                rho0,
                ts,
                params,
                backend=backend,
                progressbar=False,
            )
            vals[backend] = result.measure(0.5 * sz)[1]
        self.assertTrue(np.allclose(vals["mps"], vals["reference"], atol=1e-12))


if __name__ == "__main__":
    unittest.main()
