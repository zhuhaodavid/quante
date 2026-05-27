# -*- coding: utf-8 -*-

import unittest
import sys
from pathlib import Path

import numpy as np

import quante.tensornetwork.opensystem as qto
from quante.generate.matrix import pauli_matrix


def _import_oqupy():
    mod = sys.modules.get("tensornetwork")
    if mod is not None and "tests" in str(getattr(mod, "__file__", "")):
        del sys.modules["tensornetwork"]
    tests_path = str(Path.cwd() / "tests")
    removed = False
    if tests_path in sys.path:
        sys.path.remove(tests_path)
        removed = True
    try:
        import oqupy
    finally:
        if removed:
            sys.path.append(tests_path)
    return oqupy


oqupy = _import_oqupy()


class TestOpenSystemTempoBenchmark(unittest.TestCase):
    def setUp(self):
        self.sx = pauli_matrix("X")
        self.sz = pauli_matrix("Z")
        up = pauli_matrix("u")
        self.rho0 = up @ up.conj().T
        self.ts = np.arange(0.0, 0.3 + 0.05, 0.1)
        self.params = qto.TempoParams(dt=0.1, tcut=0.2, epsrel=1e-8)

    def _quante(self, backend):
        system = qto.System(0.5 * self.sx)
        bath = qto.Bath(0.5 * self.sz, qto.PowerLawSD(0.1, 1.0, 5.0))
        result = qto.tempo_compute(
            system,
            bath,
            self.rho0,
            self.ts,
            self.params,
            backend=backend,
            progressbar=False,
        )
        return result.measure(0.5 * self.sz, real=False)[1]

    def _oqupy(self):
        system = oqupy.System(0.5 * self.sx)
        bath = oqupy.Bath(0.5 * self.sz, oqupy.PowerLawSD(0.1, 1.0, 5.0))
        params = oqupy.TempoParameters(dt=0.1, epsrel=1e-8, tcut=0.2)
        dynamics = oqupy.tempo_compute(
            system,
            bath,
            self.rho0,
            0.0,
            float(self.ts[-1]),
            parameters=params,
            progress_type="silent",
        )
        return dynamics.expectations(0.5 * self.sz, real=False)[1]

    def test_reference_matches_oqupy_small_case(self):
        self.assertTrue(np.allclose(self._quante("reference"), self._oqupy(), atol=2e-4))

    def test_mps_matches_oqupy_small_case(self):
        self.assertTrue(np.allclose(self._quante("mps"), self._oqupy(), atol=2e-4))

    def test_mps_matches_reference_small_case(self):
        self.assertTrue(np.allclose(self._quante("mps"), self._quante("reference"), atol=1e-12))

    def test_reference_zero_bath_matches_system_backend(self):
        system = qto.System(0.5 * self.sx)
        bath = qto.Bath(0.5 * self.sz, qto.CustomCorrelation(lambda t: 0.0))
        vals = {}
        for backend in ["system", "reference"]:
            result = qto.tempo_compute(
                system,
                bath,
                self.rho0,
                self.ts,
                self.params,
                backend=backend,
                progressbar=False,
            )
            vals[backend] = result.measure(0.5 * self.sz, real=False)[1]
        self.assertTrue(np.allclose(vals["reference"], vals["system"], atol=1e-12))


if __name__ == "__main__":
    unittest.main()
