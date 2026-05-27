# -*- coding: utf-8 -*-

import unittest

import matplotlib
import numpy as np

import quante.tensornetwork.opensystem as qto
from quante.generate.matrix import pauli_matrix

matplotlib.use("Agg")


class TestOpenSystemBath(unittest.TestCase):
    def test_correlation_base_call_raises(self):
        corr = qto.Correlation()
        with self.assertRaises(NotImplementedError):
            corr(0.0)

    def test_custom_correlation_call_and_corr(self):
        corr = qto.CustomCorrelation(lambda t: 1.0 + 2.0j * t)
        self.assertEqual(corr(3.0), 1.0 + 6.0j)
        self.assertEqual(corr.corr(3.0), 1.0 + 6.0j)

    def test_correlation_integral_constant(self):
        corr = qto.CustomCorrelation(lambda t: 2.0 + 0.5j)
        self.assertAlmostEqual(corr.integral(0, 0, 0.2), (2.0 + 0.5j) * 0.04)

    def test_correlation_2d_integral_shapes_constant(self):
        corr = qto.CustomCorrelation(lambda t: 3.0 - 1.0j)
        dt = 0.2
        val = 3.0 - 1.0j
        self.assertAlmostEqual(
            corr.correlation_2d_integral(dt, 0.0, shape="square"),
            val * dt**2,
        )
        self.assertAlmostEqual(
            corr.correlation_2d_integral(dt, 0.0, shape="upper-triangle"),
            val * dt**2 / 2,
        )
        self.assertAlmostEqual(
            corr.correlation_2d_integral(dt, 0.0, shape="lower-triangle"),
            val * dt**2 / 2,
        )
        self.assertAlmostEqual(
            corr.correlation_2d_integral(dt, 0.0, 0.5, shape="rectangle"),
            val * 0.5 * dt,
        )

    def test_correlation_coefficients(self):
        corr = qto.CustomCorrelation(lambda t: 1.0)
        coeffs = corr.coefficients(3, 0.1)
        self.assertEqual(coeffs.shape, (4,))
        self.assertTrue(np.allclose(coeffs, [0.005, 0.01, 0.01, 0.01]))

    def test_power_law_spectral_density(self):
        corr = qto.PowerLawSD(alpha=0.1, zeta=1.0, cutoff=5.0)
        w = np.array([0.0, 1.0, 2.0])
        ref = 2 * 0.1 * w * np.exp(-w / 5.0)
        self.assertTrue(np.allclose(corr.spectral_density(w), ref))

    def test_power_law_corr_returns_complex(self):
        corr = qto.PowerLawSD(alpha=0.01, zeta=1.0, cutoff=2.0)
        self.assertTrue(np.iscomplexobj(corr.corr(0.1)))

    def test_bath_wraps_callable(self):
        bath = qto.Bath(pauli_matrix("z"), lambda t: 1.0)
        self.assertIsInstance(bath.corr, qto.CustomCorrelation)
        self.assertEqual(bath.dim, 2)

    def test_bath_keeps_correlation(self):
        corr = qto.CustomCorrelation(lambda t: 1.0)
        bath = qto.Bath(pauli_matrix("z"), corr)
        self.assertIs(bath.corr, corr)

    def test_bath_diagonalize_coupling(self):
        bath = qto.Bath(pauli_matrix("z"), lambda t: 1.0)
        vals, vecs = bath.diagonalize_coupling()
        self.assertTrue(np.allclose(vals, [-0.5, 0.5]))
        self.assertTrue(np.allclose(vecs.conj().T @ vecs, np.eye(2)))

    def test_plot_correlations_with_parameters(self):
        import matplotlib.pyplot as plt

        corr = qto.CustomCorrelation(lambda t: np.exp(-t) * (1 - 1j))
        params = qto.TempoParams(dt=0.1, tcut=0.3)
        fig, ax = plt.subplots()
        out = qto.plot_correlations_with_parameters(corr, params, ax=ax)
        self.assertIs(out, ax)
        self.assertGreaterEqual(len(ax.lines), 5)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
