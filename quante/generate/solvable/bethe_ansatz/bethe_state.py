# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-05 00:00:00

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


__all__ = ["BetheState", "SixVertexBetheState", "XXZBetheKernel", "plot_roots"]


@dataclass(frozen=True)
class XXZBetheKernel:
    """Regime-specific scalar kernels for XXZ Bethe equations."""

    delta: float
    regime: str
    parameter: float

    @classmethod
    def from_delta(cls, delta):
        delta = float(delta)
        if -1.0 < delta < 1.0:
            return cls(delta=delta, regime="massless", parameter=float(np.arccos(delta)))
        if delta > 1.0:
            return cls(delta=delta, regime="massive", parameter=float(np.arccosh(delta)))
        if delta < -1.0:
            return cls(delta=delta, regime="massive_negative", parameter=float(np.arccosh(-delta)))
        raise ValueError(f"XXZ Bethe helpers exclude the isotropic points delta=+/-1, got {delta}")

    @classmethod
    def from_delta_parameter(cls, delta, parameter):
        parameter = float(parameter)
        if delta is None:
            if not (0.0 < parameter < np.pi):
                raise ValueError(
                    "eta should satisfy 0 < eta < pi when delta is omitted, "
                    f"got {parameter}"
                )
            return cls(delta=float(np.cos(parameter)), regime="massless", parameter=parameter)

        kernel = cls.from_delta(delta)
        if kernel.regime == "massless":
            if not np.isclose(kernel.delta, np.cos(parameter)):
                raise ValueError("delta and eta are inconsistent")
        else:
            expected = np.cosh(parameter)
            expected = expected if kernel.delta > 0.0 else -expected
            if parameter <= 0.0:
                raise ValueError(f"gamma should be positive for abs(delta) > 1, got {parameter}")
            if not np.isclose(kernel.delta, expected):
                raise ValueError("delta and gamma are inconsistent")
        return cls(delta=kernel.delta, regime=kernel.regime, parameter=parameter)

    @property
    def parameter_name(self):
        return "eta" if self.regime == "massless" else "gamma"

    @property
    def mapped_delta(self):
        return -self.delta if self.regime == "massive_negative" else self.delta

    @property
    def is_massless(self):
        return self.regime == "massless"

    @property
    def is_massive(self):
        return self.regime in {"massive", "massive_negative"}

    @property
    def energy_sign(self):
        return -1.0 if self.regime == "massive_negative" else 1.0

    def map_alpha_to_u(self, alphas):
        return -self.parameter / 2.0 + 0.5j * np.asarray(alphas)

    @property
    def root_branch(self):
        if self.is_massless:
            return "u = -eta / 2 + 1j * alpha / 2"
        return "u = -gamma / 2 + 1j * alpha / 2, |Delta| = cosh(gamma), alpha in (-pi, pi)"

    def p_prime(self, alpha):
        alpha = np.asarray(alpha)
        if self.is_massless:
            eta = self.parameter
            return np.sin(eta) / (np.cosh(alpha) - np.cos(eta))
        gamma = self.parameter
        return np.sinh(gamma) / (np.cosh(gamma) - np.cos(alpha))

    def theta_prime(self, x):
        x = np.asarray(x)
        if self.is_massless:
            eta = self.parameter
            return np.sin(2.0 * eta) / (np.cosh(x) - np.cos(2.0 * eta))
        gamma = self.parameter
        return np.sinh(2.0 * gamma) / (np.cosh(2.0 * gamma) - np.cos(x))

    def bare_energy(self, alpha, h):
        alpha = np.asarray(alpha)
        h = float(h)
        if self.is_massless:
            eta = self.parameter
            return 2.0 * h - 4.0 * np.sin(eta) ** 2 / (np.cosh(alpha) - np.cos(eta))
        gamma = self.parameter
        return 2.0 * h - 4.0 * np.sinh(gamma) ** 2 / (np.cosh(gamma) - np.cos(alpha))

    def vacuum_energy_density(self, h):
        return float(self.delta - float(h))

    def finite_bare_energy(self, alphas):
        alphas = np.asarray(alphas, dtype=float)
        if self.is_massless:
            eta = self.parameter
            return -np.sin(eta) ** 2 / (np.cosh(alphas) - np.cos(eta))
        gamma = self.parameter
        return -np.sinh(gamma) ** 2 / (np.cosh(gamma) - np.cos(alphas))

    def finite_reference_delta(self):
        return self.delta if self.is_massless else abs(self.delta)

    def finite_energy(self, alphas, L, *, j=1.0, pauli=True):
        energy = float(j) * self.energy_sign * (
            int(L) * self.finite_reference_delta() / 4.0
            + np.sum(self.finite_bare_energy(alphas))
        )
        if pauli:
            energy *= 4.0
        return float(np.real_if_close(energy))

    def finite_theta(self, x, n: int):
        if self.is_massless:
            return 2.0 * np.arctan(np.tanh(x) / np.tan(n * self.parameter / 2.0))

        x = np.asarray(x)
        branch = np.floor((x + np.pi) / (2.0 * np.pi))
        x0 = x - branch * 2.0 * np.pi
        principal = 2.0 * np.arctan(
            np.tan(x0 / 2.0) / np.tanh(n * self.parameter / 2.0)
        )
        return principal + 2.0 * np.pi * branch

    def finite_theta_derivative(self, x, n: int):
        if self.is_massless:
            tan_half = np.tan(n * self.parameter / 2.0)
            tanh_x = np.tanh(x)
            sech2_x = 1.0 / np.cosh(x) ** 2
            return 2.0 * tan_half * sech2_x / (tan_half ** 2 + tanh_x ** 2)

        x = np.asarray(x)
        branch = np.floor((x + np.pi) / (2.0 * np.pi))
        x0 = x - branch * 2.0 * np.pi
        tan_half = np.tan(x0 / 2.0)
        tanh_gamma = np.tanh(n * self.parameter / 2.0)
        return tanh_gamma * (1.0 + tan_half ** 2) / (tanh_gamma ** 2 + tan_half ** 2)

    def finite_initial_guess(self, L, qnums):
        qnums = np.asarray(qnums, dtype=float)
        if self.is_massless:
            arg = np.tan(np.pi * qnums / int(L)) * np.tan(self.parameter / 2.0)
            arg = np.clip(arg, -0.95, 0.95)
            return 2.0 * np.arctanh(arg)

        arg = np.tan(np.pi * qnums / int(L)) * np.tanh(self.parameter / 2.0)
        return 2.0 * np.arctan(arg)


class BetheState:
    """Base container for a Bethe-ansatz solution.

    This class intentionally carries only model-independent data. Model
    specific helpers, such as XXZ energies, should live in their own modules.
    Branch-specific subclasses may add convenient coordinates.
    """

    def __init__(
        self,
        L: int,
        roots: ndarray,
        *,
        qnums: ndarray | None = None,
        lax_matrix=None,
        solver=None,
        metadata: dict | None = None,
    ):
        self.L = int(L)
        self.roots = np.asarray(roots)
        self.qnums = None if qnums is None else np.asarray(qnums, dtype=float)
        self.lax_matrix = lax_matrix
        self.solver = solver
        self.metadata = {} if metadata is None else dict(metadata)

    @property
    def converged(self):
        if self.solver is None:
            return None
        return bool(self.solver.success)

    @property
    def residual_norm(self):
        if self.solver is None:
            return None
        return float(np.linalg.norm(self.solver.fun))

    @property
    def scipy_result(self):
        """Compatibility alias for scipy-based solvers."""
        return self.solver

    def plot_roots(self, ax=None, **kwargs):
        return plot_roots(self, ax=ax, **kwargs)


class SixVertexBetheState(BetheState):
    r"""Bethe state with ``alpha`` coordinates and six-vertex anisotropy."""

    def __init__(
        self,
        L: int,
        *,
        alphas: ndarray,
        eta: float,
        map_alpha_to_u,
        qnums: ndarray | None = None,
        lax_matrix=None,
        solver=None,
        metadata: dict | None = None,
        root_branch: str | None = None,
    ):
        alphas = np.asarray(alphas, dtype=float)
        self.alphas = alphas
        self.eta = float(eta)
        self._map_alpha_to_u = map_alpha_to_u
        super().__init__(
            L=L,
            roots=self.map_alpha_to_u(),
            qnums=qnums,
            lax_matrix=lax_matrix,
            solver=solver,
            metadata=metadata,
        )
        self.root_branch = root_branch

    def map_alpha_to_u(self, alphas=None):
        if alphas is None:
            alphas = self.alphas
        return self._map_alpha_to_u(alphas, self.eta)
    
    def xxz_energy(self, j: float = 1.0, *, pauli: bool = True):
        """Return the periodic XXZ energy for a state from ``solve_xxz_state``."""
        from .finite_pbc_xxz import energy_from_rapidities
        delta = self.metadata["delta"]
        return energy_from_rapidities(
            self.alphas,
            self.L,
            delta,
            eta=self.eta,
            j=j,
            pauli=pauli,
        )
    
    def xxz_energy_density(self, j: float = 1.0, *, pauli: bool = True):
        return self.xxz_energy(j=j, pauli=pauli) / self.L


def plot_roots(state_or_roots, ax=None, **kwargs):
    """Plot complex Bethe roots and return the matplotlib axes."""
    if isinstance(state_or_roots, BetheState):
        roots = state_or_roots.roots
    else:
        roots = np.asarray(state_or_roots)

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (5, 5)))

    scatter_kwargs = {"s": 35}
    scatter_kwargs.update(kwargs)
    ax.scatter(roots.real, roots.imag, **scatter_kwargs)
    ax.axhline(0, linestyle="--", linewidth=0.8, color="0.6")
    ax.axvline(0, linestyle="--", linewidth=0.8, color="0.6")
    ax.set_xlabel(r"$\mathrm{Re}\,u$")
    ax.set_ylabel(r"$\mathrm{Im}\,u$")
    # ax.set_aspect("equal", adjustable="box")
    return ax
