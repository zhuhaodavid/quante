# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-05 00:00:00

from __future__ import annotations

import numpy as np
from numpy import ndarray


__all__ = ["BetheState", "SixVertexBetheState", "plot_roots"]


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
