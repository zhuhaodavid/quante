# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-05 00:00:00

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy import ndarray
from scipy.optimize import root

from .bethe_state import BetheState, SixVertexBetheState


__all__ = [
    "bethe_quantum_numbers",
    "solve_xxz_state",
    "map_alpha_to_u",
    "xxz_energy",
    "xxz_energy_density",
    "xxz_energy_from_rapidities",
]


def bethe_quantum_numbers(M: int, *, center: float = 0.0):
    r"""Return ``M`` consecutive Bethe quantum numbers centered at ``center``."""
    M = int(M)
    if M <= 0:
        raise ValueError(f"M should be positive, got {M}")
    return np.arange(M, dtype=float) - (M - 1) / 2.0 + center


def solve_xxz_state(
    L: int,
    delta: float | None = None,
    *,
    eta: float | None = None,
    qnums: Optional[ndarray] = None,
    M: int | None = None,
    x0: Optional[ndarray] = None,
    tol: float = 1e-12,
    method: str = "hybr",
    raise_error: bool = True,
) -> SixVertexBetheState:
    r"""Solve XXZ Bethe equations for a chosen set of quantum numbers.

    This solves the massless-regime branch
    ``u = -eta / 2 + 1j * alpha / 2`` with ``Delta = cos(eta)``. The returned
    ``SixVertexBetheState`` stores ``state.roots`` as spectral parameters and
    keeps the branch coordinates in ``state.alphas``.
    """
    L = _check_chain_length(L)
    delta, eta = _check_delta_eta(delta, eta)
    qnums = _prepare_qnums(L, qnums, M)

    if x0 is None:
        x0 = _initial_guess(L, eta, qnums)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != qnums.shape:
            raise ValueError(f"x0 shape {x0.shape} does not match qnums shape {qnums.shape}")

    sol = root(
        _residual,
        x0,
        jac=_jacobian,
        args=(L, eta, qnums),
        method=method,
        tol=tol,
    )

    order = np.argsort(sol.x)
    alphas = np.asarray(sol.x[order], dtype=float)
    qnums = np.asarray(qnums[order], dtype=float)

    if raise_error and not sol.success:
        raise RuntimeError(f"Bethe rapidity solver did not converge: {sol.message}")

    return SixVertexBetheState(
        L=L,
        qnums=qnums,
        alphas=alphas,
        map_alpha_to_u=map_alpha_to_u,
        solver=sol,
        eta=eta,
        metadata={
            "model": "XXZ",
            "delta": delta,
        },
        root_branch="u = -eta / 2 + 1j * alpha / 2",
    )


def xxz_energy(state: BetheState, j: float = 1.0, *, pauli: bool = True):
    """Return the periodic XXZ energy for a state from ``solve_xxz_state``."""
    delta = state.metadata["delta"]
    return xxz_energy_from_rapidities(
        state.alphas,
        state.L,
        delta,
        eta=state.eta,
        j=j,
        pauli=pauli,
    )


def xxz_energy_density(state: BetheState, j: float = 1.0, *, pauli: bool = True):
    return xxz_energy(state, j=j, pauli=pauli) / state.L


def map_alpha_to_u(alphas, eta: float):
    """Map alpha branch coordinates to ``u = -eta / 2 + 1j * alpha / 2``."""
    return -float(eta) / 2.0 + 0.5j * np.asarray(alphas)


def xxz_u_roots(state: BetheState):
    """Compatibility helper. Prefer ``state.roots``."""
    return state.roots


def xxz_energy_from_rapidities(
    alphas,
    L: int,
    delta: float,
    *,
    eta: float | None = None,
    j: float = 1.0,
    pauli: bool = True,
):
    r"""Return periodic XXZ energy from ``alpha`` rapidities.

    With ``Delta = cos(eta)`` and
    ``u_j = -eta / 2 + 1j * alpha_j / 2``,

    .. math::
        E = J \left[
            \frac{L \Delta}{4}
            - \sum_j
            \frac{\sin^2\eta}{\cosh\alpha_j - \cos\eta}
        \right].

    By default ``pauli=True`` returns the energy in the Pauli-matrix
    convention, which is four times the spin-operator convention above.
    """
    delta, eta = _check_delta_eta(delta, eta)
    alphas = np.asarray(alphas, dtype=float)
    bare = -np.sin(eta) ** 2 / (np.cosh(alphas) - np.cos(eta))
    energy = j * (int(L) * delta / 4.0 + np.sum(bare))
    if pauli:
        energy *= 4.0
    return float(np.real_if_close(energy))


def _theta(x, n: int, eta: float):
    return 2.0 * np.arctan(np.tanh(x) / np.tan(n * eta / 2.0))


def _theta_derivative(x, n: int, eta: float):
    tan_half = np.tan(n * eta / 2.0)
    tanh_x = np.tanh(x)
    sech2_x = 1.0 / np.cosh(x) ** 2
    return 2.0 * tan_half * sech2_x / (tan_half ** 2 + tanh_x ** 2)


def _residual(alphas, L: int, eta: float, qnums):
    alphas = np.asarray(alphas, dtype=float)
    qnums = np.asarray(qnums, dtype=float)
    diff = (alphas[:, None] - alphas[None, :]) / 2.0
    scatter = np.sum(_theta(diff, 2, eta), axis=1)
    return L * _theta(alphas / 2.0, 1, eta) - scatter - 2.0 * np.pi * qnums


def _jacobian(alphas, L: int, eta: float, qnums):
    alphas = np.asarray(alphas, dtype=float)
    diff = (alphas[:, None] - alphas[None, :]) / 2.0
    jac = 0.5 * _theta_derivative(diff, 2, eta)
    diag = (
        0.5 * L * _theta_derivative(alphas / 2.0, 1, eta)
        - np.sum(jac, axis=1)
        + np.diag(jac)
    )
    np.fill_diagonal(jac, diag)
    return jac


def _initial_guess(L: int, eta: float, qnums):
    arg = np.tan(np.pi * qnums / L) * np.tan(eta / 2.0)
    arg = np.clip(arg, -0.95, 0.95)
    return 2.0 * np.arctanh(arg)


def _prepare_qnums(L: int, qnums, M):
    if qnums is None:
        if M is None:
            M = L // 2
        return bethe_quantum_numbers(M)

    qnums = np.asarray(qnums, dtype=float)
    if qnums.ndim != 1:
        raise ValueError("qnums should be a one-dimensional array")
    if len(qnums) == 0:
        raise ValueError("qnums should not be empty")
    if M is not None and int(M) != len(qnums):
        raise ValueError(f"M={M} is inconsistent with len(qnums)={len(qnums)}")
    return qnums


def _check_chain_length(L: int):
    L = int(L)
    if L <= 0:
        raise ValueError(f"L should be positive, got {L}")
    return L


def _check_delta_eta(delta, eta):
    if eta is None:
        if delta is None:
            raise ValueError("either delta or eta should be supplied")
        delta = float(delta)
        if not (-1.0 < delta < 1.0):
            raise ValueError(f"massless XXZ solver expects -1 < delta < 1, got {delta}")
        eta = float(np.arccos(delta))
    else:
        eta = float(eta)
        if not (0.0 < eta < np.pi):
            raise ValueError(f"eta should satisfy 0 < eta < pi, got {eta}")
        if delta is None:
            delta = float(np.cos(eta))
        else:
            delta = float(delta)
            if not np.isclose(delta, np.cos(eta)):
                raise ValueError("delta and eta are inconsistent")
    return delta, eta
