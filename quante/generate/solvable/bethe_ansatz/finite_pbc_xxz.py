# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-05 00:00:00

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.optimize import root

from .bethe_state import BetheState, SixVertexBetheState


__all__ = [
    "bethe_quantum_numbers",
    "solve_xxz_state",
    "map_alpha_to_u",
    "energy",
    "energy_density",
    "energy_from_rapidities",
    "sound_velocity",
    "half_filling_linear_edge_spectrum",
    "ground_energy",
    "plot_energy_vs_h",
]


# ============================================================
# Basic functions
# ============================================================

def bethe_quantum_numbers(M: int, *, center: float = 0.0):
    r"""Return ``M`` consecutive Bethe quantum numbers centered at ``center``.

    In this logarithmic branch, ground-state real-root quantum numbers are
    integers for odd ``M`` and half-integers for even ``M``.
    """
    M = int(M)
    if M <= 0:
        raise ValueError(f"M should be positive, got {M}")
    return np.arange(M, dtype=float) - (M - 1) / 2.0 + center


def map_alpha_to_u(alphas, eta: float):
    """Map alpha branch coordinates to ``u = -eta / 2 + 1j * alpha / 2``."""
    return -float(eta) / 2.0 + 0.5j * np.asarray(alphas)


def xxz_u_roots(state: BetheState):
    """Compatibility helper. Prefer ``state.roots``."""
    return state.roots


# ============================================================
# Finite real-root Bethe solver
# ============================================================

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

    This helper is intentionally restricted to the finite real-root,
    highest-weight branch. Use spin flip outside this function for sectors
    with more than half of the spins down. Complex string solutions require a
    different parametrization and are not represented here.
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


# ============================================================
# Energies and simple spectra
# ============================================================


def energy_from_rapidities(
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


def sound_velocity(
    delta: float | None = None,
    *,
    eta: float | None = None,
    j: float = 1.0,
    pauli: bool = True,
):
    r"""Return the zero-field half-filled XXZ sound velocity.

    The spin-operator convention gives

    .. math::
        v_s = J \frac{\pi \sin\eta}{2\eta},
        \qquad \Delta = \cos\eta.

    The Pauli-matrix convention is four times this value.
    """
    _, eta = _check_delta_eta(delta, eta)
    velocity = abs(float(j)) * np.pi * np.sin(eta) / (2.0 * eta)
    if pauli:
        velocity *= 4.0
    return float(velocity)


def half_filling_linear_edge_spectrum(
    L: int,
    delta: float | None = None,
    *,
    eta: float | None = None,
    m_values=None,
    branch: str = "right",
    j: float = 1.0,
    pauli: bool = True,
):
    r"""Return the low-energy half-filled edge spectrum ``(P_m, dE_m)``.

    At zero field the half-filled Fermi rapidities sit at ``alpha = +/-inf``.
    The finite-real-root solver therefore cannot represent an edge particle
    placed outside the Bethe sea.  For the version-A wave-packet construction,
    the appropriate Bethe/TBA data near the Fermi point are

    .. math::
        \Delta P_m = \pm 2\pi m/L,\qquad
        \Delta E_m = v_s |\Delta P_m|.

    This helper returns that linearized Fermi-edge branch.
    """
    L = _check_chain_length(L)
    if branch not in {"right", "left"}:
        raise ValueError("branch should be 'right' or 'left'")
    if m_values is None:
        m_values = np.arange(1, L // 4 + 1)
    m_values = np.asarray(m_values, dtype=int)
    if np.any(m_values <= 0):
        raise ValueError("m_values should be positive integers")

    sign = 1.0 if branch == "right" else -1.0
    momentum = sign * 2.0 * np.pi * m_values / L
    denergy = sound_velocity(delta, eta=eta, j=j, pauli=pauli) * np.abs(momentum)
    return momentum.astype(float), denergy.astype(float)


# ============================================================
# Logarithmic Bethe equation internals
# ============================================================

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
    max_real_roots = int(L) // 2
    if qnums is None:
        if M is None:
            M = max_real_roots
        M = _check_real_root_count(L, M)
        return bethe_quantum_numbers(M)

    qnums = np.asarray(qnums, dtype=float)
    if qnums.ndim != 1:
        raise ValueError("qnums should be a one-dimensional array")
    if len(qnums) == 0:
        raise ValueError("qnums should not be empty")
    if M is not None and int(M) != len(qnums):
        raise ValueError(f"M={M} is inconsistent with len(qnums)={len(qnums)}")
    M_qnums = _check_real_root_count(L, len(qnums))
    _check_qnums_finite(qnums)
    _check_qnums_lattice(qnums, M_qnums)
    _check_qnums_distinct(qnums)
    _check_qnums_range(L, qnums)
    return qnums


def _check_real_root_count(L: int, M: int):
    M = int(M)
    if M <= 0:
        raise ValueError(f"M should be positive for solve_xxz_state, got {M}")
    max_real_roots = int(L) // 2
    if M > max_real_roots:
        raise ValueError(
            "solve_xxz_state only supports the finite real-root highest-weight "
            f"branch with M <= L//2; got M={M}, L={L}. Use spin flip for the "
            "opposite magnetization sector, or a separate complex-root solver "
            "for string states."
        )
    return M


def _check_qnums_finite(qnums):
    if not np.all(np.isfinite(qnums)):
        raise ValueError("qnums should contain only finite real numbers")


def _check_qnums_distinct(qnums):
    q2 = np.rint(2.0 * np.asarray(qnums, dtype=float)).astype(np.int64)
    if len(np.unique(q2)) != len(q2):
        raise ValueError("qnums should be distinct")


def _check_qnums_lattice(qnums, M):
    q2 = 2.0 * np.asarray(qnums, dtype=float)
    q2_round = np.rint(q2)
    if not np.allclose(q2, q2_round, atol=1e-10, rtol=0.0):
        raise ValueError("qnums should be integers or half-integers")

    parity = (q2_round.astype(np.int64) + int(M) - 1) % 2
    if np.any(parity != 0):
        raise ValueError(
            "qnums have the wrong integer/half-integer parity for this M: "
            "use integers for odd M and half-integers for even M."
        )


def _check_qnums_range(L, qnums):
    edge = 0.5 * int(L)
    qnums = np.asarray(qnums, dtype=float)
    if np.any(np.abs(qnums) >= edge):
        raise ValueError(
            "qnums should satisfy abs(qnum) < L/2 for this finite real-root "
            f"branch; got range [{np.min(qnums)}, {np.max(qnums)}] with L={L}."
        )


# ============================================================
# Finite-field sector envelope
# ============================================================

def ground_energy(
    L,
    j=1.0,
    delta=1.0,
    *,
    h=0.0,
    pauli=False,
    tol=1e-12,
    raise_error=True,
):
    r"""Ground-state energy for the finite periodic XXZ chain with field.

    For ``h < 0`` this function uses spin-flip symmetry and optimizes the
    positive-field minority-root branch. The returned energy is physical, but
    the internally selected Bethe sector is the flipped one.
    """
    delta = _check_delta_eta(delta, None)[0]
    if h < 0:
        warnings.warn(
            "xxz_pbc_finite_ground_energy received h < 0. Using spin-flip "
            "symmetry and the positive-field minority-root branch; only the "
            "energy is returned, so the flipped sector is not exposed.",
            RuntimeWarning,
            stacklevel=2,
        )
    if j == 0:
        return -abs(float(h)) * _sector_magnetizations(int(L), [0, int(L)], pauli=pauli).max()

    branch_delta = _check_delta_eta(delta if j > 0 else -delta, None)[0]
    L = int(L)
    if h == 0:
        state = solve_xxz_state(
            L,
            branch_delta,
            M=L // 2,
            tol=tol,
            raise_error=raise_error,
        )
        return state.xxz_energy(j=abs(j), pauli=pauli)

    return _finite_field_ground_energy_scalar(
        L,
        branch_delta,
        h=float(h),
        j=abs(j),
        pauli=pauli,
        tol=tol,
        raise_error=raise_error,
    )


def plot_energy_vs_h(
    L,
    delta,
    h_values,
    *,
    j=1.0,
    pauli=True,
    ax=None,
    tol=1e-12,
    raise_error=True,
    show_sector_lines=True,
):
    """Plot the finite-size XXZ ``E-h`` sector envelope and return ``(fig, ax, data)``."""
    data = _sector_energies(
        L,
        delta,
        h_values,
        j=j,
        pauli=pauli,
        tol=tol,
        raise_error=raise_error,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.6))
    else:
        fig = ax.figure

    n_downs = data["n_downs"]
    h_values = data["h_values"]
    sector_energies = data["sector_energies"]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(n_downs.min(), n_downs.max())

    if show_sector_lines:
        for n_down, energy_h in zip(n_downs, sector_energies):
            ax.plot(h_values, energy_h, color=cmap(norm(n_down)), linewidth=0.75, alpha=0.55)

    ax.plot(h_values, data["ground_energy"], color="black", linewidth=2.2, label="ground-state envelope")
    switch_idx = np.flatnonzero(np.diff(data["ground_sector"]) != 0)
    for idx in switch_idx:
        h_mid = 0.5 * (h_values[idx] + h_values[idx + 1])
        ax.axvline(h_mid, color="0.45", linestyle=":", linewidth=0.8)

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$E(h)$")
    ax.set_title(fr"XXZ sectors with field, $L={L}$, $\Delta={delta}$")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, pad=0.015)
    colorbar.set_label(r"$N_\downarrow$")
    fig.tight_layout()
    return fig, ax, data


# ============================================================
# Finite-field helper routines
# ============================================================

def _sector_energies(
    L,
    delta,
    h_values,
    *,
    j=1.0,
    pauli=True,
    tol=1e-12,
    raise_error=True,
):
    r"""Return all finite-size sector energies in a longitudinal field.

    The field convention is

    .. math::
        E(h) = E(0) - h M_z,

    where ``M_z = L - 2 N_down`` in the Pauli convention and
    ``M_z = (L - 2 N_down) / 2`` in the spin-operator convention.

    Negative ``h`` is supported here because all magnetization sectors are
    explicitly included.
    """
    L = int(L)
    if L <= 0:
        raise ValueError(f"L should be positive, got {L}")
    delta = _check_delta_eta(delta, None)[0]
    h_values = np.atleast_1d(np.asarray(h_values, dtype=float))
    if j == 0:
        n_downs = np.arange(L + 1)
        magnetizations = _sector_magnetizations(L, n_downs, pauli=pauli)
        sector_energies = -h_values[None, :] * magnetizations[:, None]
        return {
            "n_downs": n_downs,
            "magnetizations": magnetizations,
            "zero_field_energies": np.zeros(L + 1),
            "h_values": h_values,
            "sector_energies": sector_energies,
            "ground_energy": np.min(sector_energies, axis=0),
            "ground_sector": n_downs[np.argmin(sector_energies, axis=0)],
        }

    branch_delta = _check_delta_eta(delta if j > 0 else -delta, None)[0]
    n_downs = np.arange(L + 1)
    magnetizations = _sector_magnetizations(L, n_downs, pauli=pauli)
    zero_field_energies = np.array([
        _finite_zero_field_sector_energy(
            L,
            branch_delta,
            int(n_down),
            j=abs(j),
            pauli=pauli,
            tol=tol,
            raise_error=raise_error,
        )
        for n_down in n_downs
    ])
    sector_energies = zero_field_energies[:, None] - h_values[None, :] * magnetizations[:, None]
    ground_index = np.argmin(sector_energies, axis=0)

    return {
        "n_downs": n_downs,
        "magnetizations": magnetizations,
        "zero_field_energies": zero_field_energies,
        "h_values": h_values,
        "sector_energies": sector_energies,
        "ground_energy": sector_energies[ground_index, np.arange(len(h_values))],
        "ground_sector": n_downs[ground_index],
    }


def _finite_zero_field_sector_energy(L, delta, n_down, *, j, pauli, tol, raise_error):
    n_ba = min(int(n_down), int(L) - int(n_down))
    if n_ba == 0:
        return energy_from_rapidities([], int(L), delta, j=j, pauli=pauli)
    try:
        state = solve_xxz_state(
            int(L),
            delta,
            M=n_ba,
            tol=tol,
            raise_error=raise_error,
        )
    except RuntimeError:
        state = solve_xxz_state(
            int(L),
            delta,
            M=n_ba,
            tol=max(tol, 1e-10),
            method="lm",
            raise_error=raise_error,
        )
    return state.xxz_energy(j=j, pauli=pauli)


def _finite_field_ground_energy_scalar(L, delta, *, h, j, pauli, tol, raise_error):
    abs_h = abs(float(h))
    max_n_down = int(L) // 2
    cache = {}

    def sector_energy(n_down):
        n_down = int(n_down)
        if n_down not in cache:
            zero_field_energy = _finite_zero_field_sector_energy(
                L,
                delta,
                n_down,
                j=j,
                pauli=pauli,
                tol=tol,
                raise_error=raise_error,
            )
            magnetization = _sector_magnetizations(L, n_down, pauli=pauli)
            cache[n_down] = zero_field_energy - abs_h * magnetization
        return cache[n_down]

    if max_n_down <= 64:
        n_down = _minimize_integer_by_scan(sector_energy, 0, max_n_down)
    else:
        estimate_h = abs_h / j * (2.0 if not pauli else 1.0)
        n_down = _estimate_finite_ground_sector(L, delta, h=estimate_h)
        if n_down is None:
            n_down = _minimize_integer_unimodal(sector_energy, 0, max_n_down)
        else:
            n_down = _minimize_near_integer_guess(sector_energy, n_down, 0, max_n_down)
    return float(sector_energy(n_down))


def _minimize_integer_by_scan(func, lo, hi):
    candidates = np.arange(int(lo), int(hi) + 1)
    values = np.array([func(n) for n in candidates])
    return int(candidates[np.argmin(values)])


def _estimate_finite_ground_sector(L, delta, *, h):
    if h == 0:
        return int(L) // 2
    try:
        from .infinite_pbc_xxz import compute_ground_state_density

        _, eta = _check_delta_eta(delta, None)
        data = compute_ground_state_density(
            eta=eta,
            h=h,
            n_quad=120,
            B_max=40.0,
        )
    except Exception:
        return None

    filling = min(max(data.filling_density(), 0.0), 0.5)
    return int(round(int(L) * filling))


def _minimize_near_integer_guess(func, guess, lo, hi):
    lo = int(lo)
    hi = int(hi)
    guess = int(np.clip(int(guess), lo, hi))
    radius = max(4, min(32, (hi - lo + 1) // 100))

    while True:
        left = max(lo, guess - radius)
        right = min(hi, guess + radius)
        candidates = np.arange(left, right + 1)
        values = np.array([func(n) for n in candidates])
        best = int(candidates[np.argmin(values)])

        if (best != left or left == lo) and (best != right or right == hi):
            return best
        if left == lo and right == hi:
            return best

        guess = best
        radius = min(hi - lo + 1, radius * 2)


def _minimize_integer_unimodal(func, lo, hi):
    lo = int(lo)
    hi = int(hi)
    while hi - lo > 8:
        third = (hi - lo) // 3
        m1 = lo + third
        m2 = hi - third
        if func(m1) <= func(m2):
            hi = m2 - 1
        else:
            lo = m1 + 1

    candidates = np.arange(lo, hi + 1)
    values = np.array([func(n) for n in candidates])
    return int(candidates[np.argmin(values)])


def _sector_magnetizations(L, n_downs, *, pauli):
    magnetizations = int(L) - 2 * np.asarray(n_downs, dtype=float)
    if not pauli:
        magnetizations /= 2.0
    return magnetizations


# ============================================================
# Validation helpers
# ============================================================

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
