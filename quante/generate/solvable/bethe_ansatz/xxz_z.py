# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-06 17:55:06
# @Last Modified by:   hzhu
# @Last Modified time: 2026-06-06 19:43:25
import numpy as np
from scipy import integrate
from scipy.linalg import solve
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .xxz import solve_xxz_state, xxz_energy, xxz_energy_from_rapidities


__all__ = [
    "XXZGroundStateDensity",
    "p_prime",
    "theta_prime",
    "bare_energy",
    "solve_dressed_energy_fixed_B",
    "evaluate_dressed_energy",
    "epsilon_boundary",
    "find_fermi_B",
    "solve_root_density_fixed_B",
    "energy_density",
    "compute_ground_state_density",
    "xxz_pbc_finite_sector_energies",
    "xxz_pbc_finite_ground_energy",
    "xxz_pbc_infinite_ground_energy",
    "plot_xxz_pbc_finite_energy_vs_h",
    "plot_xxz_root_distribution",
]


@dataclass
class XXZGroundStateDensity:
    """Root-density solution for the infinite periodic XXZ chain."""

    eta: float
    h: float
    B: float
    alpha: np.ndarray
    weight: np.ndarray
    rho: np.ndarray
    epsilon_alpha: np.ndarray
    epsilon: np.ndarray
    energy_density: float

    def filling_density(self):
        """Return the density of down spins, ``N_down / L``."""
        if self.B == 0:
            return 0.0
        if np.isinf(self.B):
            return 0.5
        return float(np.sum(self.weight * self.rho))

    def magnetization_density(self, *, pauli=True):
        """Return magnetization per site."""
        magnetization = 1.0 - 2.0 * self.filling_density()
        if not pauli:
            magnetization /= 2.0
        return float(magnetization)

    def total_magnetization(self, L, *, pauli=True):
        """Return magnetization for a finite length estimated from this density."""
        return float(int(L) * self.magnetization_density(pauli=pauli))

    def asdict(self):
        return {
            "eta": self.eta,
            "h": self.h,
            "B": self.B,
            "alpha": self.alpha,
            "weight": self.weight,
            "rho": self.rho,
            "epsilon_alpha": self.epsilon_alpha,
            "epsilon": self.epsilon,
            "energy_density": self.energy_density,
        }

    def __getitem__(self, key):
        return self.asdict()[key]

    def keys(self):
        return self.asdict().keys()

    def items(self):
        return self.asdict().items()


def _check_delta(delta):
    delta = float(delta)
    if not (-1.0 < delta < 1.0):
        raise ValueError(f"XXZ Bethe-ansatz helpers currently support -1 < delta < 1, got {delta}")
    return delta


def _eta_from_delta(delta):
    return float(np.arccos(_check_delta(delta)))


# ============================================================
# Basic functions
# ============================================================

def p_prime(alpha, eta):
    """
    p'(alpha) for the convention u = -eta/2 + i alpha/2.
    """
    return np.sin(eta) / (np.cosh(alpha) - np.cos(eta))


def theta_prime(x, eta):
    """
    theta'(x) for the two-body scattering phase.
    """
    return np.sin(2 * eta) / (np.cosh(x) - np.cos(2 * eta))


def bare_energy(alpha, eta, h):
    """
    Bare energy cost of adding one Bethe root:
        2 h - 4 sin^2 eta / (cosh alpha - cos eta)
    """
    return 2 * h - 4 * np.sin(eta)**2 / (np.cosh(alpha) - np.cos(eta))


# ============================================================
# Quadrature
# ============================================================

def gauss_legendre_on_minus_B_B(B, n_quad):
    """
    Gauss-Legendre nodes and weights on [-B, B].
    """
    x, w = np.polynomial.legendre.leggauss(n_quad)
    alpha = B * x
    weight = B * w
    return alpha, weight


# ============================================================
# Solve dressed energy for a fixed B
# ============================================================

def solve_dressed_energy_fixed_B(B, eta, h, n_quad=240):
    """
    Solve

        epsilon_B(beta)
        =
        2h - 4 sin^2 eta / (cosh beta - cos eta)
        - 1/(2 pi) int_{-B}^B theta'(alpha - beta) epsilon_B(alpha) d alpha

    for a fixed B.
    """

    alpha, weight = gauss_legendre_on_minus_B_B(B, n_quad)

    # Discretized equation:
    #
    # epsilon_i
    # + 1/(2 pi) sum_j weight_j theta'(alpha_j - alpha_i) epsilon_j
    # =
    # 2h - 4 sin^2 eta / (cosh alpha_i - cos eta)

    diff = alpha[None, :] - alpha[:, None]
    matrix = np.eye(n_quad) + (weight[None, :] / (2 * np.pi)) * theta_prime(diff, eta)

    rhs = bare_energy(alpha, eta, h)

    epsilon = solve(matrix, rhs, assume_a="gen")

    return alpha, weight, epsilon


def evaluate_dressed_energy(beta, alpha, weight, epsilon, eta, h):
    """
    Evaluate epsilon_B(beta) from the integral equation after epsilon_B(alpha)
    has been solved on quadrature nodes.
    """

    return (
        bare_energy(beta, eta, h)
        - np.sum(weight * theta_prime(alpha - beta, eta) * epsilon) / (2 * np.pi)
    )


def epsilon_boundary(B, eta, h, n_quad=240):
    """
    Compute epsilon_B(B).
    """

    alpha, weight, epsilon = solve_dressed_energy_fixed_B(B, eta, h, n_quad=n_quad)

    return evaluate_dressed_energy(B, alpha, weight, epsilon, eta, h)


# ============================================================
# Find B from epsilon_B(B) = 0
# ============================================================

def find_fermi_B(eta, h, n_quad=240, B_min=1e-6, B_max=40.0, n_scan=80):
    """
    Find B such that epsilon_B(B) = 0.

    If epsilon_B(0+) > 0, the root sea is empty, so B = 0.
    If no crossing is found up to B_max, the solution is probably B = infinity
    or B_max is too small.
    """

    value_min = epsilon_boundary(B_min, eta, h, n_quad=n_quad)

    if value_min > 0:
        print("epsilon_B(0+) > 0: the sea is empty. Returning B = 0.")
        return 0.0

    B_values = np.linspace(B_min, B_max, n_scan)
    values = np.array([epsilon_boundary(B, eta, h, n_quad=n_quad) for B in B_values])

    for i in range(len(B_values) - 1):
        if values[i] == 0:
            return B_values[i]

        if values[i] * values[i + 1] < 0:
            B_left = B_values[i]
            B_right = B_values[i + 1]

            B_root = brentq(
                lambda B: epsilon_boundary(B, eta, h, n_quad=n_quad),
                B_left,
                B_right,
                xtol=1e-11,
                rtol=1e-11,
                maxiter=100,
            )

            return B_root

    raise RuntimeError(
        "No finite B found. This may mean B is very large or infinite. "
        "Try increasing B_max, or check whether h is close to zero."
    )


# ============================================================
# Solve root density after B is known
# ============================================================

def solve_root_density_fixed_B(B, eta, n_quad=240):
    """
    Solve

        rho_B(alpha)
        =
        1/(2 pi) p'(alpha)
        - 1/(2 pi) int_{-B}^B theta'(alpha - beta) rho_B(beta) d beta

    for a fixed B.
    """

    alpha, weight = gauss_legendre_on_minus_B_B(B, n_quad)

    # Discretized equation:
    #
    # rho_i
    # + 1/(2 pi) sum_j weight_j theta'(alpha_i - alpha_j) rho_j
    # =
    # 1/(2 pi) p'(alpha_i)

    diff = alpha[:, None] - alpha[None, :]
    matrix = np.eye(n_quad) + (weight[None, :] / (2 * np.pi)) * theta_prime(diff, eta)

    rhs = p_prime(alpha, eta) / (2 * np.pi)

    rho = solve(matrix, rhs, assume_a="gen")

    return alpha, weight, rho


# ============================================================
# Energy density
# ============================================================

def energy_density(B, eta, h, alpha, weight, rho):
    """
    e(B) =
    cos eta - h
    + int_{-B}^B [2h - 4 sin^2 eta/(cosh alpha - cos eta)] rho_B(alpha) d alpha
    """

    integral = np.sum(weight * bare_energy(alpha, eta, h) * rho)

    return np.cos(eta) - h + integral


# ============================================================
# Main driver
# ============================================================

def compute_ground_state_density(eta, h, n_quad=240, B_max=40.0):
    """
    1. Find B from epsilon_B(B) = 0.
    2. Solve rho_B(alpha).
    3. Return data.
    """
    eta = float(eta)
    h = float(h)
    if h == 0:
        alpha = np.linspace(-float(B_max), float(B_max), int(n_quad))
        rho = _zero_field_root_density(alpha, eta)
        return XXZGroundStateDensity(
            eta=eta,
            h=h,
            B=np.inf,
            alpha=alpha,
            weight=np.array([]),
            rho=rho,
            epsilon_alpha=np.array([]),
            epsilon=np.array([]),
            energy_density=_zero_field_infinite_energy_density(eta),
        )

    B = find_fermi_B(eta, h, n_quad=n_quad, B_max=B_max)

    if B == 0.0:
        return XXZGroundStateDensity(
            eta=eta,
            h=h,
            B=0.0,
            alpha=np.array([]),
            weight=np.array([]),
            rho=np.array([]),
            epsilon_alpha=np.array([]),
            epsilon=np.array([]),
            energy_density=np.cos(eta) - h,
        )

    epsilon_alpha, epsilon_weight, epsilon = solve_dressed_energy_fixed_B(
        B, eta, h, n_quad=n_quad
    )

    alpha, weight, rho = solve_root_density_fixed_B(
        B, eta, n_quad=n_quad
    )

    e = energy_density(B, eta, h, alpha, weight, rho)

    return XXZGroundStateDensity(
        eta=eta,
        h=h,
        B=B,
        alpha=alpha,
        weight=weight,
        rho=rho,
        epsilon_alpha=epsilon_alpha,
        epsilon=epsilon,
        energy_density=e,
    )


# ============================================================
# Public finite/infinite XXZ helpers with longitudinal field
# ============================================================

def xxz_pbc_finite_sector_energies(
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
    """
    L = int(L)
    if L <= 0:
        raise ValueError(f"L should be positive, got {L}")
    _check_delta(delta)
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

    branch_delta = _check_delta(delta if j > 0 else -delta)
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


def xxz_pbc_finite_ground_energy(
    L,
    j=1.0,
    delta=1.0,
    *,
    h=0.0,
    pauli=False,
    tol=1e-12,
    raise_error=True,
):
    r"""Ground-state energy for the finite periodic XXZ chain with field."""
    _check_delta(delta)
    if j == 0:
        return -abs(float(h)) * _sector_magnetizations(int(L), [0, int(L)], pauli=pauli).max()

    branch_delta = _check_delta(delta if j > 0 else -delta)
    L = int(L)
    if h == 0:
        state = solve_xxz_state(
            L,
            branch_delta,
            M=L // 2,
            tol=tol,
            raise_error=raise_error,
        )
        return xxz_energy(state, j=abs(j), pauli=pauli)

    return _finite_field_ground_energy_scalar(
        L,
        branch_delta,
        h=float(h),
        j=abs(j),
        pauli=pauli,
        tol=tol,
        raise_error=raise_error,
    )


def xxz_pbc_infinite_ground_energy(
    j=1.0,
    delta=1.0,
    *,
    h=0.0,
    pauli=False,
    n_quad=240,
    B_max=40.0,
    epsrel=1e-12,
    limit=256,
):
    r"""Ground-state energy density for the infinite periodic XXZ chain.

    This uses the massless-regime TBA equation in this file and currently
    supports only ``-1 < delta < 1`` after the ``j < 0`` branch mapping.
    """
    _check_delta(delta)
    if j == 0:
        return -abs(float(h)) * (1.0 if pauli else 0.5)

    branch_delta = _check_delta(delta if j > 0 else -delta)
    eta = _eta_from_delta(branch_delta)
    scaled_h = abs(float(h)) / abs(j)
    if scaled_h == 0:
        energy_density = abs(j) * _zero_field_infinite_energy_density(
            eta,
            epsrel=epsrel,
            limit=limit,
        )
    else:
        result = compute_ground_state_density(
            eta=eta,
            h=scaled_h,
            n_quad=n_quad,
            B_max=B_max,
        )
        energy_density = abs(j) * result["energy_density"]
    if not pauli:
        energy_density /= 4.0
    return float(np.real_if_close(energy_density))


def plot_xxz_pbc_finite_energy_vs_h(
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
    data = xxz_pbc_finite_sector_energies(
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


def plot_xxz_root_distribution(
    delta,
    h,
    *,
    j=1.0,
    n_quad=240,
    B_max=40.0,
    ax=None,
):
    """Plot the infinite-chain root density at fixed field and return ``(fig, ax, data)``."""
    if j == 0:
        raise ValueError("root distribution is not defined for j=0")
    branch_delta = _check_delta(delta if j > 0 else -delta)
    data = compute_ground_state_density(
        eta=_eta_from_delta(branch_delta),
        h=abs(float(h)) / abs(j),
        n_quad=n_quad,
        B_max=B_max,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.6))
    else:
        fig = ax.figure

    if data["B"] > 0:
        ax.plot(data["alpha"], data["rho"], label=r"$\rho_B(\alpha)$")
    else:
        ax.plot([], [], label=r"$\rho_B(\alpha)$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho_B(\alpha)$")
    ax.set_title(fr"XXZ root distribution, $\Delta={delta}$, $h={h}$")
    ax.legend()
    fig.tight_layout()
    return fig, ax, data


def _finite_zero_field_sector_energy(L, delta, n_down, *, j, pauli, tol, raise_error):
    n_ba = min(int(n_down), int(L) - int(n_down))
    if n_ba == 0:
        return xxz_energy_from_rapidities([], int(L), delta, j=j, pauli=pauli)
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
    return xxz_energy(state, j=j, pauli=pauli)


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
        data = compute_ground_state_density(
            eta=_eta_from_delta(delta),
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


def _zero_field_infinite_energy_density(eta, *, epsrel=1e-12, limit=256):
    def integrand(alpha):
        with np.errstate(over="ignore"):
            return (
                1.0 / np.cosh(np.pi * alpha / (2.0 * eta))
                / (np.cosh(alpha) - np.cos(eta))
            )

    integral = integrate.quad(
        integrand,
        -np.inf,
        np.inf,
        epsrel=epsrel,
        limit=limit,
    )[0]
    return np.cos(eta) - np.sin(eta) ** 2 / eta * integral


def _zero_field_root_density(alpha, eta):
    return 1.0 / (4.0 * eta * np.cosh(np.pi * np.asarray(alpha) / (2.0 * eta)))
