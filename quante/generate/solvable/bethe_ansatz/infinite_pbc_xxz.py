# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-06 17:55:06
# @Last Modified by:   hzhu
# @Last Modified time: 2026-06-11 11:16:53
import numpy as np
import warnings
from scipy import integrate
from scipy.linalg import solve
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .bethe_state import XXZBetheKernel


__all__ = [
    "XXZBetheKernel",
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
    "ground_energy",
    "plot_xxz_root_distribution",
]


@dataclass
class XXZGroundStateDensity:
    """Root-density solution for the infinite periodic XXZ chain."""

    delta: float
    regime: str
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
            "delta": self.delta,
            "regime": self.regime,
            "eta": self.eta,
            "parameter": self.eta,
            "parameter_name": "eta" if self.regime == "massless" else "gamma",
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
    

    def dressed_energy(self, alphas, chunk_size=2048):
        kernel = XXZBetheKernel.from_delta(self.delta)
        h = float(self.h) 
        alpha_eval = np.asarray(alphas, dtype=float)

        if h == 0.0 and kernel.regime == "massive":
            return _zero_field_massive_dressed_energy(alpha_eval, kernel.parameter)

        # These beta points are the sea rapidities where epsilon_B(beta) was solved.
        beta = np.asarray(self.epsilon_alpha, dtype=float)
        eps_beta = np.asarray(self.epsilon, dtype=float)
        weight = np.asarray(self.weight, dtype=float)

        # Start from the bare energy.
        epsilon_full = kernel.bare_energy(alpha_eval, h)

        # If B = 0, the sea is empty, so there is no dressing integral.
        if self.B == 0 or len(beta) == 0:
            return epsilon_full

        weighted_eps = weight * eps_beta

        # Evaluate
        #
        # epsilon(alpha)
        # = bare(alpha)
        #   - 1/(2 pi) int_{-B}^{B} epsilon(beta) theta'(alpha-beta) d beta
        #
        # using quadrature on the solved sea points beta.
        for start in range(0, len(alpha_eval), chunk_size):
            stop = min(start + chunk_size, len(alpha_eval))
            a = alpha_eval[start:stop]

            K = kernel.theta_prime(a[:, None] - beta[None, :])
            correction = K @ weighted_eps

            epsilon_full[start:stop] -= correction / (2.0 * np.pi)

        return epsilon_full

    def plot_dressed_energy(self, alphas, *, n_solved_points=600):
        alphas = np.asarray(alphas, dtype=float)
        epsilon_plot = self.dressed_energy(alphas)
        fig, ax = plt.subplots()
        ax.plot(alphas, epsilon_plot, label=r"full $\varepsilon_B(\alpha)$", c='b')
        xlim1, xlim2 = ax.get_xlim()
        ax.axhline(0, color="0.7", linestyle="--", linewidth=0.8)
        solved_alphas = np.linspace(float(np.min(alphas)), float(np.max(alphas)), int(n_solved_points))
        solved_epsilon = self.dressed_energy(solved_alphas)
        solved_mask = solved_epsilon < 0.0
        ax.plot(
            solved_alphas[solved_mask],
            solved_epsilon[solved_mask],
            ".",
            color="orange",
            markersize=3,
            label=r"solved points in $[-B,B]$",
        )
        ax.fill_between(
            alphas,
            epsilon_plot,
            0,
            where=epsilon_plot < 0.0,
            interpolate=True,
            color="orange",
            alpha=0.25,
            label=r"$\varepsilon_B(\alpha)<0$",
        )
        ax.axvline(0, color="0.7", linestyle="--", linewidth=0.8)
        B = self.B
        if self.B != 0:
            ax.text(-B, 0.0, r"$-B$", ha="right", va="bottom")
            ax.text(B, 0.0, r"$B$", ha="left", va="bottom")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\varepsilon_B(\alpha)$")
        ax.set_title(
            rf"Dressed energy, $\Delta={self.delta:.4g}$, $h={self.h:.4g}$, $B={self.B:.4g}$"
        )
        ax.set_xlim(xlim1, xlim2)
        ax.legend(loc="lower left")
        return epsilon_plot, ax


        
def _check_delta(delta):
    delta = float(delta)
    if not (-1.0 < delta < 1.0):
        raise ValueError(f"XXZ Bethe-ansatz helpers currently support -1 < delta < 1, got {delta}")
    return delta


def _check_ground_delta(delta):
    delta = float(delta)
    if delta != -1.0 and delta != 1.0:
        return delta
    raise ValueError(
        "XXZ infinite ground_energy currently excludes the singular isotropic "
        f"points delta=+/-1, got {delta}"
    )


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

def solve_dressed_energy_fixed_B(B, kernel: XXZBetheKernel, h, n_quad=240):
    """
    Solve

        epsilon_B(beta)
        =
        bare_energy(beta)
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
    matrix = np.eye(n_quad) + (weight[None, :] / (2 * np.pi)) * kernel.theta_prime(diff)

    rhs = kernel.bare_energy(alpha, h)

    epsilon = solve(matrix, rhs, assume_a="gen")

    return alpha, weight, epsilon


def evaluate_dressed_energy(beta, alpha, weight, epsilon, kernel: XXZBetheKernel, h):
    """
    Evaluate epsilon_B(beta) from the integral equation after epsilon_B(alpha)
    has been solved on quadrature nodes.
    """

    return (
        kernel.bare_energy(beta, h)
        - np.sum(weight * kernel.theta_prime(alpha - beta) * epsilon) / (2 * np.pi)
    )


def epsilon_boundary(B, kernel: XXZBetheKernel, h, n_quad=240):
    """
    Compute epsilon_B(B).
    """

    alpha, weight, epsilon = solve_dressed_energy_fixed_B(B, kernel, h, n_quad=n_quad)

    return evaluate_dressed_energy(B, alpha, weight, epsilon, kernel, h)


# ============================================================
# Find B from epsilon_B(B) = 0
# ============================================================

def find_fermi_B(kernel: XXZBetheKernel, h, n_quad=240, B_min=1e-6, B_max=40.0, n_scan=80):
    """
    Find B such that epsilon_B(B) = 0.

    If epsilon_B(0+) > 0, the root sea is empty, so B = 0.
    If no crossing is found up to B_max, the solution is probably B = infinity
    or B_max is too small.
    """

    if kernel.regime == "massive_negative":
        return 0.0

    if kernel.regime == "massive":
        value_max = epsilon_boundary(np.pi, kernel, h, n_quad=n_quad)
        if value_max <= 0.0:
            return float(np.pi)
        B_max = min(float(B_max), float(np.pi))

    value_min = epsilon_boundary(B_min, kernel, h, n_quad=n_quad)

    if value_min > 0:
        print("epsilon_B(0+) > 0: the sea is empty. Returning B = 0.")
        return 0.0

    B_values = np.linspace(B_min, B_max, n_scan)
    values = np.array([epsilon_boundary(B, kernel, h, n_quad=n_quad) for B in B_values])

    for i in range(len(B_values) - 1):
        if values[i] == 0:
            return B_values[i]

        if values[i] * values[i + 1] < 0:
            B_left = B_values[i]
            B_right = B_values[i + 1]

            B_root = brentq(
                lambda B: epsilon_boundary(B, kernel, h, n_quad=n_quad),
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

def solve_root_density_fixed_B(
    B,
    kernel: XXZBetheKernel,
    n_quad=240,
    *,
    alpha_cut=40.0,
    series_terms=4096,
    series_tol=1e-14,
):
    """
    Solve

        rho_B(alpha)
        =
        1/(2 pi) p'(alpha)
        - 1/(2 pi) int_{-B}^B theta'(alpha - beta) rho_B(beta) d beta

    for a fixed B.

    The kernel chooses the massless or massive XXZ equation. For the
    zero-field closed-form case, pass ``B=np.inf`` for massless or
    ``B=np.pi`` for massive:

    - ``|Delta| < 1`` returns
      ``rho(alpha) = sech(pi alpha / (2 eta)) / (4 eta)`` on
      ``[-alpha_cut, alpha_cut]``.
    - ``Delta > 1`` returns the massive branch with
      ``gamma = arccosh(|delta|)`` on ``[-pi, pi]``:
      ``rho(alpha) = (1 + 2 sum_l cos(l alpha) / cosh(l gamma)) / (4 pi)``.
    """
    B = float(B)

    if kernel.regime == "massive_negative":
        return np.array([]), np.array([]), np.array([])

    if np.isinf(B) or (kernel.regime == "massive" and np.isclose(B, np.pi)):
        if kernel.regime == "massless":
            alpha = np.linspace(-float(alpha_cut), float(alpha_cut), int(n_quad))
            weight = np.array([])
            rho = _zero_field_density(kernel, alpha)
            return alpha, weight, rho

        alpha, weight = gauss_legendre_on_minus_B_B(np.pi, int(n_quad))
        rho = _zero_field_density(
            kernel,
            alpha,
            series_terms=series_terms,
            series_tol=series_tol,
        )
        return alpha, weight, rho

    alpha, weight = gauss_legendre_on_minus_B_B(B, n_quad)

    # Discretized equation:
    #
    # rho_i
    # + 1/(2 pi) sum_j weight_j theta'(alpha_i - alpha_j) rho_j
    # =
    # 1/(2 pi) p'(alpha_i)

    diff = alpha[:, None] - alpha[None, :]
    matrix = np.eye(n_quad) + (weight[None, :] / (2 * np.pi)) * kernel.theta_prime(diff)

    rhs = kernel.p_prime(alpha) / (2 * np.pi)

    rho = solve(matrix, rhs, assume_a="gen")

    return alpha, weight, rho


# ============================================================
# Energy density
# ============================================================

def energy_density(B, kernel: XXZBetheKernel, h, alpha, weight, rho):
    """
    e(B) = reference energy density + int bare_energy(alpha) rho_B(alpha) d alpha
    """

    integral = np.sum(weight * kernel.bare_energy(alpha, h) * rho)

    return kernel.vacuum_energy_density(h) + integral


# ============================================================
# Main driver
# ============================================================

def compute_ground_state_density(
    delta,
    h,
    n_quad=240,
    B_max=40.0,
    *,
    epsrel=1e-12,
    limit=256,
    series_terms=4096,
    series_tol=1e-14,
):
    """
    1. Find B from epsilon_B(B) = 0.
    2. Solve rho_B(alpha).
    3. Return data.
    """
    kernel = XXZBetheKernel.from_delta(delta)
    h = float(h)
    if h < 0:
        warnings.warn(
            "compute_ground_state_density received h < 0. The TBA equations "
            "in this file solve the positive-field minority-root branch; using "
            "abs(h). The returned filling_density is the branch/root filling, "
            "not the spin-flipped physical N_down/L.",
            RuntimeWarning,
            stacklevel=2,
        )
        h = abs(h)

    if kernel.regime == "massive_negative":
        return XXZGroundStateDensity(
            delta=kernel.delta,
            regime=kernel.regime,
            eta=kernel.parameter,
            h=h,
            B=0.0,
            alpha=np.array([]),
            weight=np.array([]),
            rho=np.array([]),
            epsilon_alpha=np.array([]),
            epsilon=np.array([]),
            energy_density=kernel.vacuum_energy_density(h),
        )

    if h == 0:
        B = np.inf if kernel.regime == "massless" else np.pi
        alpha, weight, rho = solve_root_density_fixed_B(
            B,
            kernel,
            n_quad=n_quad,
            alpha_cut=B_max,
            series_terms=series_terms,
            series_tol=series_tol,
        )
        return XXZGroundStateDensity(
            delta=kernel.delta,
            regime=kernel.regime,
            eta=kernel.parameter,
            h=h,
            B=B,
            alpha=alpha,
            weight=weight,
            rho=rho,
            epsilon_alpha=np.array([]),
            epsilon=np.array([]),
            energy_density=_zero_field_energy_density(kernel, epsrel=epsrel, limit=limit),
        )

    search_B_max = min(float(B_max), np.pi) if kernel.regime == "massive" else float(B_max)
    B = find_fermi_B(kernel, h, n_quad=n_quad, B_max=search_B_max)

    if B == 0.0:
        return XXZGroundStateDensity(
            delta=kernel.delta,
            regime=kernel.regime,
            eta=kernel.parameter,
            h=h,
            B=0.0,
            alpha=np.array([]),
            weight=np.array([]),
            rho=np.array([]),
            epsilon_alpha=np.array([]),
            epsilon=np.array([]),
            energy_density=kernel.vacuum_energy_density(h),
        )

    epsilon_alpha, epsilon_weight, epsilon = solve_dressed_energy_fixed_B(
        B, kernel, h, n_quad=n_quad
    )

    alpha, weight, rho = solve_root_density_fixed_B(
        B,
        kernel,
        n_quad=n_quad,
        alpha_cut=B_max,
        series_terms=series_terms,
        series_tol=series_tol,
    )

    e = energy_density(B, kernel, h, alpha, weight, rho)

    return XXZGroundStateDensity(
        delta=kernel.delta,
        regime=kernel.regime,
        eta=kernel.parameter,
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
# Public infinite XXZ helpers with longitudinal field
# ============================================================

def ground_energy(
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

    This uses the massless-regime TBA equation for ``-1 < delta < 1``. At
    zero field it also supports the massive antiferromagnetic branch
    ``delta = cosh(gamma) > 1`` using the closed-form series. For
    ``delta < -1`` it returns the fully polarized ferromagnetic branch.
    """
    _check_ground_delta(delta)
    if h < 0:
        warnings.warn(
            "xxz_pbc_infinite_ground_energy received h < 0. Using spin-flip "
            "symmetry and abs(h) for the positive-field TBA branch; the energy "
            "density is unchanged by this flip.",
            RuntimeWarning,
            stacklevel=2,
        )
    if j == 0:
        return -abs(float(h)) * (1.0 if pauli else 0.5)

    branch_delta = _check_ground_delta(delta if j > 0 else -delta)
    result = compute_ground_state_density(
        delta=branch_delta,
        h=abs(float(h)) / abs(j),
        n_quad=n_quad,
        B_max=B_max,
        epsrel=epsrel,
        limit=limit,
    )
    energy_density = abs(j) * result["energy_density"]
    if not pauli:
        energy_density /= 4.0
    return float(np.real_if_close(energy_density))


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
    if h < 0:
        warnings.warn(
            "plot_xxz_root_distribution received h < 0. Plotting the "
            "spin-flipped positive-field minority-root density using abs(h).",
            RuntimeWarning,
            stacklevel=2,
        )
    branch_delta = _check_ground_delta(delta if j > 0 else -delta)
    data = compute_ground_state_density(
        delta=branch_delta,
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


def _zero_field_energy_density(kernel: XXZBetheKernel, *, epsrel=1e-12, limit=256):
    if kernel.regime == "massless":
        return _zero_field_infinite_energy_density(kernel.parameter, epsrel=epsrel, limit=limit)
    if kernel.regime == "massive":
        return _zero_field_massive_infinite_energy_density(kernel.parameter, epsrel=epsrel, limit=limit)
    return float(kernel.delta)


def _zero_field_massive_infinite_energy_density(gamma, *, epsrel=1e-12, limit=256):
    r"""Zero-field infinite-chain energy density for ``Delta = cosh(gamma) > 1``."""
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError(f"gamma should be positive for Delta > 1, got {gamma}")

    max_terms = max(int(limit), int(np.ceil(40.0 / gamma)))
    series = 0.0
    for ell in range(1, max_terms + 1):
        x = ell * gamma
        if x > 700.0:
            break
        term = np.exp(-x) / np.cosh(x)
        series += term
        if 2.0 * abs(term) < epsrel:
            break
    return np.cosh(gamma) - 2.0 * np.sinh(gamma) * (1.0 + 2.0 * series)


def _zero_field_massive_dressed_energy(
    alpha,
    gamma,
    *,
    series_terms=4096,
    series_tol=1e-14,
):
    alpha = np.asarray(alpha, dtype=float)
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError(f"gamma should be positive for Delta > 1, got {gamma}")

    series = np.ones_like(alpha, dtype=float)
    for ell in range(1, int(series_terms) + 1):
        x = ell * gamma
        if x > 700.0:
            break
        coeff = 2.0 / np.cosh(x)
        if coeff < series_tol:
            break
        series += coeff * np.cos(ell * alpha)
    return -2.0 * np.sinh(gamma) * series


def _zero_field_root_density(alpha, eta):
    return 1.0 / (4.0 * eta * np.cosh(np.pi * np.asarray(alpha) / (2.0 * eta)))


def _zero_field_density(
    kernel: XXZBetheKernel,
    alpha,
    *,
    series_terms=4096,
    series_tol=1e-14,
):
    if kernel.regime == "massless":
        return _zero_field_root_density(alpha, kernel.parameter)
    if kernel.regime == "massive":
        return _zero_field_massive_root_density(
            alpha,
            kernel.parameter,
            series_terms=series_terms,
            series_tol=series_tol,
        )
    return np.zeros_like(np.asarray(alpha, dtype=float))


def _zero_field_massive_root_density(
    alpha,
    gamma,
    *,
    series_terms=4096,
    series_tol=1e-14,
):
    alpha = np.asarray(alpha, dtype=float)
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError(f"gamma should be positive for |Delta| > 1, got {gamma}")

    rho = np.ones_like(alpha, dtype=float)
    for ell in range(1, int(series_terms) + 1):
        x = ell * gamma
        if x > 700.0:
            break
        coeff = 2.0 / np.cosh(x)
        if coeff < series_tol:
            break
        rho += coeff * np.cos(ell * alpha)
    return rho / (4.0 * np.pi)


def _root_density_regime(eta, delta):
    parameter = float(eta)
    if delta is None:
        if not (0.0 < parameter < np.pi):
            raise ValueError(
                "eta should satisfy 0 < eta < pi for the massless root-density "
                f"equation, got {parameter}. Pass delta for |Delta| > 1."
            )
        return "massless", parameter

    delta = float(delta)
    if -1.0 < delta < 1.0:
        expected = np.arccos(delta)
        if not np.isclose(parameter, expected):
            raise ValueError("delta and eta are inconsistent")
        return "massless", parameter

    if abs(delta) > 1.0:
        expected = np.arccosh(abs(delta))
        if not np.isclose(parameter, expected):
            raise ValueError("delta and gamma are inconsistent")
        return "massive", parameter

    raise ValueError(f"zero-field closed-form density is singular at delta={delta}")


def evaluate_full_dressed_energy(
    gs,
    alpha_eval,
    *,
    chunk_size=2048,
):
    kernel = XXZBetheKernel.from_delta(gs.delta)
    h = float(gs.h)

    alpha_eval = np.asarray(alpha_eval, dtype=float)

    # These beta points are the sea rapidities where epsilon_B(beta) was solved.
    beta = np.asarray(gs.epsilon_alpha, dtype=float)
    eps_beta = np.asarray(gs.epsilon, dtype=float)
    weight = np.asarray(gs.weight, dtype=float)

    if beta.shape != eps_beta.shape:
        raise ValueError(
            "gs.epsilon_alpha and gs.epsilon must have the same shape."
        )

    if weight.shape != eps_beta.shape:
        raise ValueError(
            "gs.weight and gs.epsilon must have the same shape. "
            "This function assumes the quadrature weights correspond to epsilon_alpha."
        )

    # Start from the bare energy.
    epsilon_full = kernel.bare_energy(alpha_eval, h)

    # If B = 0, the sea is empty, so there is no dressing integral.
    if gs.B == 0 or len(beta) == 0:
        return epsilon_full

    weighted_eps = weight * eps_beta

    # Evaluate
    #
    # epsilon(alpha)
    # = bare(alpha)
    #   - 1/(2 pi) int_{-B}^{B} epsilon(beta) theta'(alpha-beta) d beta
    #
    # using quadrature on the solved sea points beta.
    for start in range(0, len(alpha_eval), chunk_size):
        stop = min(start + chunk_size, len(alpha_eval))
        a = alpha_eval[start:stop]

        K = kernel.theta_prime(a[:, None] - beta[None, :])
        correction = K @ weighted_eps

        epsilon_full[start:stop] -= correction / (2.0 * np.pi)

    return epsilon_full

