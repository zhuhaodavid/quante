# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-15 00:00:00

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import ndarray
from scipy.optimize import root

from .bethe_state import SixVertexBetheState, XXZBetheKernel


__all__ = [
    "FreeOpenXXZBoundary",
    "DiagonalOpenXXZBoundary",
    "OpenXXZBetheState",
    "bethe_quantum_numbers",
    "solve_xxz_state",
    "energy_from_rapidities",
    "ground_energy",
]


@dataclass(frozen=True)
class FreeOpenXXZBoundary:
    """Free diagonal boundaries for the open XXZ chain."""

    def phase(self, alpha, kernel: XXZBetheKernel):
        alpha = np.asarray(alpha, dtype=float)
        if kernel.is_massless:
            eta = kernel.parameter
            return 4.0 * np.arctan(
                np.tan(eta / 2.0) * np.tanh(alpha / 2.0)
            )
        if kernel.is_massive:
            gamma = kernel.parameter
            return -4.0 * np.arctan(
                np.tanh(gamma / 2.0) * np.tan(alpha / 2.0)
            )
        if kernel.is_isotropic:
            return np.zeros_like(alpha)
        _raise_unsupported_regime()

    def phase_derivative(self, alpha, kernel: XXZBetheKernel):
        alpha = np.asarray(alpha, dtype=float)
        if kernel.is_massless:
            tan_eta_half = np.tan(kernel.parameter / 2.0)
            tanh_alpha_half = np.tanh(alpha / 2.0)
            sech2_alpha_half = 1.0 / np.cosh(alpha / 2.0) ** 2
            return 2.0 * tan_eta_half * sech2_alpha_half / (
                1.0 + tan_eta_half ** 2 * tanh_alpha_half ** 2
            )
        if kernel.is_massive:
            tanh_gamma_half = np.tanh(kernel.parameter / 2.0)
            tan_alpha_half = np.tan(alpha / 2.0)
            return -2.0 * tanh_gamma_half * (1.0 + tan_alpha_half ** 2) / (
                1.0 + tanh_gamma_half ** 2 * tan_alpha_half ** 2
            )
        if kernel.is_isotropic:
            return np.zeros_like(alpha)
        _raise_unsupported_regime()

    def energy_offset(self, kernel: XXZBetheKernel):
        _check_supported_kernel(kernel)
        return 0.0


@dataclass(frozen=True)
class DiagonalOpenXXZBoundary:
    r"""Longitudinal diagonal boundary fields.

    The Pauli-convention Hamiltonian contains
    ``h_minus * sigma_z[0] + h_plus * sigma_z[-1]`` inside the overall
    exchange scale ``j``. With ``pauli=False``, the full Pauli Hamiltonian is
    divided by four, so these terms become
    ``h_minus / 2 * S_z[0] + h_plus / 2 * S_z[-1]``. The implementation
    covers every XXZ anisotropy branch supported by
    :class:`XXZBetheKernel`.

    In the massive regime the ordinary real-root equations require
    ``|h_minus|, |h_plus| < sinh(gamma)`` on the mapped branch. Stronger
    fields generate boundary bound-state roots and need a complex-root
    solver.
    """

    h_minus: float = 0.0
    h_plus: float = 0.0

    def __post_init__(self):
        for name, value in (
            ("h_minus", self.h_minus),
            ("h_plus", self.h_plus),
        ):
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} should be finite, got {value}")
            object.__setattr__(self, name, value)

    def flipped(self):
        """Return the boundary obtained by a global spin flip."""
        return type(self)(-self.h_minus, -self.h_plus)

    def phase(self, alpha, kernel: XXZBetheKernel):
        alpha = np.asarray(alpha, dtype=float)
        h_minus, h_plus = self._mapped_fields(kernel)
        if kernel.is_massless:
            c_minus, c_plus = self._massless_phase_coefficients(kernel)
            tanh_half = np.tanh(alpha / 2.0)
            return (
                2.0 * np.arctan(c_minus * tanh_half)
                + 2.0 * np.arctan(c_plus * tanh_half)
            )
        if kernel.is_massive:
            c_minus, c_plus = self._massive_phase_coefficients(kernel)
            tan_half = np.tan(alpha / 2.0)
            return (
                2.0 * np.arctan(c_minus * tan_half)
                + 2.0 * np.arctan(c_plus * tan_half)
            )
        if kernel.is_isotropic:
            c_minus = _isotropic_boundary_coefficient(h_minus)
            c_plus = _isotropic_boundary_coefficient(h_plus)
            return (
                2.0 * np.arctan(c_minus * alpha)
                + 2.0 * np.arctan(c_plus * alpha)
            )
        _raise_unsupported_regime()

    def phase_derivative(self, alpha, kernel: XXZBetheKernel):
        alpha = np.asarray(alpha, dtype=float)
        h_minus, h_plus = self._mapped_fields(kernel)
        if kernel.is_massless:
            c_minus, c_plus = self._massless_phase_coefficients(kernel)
            tanh_half = np.tanh(alpha / 2.0)
            sech2_half = 1.0 / np.cosh(alpha / 2.0) ** 2
            return sech2_half * (
                c_minus / (1.0 + c_minus ** 2 * tanh_half ** 2)
                + c_plus / (1.0 + c_plus ** 2 * tanh_half ** 2)
            )
        if kernel.is_massive:
            c_minus, c_plus = self._massive_phase_coefficients(kernel)
            tan_half = np.tan(alpha / 2.0)
            sec2_half = 1.0 + tan_half ** 2
            return sec2_half * (
                c_minus / (1.0 + c_minus ** 2 * tan_half ** 2)
                + c_plus / (1.0 + c_plus ** 2 * tan_half ** 2)
            )
        if kernel.is_isotropic:
            c_minus = _isotropic_boundary_coefficient(h_minus)
            c_plus = _isotropic_boundary_coefficient(h_plus)
            return (
                2.0 * c_minus / (1.0 + c_minus ** 2 * alpha ** 2)
                + 2.0 * c_plus / (1.0 + c_plus ** 2 * alpha ** 2)
            )
        _raise_unsupported_regime()

    def energy_offset(self, kernel: XXZBetheKernel):
        h_minus, h_plus = self._mapped_fields(kernel)
        return h_minus + h_plus

    def _mapped_fields(self, kernel):
        _check_supported_kernel(kernel)
        return (
            kernel.energy_sign * self.h_minus,
            kernel.energy_sign * self.h_plus,
        )

    def _massless_phase_coefficients(self, kernel):
        eta = kernel.parameter
        h_minus, h_plus = self._mapped_fields(kernel)
        threshold = 1.0 + np.cos(eta)
        if _at_or_above(h_minus, threshold) or _at_or_above(h_plus, threshold):
            raise ValueError(
                "The massless ordinary real-root boundary equations require "
                "h_minus and h_plus < 1 + delta on the mapped branch; fields "
                "at or above this threshold require complex boundary roots."
            )
        xi_minus = np.arctan2(np.sin(eta), h_minus)
        xi_plus = np.arctan2(np.sin(eta), -h_plus)
        return (
            1.0 / np.tan(xi_minus - eta / 2.0),
            -1.0 / np.tan(xi_plus + eta / 2.0),
        )

    def _massive_phase_coefficients(self, kernel):
        gamma = kernel.parameter
        scale = np.sinh(gamma)
        h_minus, h_plus = self._mapped_fields(kernel)
        ratios = np.array([h_minus / scale, -h_plus / scale])
        if np.any(
            (np.abs(ratios) > 1.0)
            | np.isclose(np.abs(ratios), 1.0, atol=1e-13, rtol=0.0)
        ):
            raise ValueError(
                "The massive ordinary real-root boundary equations require "
                "|h_minus| and |h_plus| < sinh(gamma) on the mapped branch; "
                "stronger fields require complex boundary roots."
            )
        zeta_minus, zeta_plus = np.arctanh(ratios)
        return (
            np.tanh(zeta_minus - gamma / 2.0),
            -np.tanh(zeta_plus + gamma / 2.0),
        )


def _isotropic_boundary_coefficient(field):
    if _at_or_above(field, 2.0):
        raise ValueError(
            "The isotropic ordinary real-root boundary equations require "
            "h_minus and h_plus < 2 on the mapped branch; fields at or above "
            "this threshold require complex boundary roots."
        )
    return field / (2.0 - field)


def _at_or_above(value, threshold):
    return value > threshold or np.isclose(
        value, threshold, atol=1e-13, rtol=0.0
    )


class OpenXXZBetheState(SixVertexBetheState):
    """Bethe state for the finite open XXZ chain."""

    def __init__(self, *args, boundary, **kwargs):
        self.boundary = boundary
        super().__init__(*args, **kwargs)

    def xxz_energy(self, j: float = 1.0, *, pauli: bool = True):
        return energy_from_rapidities(
            self.alphas,
            self.L,
            self.metadata["delta"],
            eta=self.eta,
            j=j,
            pauli=pauli,
            boundary=self.boundary,
        )


def bethe_quantum_numbers(M: int):
    """Return the positive consecutive OBC ground-state quantum numbers."""
    M = int(M)
    if M <= 0:
        raise ValueError(f"M should be positive, got {M}")
    return np.arange(1, M + 1, dtype=float)


def _has_singular_massive_root(alphas):
    """Return whether a massive real-root solution contains alpha = pi mod 2 pi."""
    return np.any(
        np.isclose(np.cos(np.asarray(alphas, dtype=float)), -1.0, atol=1e-10, rtol=0.0)
    )


def _singular_massive_root_error(L, delta, M):
    return RuntimeError(
        "The open-chain massive Bethe equations reached a singular alpha=pi "
        f"root for L={L}, delta={delta}, M={M}. This is not a root-method "
        "failure: the state requires a singular/complex boundary-root "
        "treatment beyond the current ordinary real-root solver."
    )


def solve_xxz_state(
    L: int,
    delta: float | None = None,
    *,
    eta: float | None = None,
    qnums: Optional[ndarray] = None,
    M: int | None = None,
    boundary: FreeOpenXXZBoundary | DiagonalOpenXXZBoundary | None = None,
    x0: Optional[ndarray] = None,
    tol: float = 1e-12,
    method: str = "hybr",
    raise_error: bool = True,
) -> OpenXXZBetheState:
    r"""Solve finite open XXZ Bethe states on the mapped real-root branches.

    The logarithmic equations are

    .. math::
        2L p_\eta(\alpha_j)
        + \varphi_-(\alpha_j) + \varphi_+(\alpha_j)
        - \sum_{k\ne j}\left[
            \theta_\eta(\alpha_j-\alpha_k)
            + \theta_\eta(\alpha_j+\alpha_k)
        \right]
        = 2\pi I_j.

    The massless branch uses ``Delta = cos(eta)``, the isotropic branch uses
    ``|Delta| = 1``, and the massive branch uses
    ``|Delta| = cosh(gamma)``. For ``Delta <= -1``, this is the branch obtained
    from the open-chain staggered-rotation map
    ``H(Delta) ~ -H(-Delta)``.
    The returned spectral roots are
    ``u_j = -eta / 2 + 1j * alpha_j / 2`` or
    ``u_j = -1 / 2 + 1j * alpha_j / 2`` or
    ``u_j = -gamma / 2 + 1j * alpha_j / 2``, respectively.
    """
    L = _check_chain_length(L)
    kernel = _check_supported_xxz_kernel(delta, eta)
    boundary = FreeOpenXXZBoundary() if boundary is None else boundary
    qnums = _prepare_qnums(L, qnums, M)

    if x0 is None:
        x0 = _initial_guess(L, qnums, kernel)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != qnums.shape:
            raise ValueError(
                f"x0 shape {x0.shape} does not match qnums shape {qnums.shape}"
            )

    sol = root(
        _residual,
        x0,
        jac=_jacobian,
        args=(L, kernel, boundary, qnums),
        method=method,
        tol=tol,
    )

    if raise_error and not sol.success:
        # HYBR commonly reports stagnation when the massive ground-state
        # branch terminates at a singular alpha=pi root. LM is diagnostic
        # only here; its singular solution must not be returned.
        if kernel.is_massive and method == "hybr":
            diagnostic = root(
                _residual,
                x0,
                jac=_jacobian,
                args=(L, kernel, boundary, qnums),
                method="lm",
                tol=max(tol, 1e-11),
            )
            if diagnostic.success and _has_singular_massive_root(diagnostic.x):
                raise _singular_massive_root_error(L, delta, qnums.size)
        raise RuntimeError(f"Open-chain Bethe solver did not converge: {sol.message}")

    order = np.argsort(sol.x)
    alphas = np.asarray(sol.x[order], dtype=float)
    qnums = np.asarray(qnums[order], dtype=float)
    if (
        raise_error
        and kernel.is_massive
        and _has_singular_massive_root(alphas)
    ):
        raise _singular_massive_root_error(L, delta, qnums.size)

    def mapper(alphas, _parameter):
        return kernel.map_alpha_to_u(alphas)

    return OpenXXZBetheState(
        L=L,
        boundary=boundary,
        qnums=qnums,
        alphas=alphas,
        map_alpha_to_u=mapper,
        solver=sol,
        eta=kernel.parameter,
        metadata={
            "model": "XXZ",
            "boundary_condition": "open",
            "boundary": boundary.__class__.__name__,
            "h_minus": getattr(boundary, "h_minus", 0.0),
            "h_plus": getattr(boundary, "h_plus", 0.0),
            "delta": kernel.delta,
            "regime": kernel.regime,
            "anisotropy_parameter": kernel.parameter,
            "anisotropy_name": kernel.parameter_name,
            "mapped_delta": kernel.mapped_delta,
        },
        root_branch=kernel.root_branch,
    )


def energy_from_rapidities(
    alphas,
    L: int,
    delta: float,
    *,
    eta: float | None = None,
    j: float = 1.0,
    pauli: bool = True,
    boundary: FreeOpenXXZBoundary | DiagonalOpenXXZBoundary | None = None,
):
    r"""Return the open XXZ energy, including diagonal boundary fields.

    In the Pauli-matrix convention, the massless branch uses

    .. math::
        E = J\left[
            (L-1)\cos\eta
            - \sum_j \frac{4\sin^2\eta}
            {\cosh\alpha_j-\cos\eta}
        \right].

    For ``Delta = cosh(gamma) > 1``, it uses

    .. math::
        E = J\left[
            (L-1)\cosh\gamma
            - \sum_j \frac{4\sinh^2\gamma}
            {\cosh\gamma-\cos\alpha_j}
        \right].

    At ``Delta = 1``, the rational rapidities give

    .. math::
        E = J\left[
            L-1-\sum_j\frac{8}{1+\alpha_j^2}
        \right].

    A diagonal boundary contributes ``h_minus + h_plus`` in the Pauli
    convention. For ``Delta <= -1``, the mapped anisotropy and boundary
    fields are evaluated according to the open-chain staggered rotation.
    """
    L = _check_chain_length(L)
    kernel = _check_supported_xxz_kernel(delta, eta)
    boundary = FreeOpenXXZBoundary() if boundary is None else boundary
    alphas = np.asarray(alphas, dtype=float)
    if kernel.is_massless:
        eta = kernel.parameter
        exchange = (
            (L - 1) * np.cos(eta)
            - np.sum(
                4.0 * np.sin(eta) ** 2
                / (np.cosh(alphas) - np.cos(eta))
            )
        )
    elif kernel.is_massive:
        gamma = kernel.parameter
        exchange = (
            (L - 1) * np.cosh(gamma)
            - np.sum(
                4.0 * np.sinh(gamma) ** 2
                / (np.cosh(gamma) - np.cos(alphas))
            )
        )
    else:
        exchange = (L - 1) - np.sum(8.0 / (1.0 + alphas ** 2))
    boundary_energy = boundary.energy_offset(kernel)
    energy = float(j) * kernel.energy_sign * (
        exchange + boundary_energy
    )
    if not pauli:
        energy /= 4.0
    return float(np.real_if_close(energy))


def ground_energy(
    L: int,
    j: float = 1.0,
    delta: float = 0.0,
    *,
    h_minus: float = 0.0,
    h_plus: float = 0.0,
    pauli: bool = True,
    tol: float = 1e-12,
    raise_error: bool = True,
):
    """Return the finite open-chain ground-state energy.

    For ``j < 0``, the open-chain staggered rotation maps the ground-state
    problem to the positive-coupling branch with
    ``(delta, h_minus, h_plus) -> (-delta, -h_minus, -h_plus)``.

    With nonzero diagonal boundary fields, all real-root magnetization
    sectors are compared. Fields beyond the boundary-bound-state threshold
    are rejected by :class:`DiagonalOpenXXZBoundary`.
    """
    L = _check_chain_length(L)
    j = float(j)
    delta = float(delta)
    if not np.isfinite(j):
        raise ValueError(f"j should be finite, got {j}")
    if not np.isfinite(delta):
        raise ValueError(f"delta should be finite, got {delta}")
    h_minus = float(h_minus)
    h_plus = float(h_plus)
    if not np.isfinite(h_minus) or not np.isfinite(h_plus):
        raise ValueError("h_minus and h_plus should be finite")
    if j == 0.0:
        return 0.0

    branch_delta = delta if j > 0.0 else -delta
    branch_boundary = DiagonalOpenXXZBoundary(
        h_minus if j > 0.0 else -h_minus,
        h_plus if j > 0.0 else -h_plus,
    )
    coupling = abs(j)
    if h_minus == 0.0 and h_plus == 0.0 and branch_delta <= -1.0:
        return energy_from_rapidities(
            [], L, branch_delta, j=coupling, pauli=pauli
        )
    if h_minus == 0.0 and h_plus == 0.0:
        state = solve_xxz_state(
            L,
            branch_delta,
            M=L // 2,
            tol=tol,
            raise_error=raise_error,
        )
        return state.xxz_energy(j=coupling, pauli=pauli)

    if branch_delta <= -1.0:
        if branch_boundary.h_minus * branch_boundary.h_plus < 0.0:
            raise NotImplementedError(
                "For delta <= -1, oppositely directed boundary fields can "
                "produce a domain-wall/complex boundary-root ground state. "
                "The current ordinary real-root solver supports mapped "
                "sector states, but not this ground-state branch."
            )
        polarized = [
            energy_from_rapidities(
                [],
                L,
                branch_delta,
                j=coupling,
                pauli=pauli,
                boundary=boundary,
            )
            for boundary in (branch_boundary, branch_boundary.flipped())
        ]
        return float(np.min(polarized))

    energies = [
        _boundary_sector_energy(
            L,
            branch_delta,
            n_down,
            boundary=branch_boundary,
            j=coupling,
            pauli=pauli,
            tol=tol,
            raise_error=raise_error,
        )
        for n_down in range(L + 1)
    ]
    return float(np.min(energies))


def _boundary_sector_energy(
    L,
    delta,
    n_down,
    *,
    boundary,
    j,
    pauli,
    tol,
    raise_error,
):
    n_down = int(n_down)
    if n_down <= L // 2:
        M = n_down
        sector_boundary = boundary
    else:
        M = L - n_down
        sector_boundary = boundary.flipped()

    if M == 0:
        return energy_from_rapidities(
            [],
            L,
            delta,
            j=j,
            pauli=pauli,
            boundary=sector_boundary,
        )
    state = solve_xxz_state(
        L,
        delta,
        M=M,
        boundary=sector_boundary,
        tol=tol,
        raise_error=raise_error,
    )
    return state.xxz_energy(j=j, pauli=pauli)


def _p(alpha, kernel):
    alpha = np.asarray(alpha)
    if kernel.is_massless:
        return kernel.finite_theta(alpha / 2.0, 1)
    return kernel.finite_theta(alpha, 1)


def _p_derivative(alpha, kernel):
    alpha = np.asarray(alpha)
    if kernel.is_massless:
        return 0.5 * kernel.finite_theta_derivative(alpha / 2.0, 1)
    return kernel.finite_theta_derivative(alpha, 1)


def _theta(alpha, kernel):
    alpha = np.asarray(alpha)
    if kernel.is_massless:
        return kernel.finite_theta(alpha / 2.0, 2)
    return kernel.finite_theta(alpha, 2)


def _theta_derivative(alpha, kernel):
    alpha = np.asarray(alpha)
    if kernel.is_massless:
        return 0.5 * kernel.finite_theta_derivative(alpha / 2.0, 2)
    return kernel.finite_theta_derivative(alpha, 2)


def _residual(alphas, L, kernel, boundary, qnums):
    alphas = np.asarray(alphas, dtype=float)
    diff = alphas[:, None] - alphas[None, :]
    pair_sum = alphas[:, None] + alphas[None, :]
    scatter = _theta(diff, kernel) + _theta(pair_sum, kernel)
    np.fill_diagonal(scatter, 0.0)
    return (
        2.0 * L * _p(alphas, kernel)
        + boundary.phase(alphas, kernel)
        - np.sum(scatter, axis=1)
        - 2.0 * np.pi * np.asarray(qnums, dtype=float)
    )


def _jacobian(alphas, L, kernel, boundary, qnums):
    alphas = np.asarray(alphas, dtype=float)
    diff = alphas[:, None] - alphas[None, :]
    pair_sum = alphas[:, None] + alphas[None, :]
    theta_diff = _theta_derivative(diff, kernel)
    theta_sum = _theta_derivative(pair_sum, kernel)

    jac = theta_diff - theta_sum
    scatter_diag = theta_diff + theta_sum
    np.fill_diagonal(scatter_diag, 0.0)
    diag = (
        2.0 * L * _p_derivative(alphas, kernel)
        + boundary.phase_derivative(alphas, kernel)
        - np.sum(scatter_diag, axis=1)
    )
    np.fill_diagonal(jac, diag)
    return jac


def _initial_guess(L, qnums, kernel):
    momentum = np.pi * np.asarray(qnums, dtype=float) / int(L)
    if kernel.is_massless:
        arg = np.tan(momentum / 2.0) * np.tan(kernel.parameter / 2.0)
        arg = np.clip(arg, -0.95, 0.95)
        return 2.0 * np.arctanh(arg)
    if kernel.is_isotropic:
        return np.tan(momentum / 2.0)
    arg = np.tan(momentum / 2.0) * np.tanh(kernel.parameter / 2.0)
    return 2.0 * np.arctan(arg)


def _prepare_qnums(L, qnums, M):
    if qnums is None:
        M = _check_root_count(L, L // 2 if M is None else M)
        return bethe_quantum_numbers(M)

    qnums = np.asarray(qnums, dtype=float)
    if qnums.ndim != 1 or len(qnums) == 0:
        raise ValueError("qnums should be a nonempty one-dimensional array")
    if M is not None and int(M) != len(qnums):
        raise ValueError(f"M={M} is inconsistent with len(qnums)={len(qnums)}")
    _check_root_count(L, len(qnums))
    if not np.all(np.isfinite(qnums)):
        raise ValueError("qnums should contain only finite values")
    if np.any(qnums <= 0.0):
        raise ValueError("open-chain qnums should be positive")
    if not np.allclose(qnums, np.rint(qnums), atol=1e-10, rtol=0.0):
        raise ValueError("open-chain qnums should be integers")
    if len(np.unique(qnums)) != len(qnums):
        raise ValueError("qnums should be distinct")
    return qnums


def _check_root_count(L, M):
    M = int(M)
    if M <= 0 or M > L // 2:
        raise ValueError(f"M should satisfy 1 <= M <= L//2; got M={M}, L={L}")
    return M


def _check_chain_length(L):
    L = int(L)
    if L <= 1:
        raise ValueError(f"L should be greater than one, got {L}")
    return L


def _check_supported_xxz_kernel(delta, eta):
    if eta is None:
        if delta is None:
            raise ValueError("either delta or eta should be supplied")
        kernel = XXZBetheKernel.from_delta(delta)
    else:
        kernel = XXZBetheKernel.from_delta_parameter(delta, eta)
    _check_supported_kernel(kernel)
    return kernel


def _check_supported_kernel(kernel):
    if kernel.regime not in {
        "massless",
        "isotropic",
        "isotropic_negative",
        "massive",
        "massive_negative",
    }:
        _raise_unsupported_regime()
    return kernel


def _raise_unsupported_regime():
    raise ValueError("unsupported finite OBC XXZ anisotropy")
