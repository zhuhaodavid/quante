# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-27 13:17:07
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-27 13:31:18


import numpy as np
import scipy.integrate

from .system import as_matrix


def _complex_quad(func, a, b, *, epsrel=1e-8, limit=200):
    """Integrate a complex-valued scalar function by integrating both parts."""
    real = scipy.integrate.quad(
        lambda x: np.real(func(x)), a, b, epsrel=epsrel, limit=limit
    )[0]
    imag = scipy.integrate.quad(
        lambda x: np.imag(func(x)), a, b, epsrel=epsrel, limit=limit
    )[0]
    return real + 1j * imag


class Correlation:
    """Light base class for bath correlation functions."""

    def __call__(self, tau):
        """Evaluate the correlation function at time difference ``tau``."""
        return self.corr(tau)

    def corr(self, tau):
        """Return ``C(tau)`` for the bath; subclasses provide the formula."""
        raise NotImplementedError

    def correlation_2d_integral(
        self,
        delta,
        time_1,
        time_2=None,
        *,
        shape="square",
        epsrel=1e-8,
        subdiv_limit=200,
    ):
        """Integrate ``C(t' - t'')`` over a TEMPO influence cell.

        Parameters
        ----------
        delta : float
            Length of the short integration interval.
        time_1 : float
            Lower bound of the outer ``t'`` integral.
        time_2 : float | None
            Upper bound for the outer integral. Only used by ``rectangle``.
        shape : {"square", "upper-triangle", "lower-triangle", "rectangle"}
            Integration region. OQuPy uses square, upper-triangle, and
            rectangle; lower-triangle is included here as the complement inside
            the square.
        epsrel : float
            Relative error tolerance passed to SciPy.
        subdiv_limit : int
            Maximum number of subdivisions passed to SciPy.
        """
        if time_2 is None:
            time_2 = time_1 + delta
        elif shape != "rectangle":
            raise ValueError("time_2 can only be used with shape='rectangle'")

        lower_boundary = {
            "square": lambda x: 0.0,
            "upper-triangle": lambda x: 0.0,
            "lower-triangle": lambda x: x - time_1,
            "rectangle": lambda x: 0.0,
        }
        upper_boundary = {
            "square": lambda x: delta,
            "upper-triangle": lambda x: x - time_1,
            "lower-triangle": lambda x: delta,
            "rectangle": lambda x: delta,
        }
        if shape not in lower_boundary:
            raise NotImplementedError(f"shape {shape!r} is not implemented")

        def real_part(t2, t1):
            return np.real(self.corr(t1 - t2))

        def imag_part(t2, t1):
            return np.imag(self.corr(t1 - t2))

        real = scipy.integrate.dblquad(
            real_part,
            time_1,
            time_2,
            lower_boundary[shape],
            upper_boundary[shape],
            epsrel=epsrel,
        )[0]
        imag = scipy.integrate.dblquad(
            imag_part,
            time_1,
            time_2,
            lower_boundary[shape],
            upper_boundary[shape],
            epsrel=epsrel,
        )[0]
        return real + 1j * imag

    def integral(self, t1, t2, dt, *, shape="square"):
        """Convenience wrapper using integer cell indices.

        ``t1`` and ``t2`` are cell indices. The method maps the cell pair to
        the OQuPy-style ``correlation_2d_integral`` interface.
        """
        time_1 = (t1 - t2) * dt
        if shape == "rectangle":
            return self.correlation_2d_integral(
                dt, time_1, time_1 + dt, shape=shape
            )
        return self.correlation_2d_integral(dt, time_1, shape=shape)

    def coefficients(self, n_steps, dt):
        """Return TEMPO influence coefficients for memory offsets.

        Following the TEMPO convention used by OQuPy, the equal-time cell
        ``k=0`` uses the upper triangular region and all later memory offsets
        use square cells.
        """
        coeffs = [self.integral(0, 0, dt, shape="upper-triangle")]
        coeffs.extend(self.integral(k, 0, dt) for k in range(1, n_steps + 1))
        return np.array(coeffs, dtype=np.complex128)


class CustomCorrelation(Correlation):
    """Correlation object backed by a user supplied callable."""

    def __init__(self, func):
        """Store a vectorizable callable ``func(tau)``."""
        if not callable(func):
            raise TypeError("func should be callable")
        self.func = func

    def corr(self, tau):
        """Evaluate the wrapped callable at ``tau``."""
        return self.func(tau)


class PowerLawSD(Correlation):
    """Power-law spectral density with exponential cutoff."""

    def __init__(self, alpha, zeta, cutoff, *, temperature=0.0, cutoff_type="exponential"):
        """Store parameters for ``J(w) = 2 alpha w^zeta wc^(1-zeta) X``."""
        if cutoff_type != "exponential":
            raise ValueError("only exponential cutoff is supported")
        if temperature != 0.0:
            raise NotImplementedError("finite temperature is not implemented yet")
        self.alpha = alpha
        self.zeta = zeta
        self.cutoff = cutoff
        self.temperature = temperature
        self.cutoff_type = cutoff_type

    def spectral_density(self, omega):
        """Return the spectral density ``J(omega)`` for nonnegative frequencies."""
        omega = np.asarray(omega)
        res = (
            2
            * self.alpha
            * omega**self.zeta
            * self.cutoff ** (1 - self.zeta)
            * np.exp(-omega / self.cutoff)
        )
        res = np.where(omega >= 0, res, 0.0)
        return res.item() if res.ndim == 0 else res

    def corr(self, tau):
        """Return ``C(tau) = int_0^inf J(w) exp(-i w tau) dw`` at ``T=0``."""
        tau_arr = np.asarray(tau)

        def one(t):
            return _complex_quad(
                lambda w: self.spectral_density(w) * np.exp(-1j * w * t),
                0,
                np.inf,
            )

        if tau_arr.ndim == 0:
            return one(float(tau_arr))
        return np.vectorize(one, otypes=[np.complex128])(tau_arr)

    def eta_function(self, tau, *, epsrel=1e-8, subdiv_limit=200):
        """Return the primitive used for OQuPy-style 2D influence integrals."""
        def integrand(w):
            if w == 0:
                return 0.0
            return (
                self.spectral_density(w)
                / w**2
                * ((np.exp(-1j * w * tau) - 1.0) + 1j * w * tau)
            )

        integral = _complex_quad(
            integrand, 0.0, self.cutoff, epsrel=epsrel, limit=subdiv_limit
        )
        integral += _complex_quad(
            integrand, self.cutoff, np.inf, epsrel=epsrel, limit=subdiv_limit
        )
        return -integral

    def correlation_2d_integral(
        self,
        delta,
        time_1,
        time_2=None,
        *,
        shape="square",
        epsrel=1e-8,
        subdiv_limit=200,
    ):
        """Evaluate TEMPO 2D integrals via ``eta_function``.

        This follows OQuPy's spectral-density route: square, upper-triangle and
        rectangle regions can be written as finite differences of ``eta``.
        ``lower-triangle`` is implemented as ``square - upper-triangle``.
        """
        eta = lambda t: self.eta_function(
            t, epsrel=epsrel, subdiv_limit=subdiv_limit
        )
        if shape == "upper-triangle":
            return eta(time_1 + delta) - eta(time_1)
        if shape == "square":
            return eta(time_1 + delta) - 2.0 * eta(time_1) + eta(time_1 - delta)
        if shape == "lower-triangle":
            return self.correlation_2d_integral(
                delta, time_1, shape="square", epsrel=epsrel, subdiv_limit=subdiv_limit
            ) - self.correlation_2d_integral(
                delta,
                time_1,
                shape="upper-triangle",
                epsrel=epsrel,
                subdiv_limit=subdiv_limit,
            )
        if shape == "rectangle":
            if time_2 is None:
                raise ValueError("time_2 is required for shape='rectangle'")
            return eta(time_2) - eta(time_1) - eta(time_2 - delta) + eta(time_1 - delta)
        raise NotImplementedError(f"shape {shape!r} is not implemented")


class Bath:
    """Bath coupling operator and correlation function."""

    def __init__(self, coupling, corr, *, basis=None, pauli=False):
        """Store coupling matrix and wrap callable correlations when needed."""
        self.coupling = as_matrix(coupling, basis=basis, pauli=pauli, sparse=False)
        self.coupling = np.asarray(self.coupling)
        self.corr = corr if isinstance(corr, Correlation) else CustomCorrelation(corr)
        if self.coupling.ndim != 2 or self.coupling.shape[0] != self.coupling.shape[1]:
            raise ValueError("coupling should be a square matrix")
        self.dim = self.coupling.shape[0]

    def diagonalize_coupling(self):
        """Return eigenvalues and eigenvectors of the coupling operator."""
        return np.linalg.eigh(self.coupling)


def plot_correlations_with_parameters(correlation, params, ax=None):
    """Plot a bath correlation function on the TEMPO memory grid.

    This mirrors OQuPy's quickstart helper in a lightweight form. Diamond and
    circle markers show the actual TEMPO memory grid points, while solid lines
    show a denser sampling of the real and imaginary parts.
    """
    import matplotlib.pyplot as plt

    dt = params.dt
    dkmax = params.memory_steps()
    times_infl = dt / 3.0 * np.arange((dkmax + 1) * 3 - 2)
    sample = [3 * i for i in range(dkmax + 1)]
    times_extra = np.linspace(times_infl[-1], times_infl[-1] * 1.5, 10)

    corr = np.vectorize(correlation.corr)
    corr_infl = corr(times_infl)
    corr_extra = corr(times_extra)

    show = False
    if ax is None:
        fig, ax = plt.subplots()
        show = True
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$C(\tau)$")

    ax.plot(times_infl, np.real(corr_infl), color="C0", linestyle="-", label="real")
    ax.scatter(times_infl[sample], np.real(corr_infl[sample]), marker="d", color="C0")
    ax.plot(times_infl, np.imag(corr_infl), color="C1", linestyle="-", label="imag")
    ax.scatter(times_infl[sample], np.imag(corr_infl[sample]), marker="o", color="C1")
    ax.plot(times_extra, np.real(corr_extra), color="C0", linestyle="-")
    ax.plot(times_extra, np.imag(corr_extra), color="C1", linestyle="-")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.axvline(dt * dkmax, color="k", linestyle="dashed")
    ax.legend()

    if show:
        fig.show()
    return ax
