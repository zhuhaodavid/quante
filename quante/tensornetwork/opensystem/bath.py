# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-27 21:53:55
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-28 02:49:54

import numpy as np
from numpy import ndarray

from ...linalg.matops import super as opr
from .bath_correlation import SpectralDensity

from typing import Callable, Optional, Text

from scipy import integrate
from functools import lru_cache


class Bath:
    def __init__(
            self,
            coupling_oper: ndarray,
            spectral_density: SpectralDensity,
        ) -> None:
        assert np.allclose(np.diag(coupling_oper.diagonal()), coupling_oper), \
            "Coupling operator must be hermitian."
        self.dim = coupling_oper.shape[0]
        
        if np.allclose(np.diag(coupling_oper.diagonal()), coupling_oper):
            self.coupling_operator = coupling_oper
            self.unitary = None
        else:
            w, v = np.linalg.eigh(coupling_oper)
            self.coupling_operator = np.diag(w)
            self.unitary = v
            assert np.allclose(coupling_oper, \
                self._unitary @ self._coupling_operator \
                @ self._unitary.conjugate().T), \
                "singular coupling operator."
        
        self.spectral_density = spectral_density
        self.coupling_comm = opr.commutator(self.coupling_operator).diagonal()
        self.coupling_acomm = opr.acommutator(self.coupling_operator).diagonal()


def _complex_integral(
        integrand: Callable[[float], complex],
        a: Optional[float] = 0.0,
        b: Optional[float] = 1.0,
        epsrel: Optional[float] = 2**(-26),
        limit: Optional[int] = 256,
    ) -> complex:
    re_int = integrate.quad(lambda x: np.real(integrand(x)),
                            a=a,
                            b=b,
                            epsrel=epsrel,
                            limit=limit)[0]
    im_int = integrate.quad(lambda x: np.imag(integrand(x)),
                            a=a,
                            b=b,
                            epsrel=epsrel,
                            limit=limit)[0]

    return re_int + 1j * im_int

def _hard_cutoff(omega , omega_c: float):
    """Hard cutoff function."""
    return np.heaviside(omega_c - omega, 0)

def _exponential_cutoff(omega, omega_c: float):
    """Exponential cutoff function."""
    return np.exp(-omega / omega_c)

def _gaussian_cutoff(omega, omega_c: float):
    """Gaussian cutoff function."""
    return np.exp(-(omega / omega_c) ** 2)

# dictionary for the various cutoffs in the form:
#   'cutoff_name': cutoff_function
CUTOFF_DICT = {
    'hard': _hard_cutoff,
    'exponential': _exponential_cutoff,
    'gaussian': _gaussian_cutoff,
}

class SpectralDensity:
    r"""
    Correlations corresponding to a custom spectral density for a thermal
    system with known temperature. The resulting spectral density is

    .. math::

        J(\omega) = j(\omega) X(\omega,\omega_c) ,

    with `j_function` :math:`j`, `cutoff` :math:`\omega_c` and a cutoff type
    :math:`X`.

    If `cutoff_type` is

    - ``'hard'`` then
      :math:`X(\omega,\omega_c)=\Theta(\omega_c-\omega)`, where
      :math:`\Theta` is the Heaviside step function.
    - ``'exponential'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega/\omega_c)`.
    - ``'gaussian'`` then
      :math:`X(\omega,\omega_c)=\exp(-\omega^2/\omega_c^2)`.
    """
    def __init__(
            self,
            j_function: Callable[[float], float],
            cutoff: float,
            cutoff_type: Text = 'exponential',
            temperature: Optional[float] = 0.0,
        ) -> None:
        self.j_function = j_function
        self.cutoff = cutoff
        self.cutoff_type = cutoff_type
        self.temperature = temperature

        self._cutoff_function = \
            lambda omega: CUTOFF_DICT[self.cutoff_type](omega, self.cutoff)
        self._spectral_density = \
            lambda omega: self.j_function(omega) * self._cutoff_function(omega)

    def correlation_2d_integral(
            self,
            delta: float,
            t1: float,
            t2: float | None = None,
            shape: Text | None = 'square',
            epsrel: float | None = 2**(-26),
            subdiv_limit: int | None = 256,
        ) -> complex:
        r"""
        2D integrals of the correlation function

        .. math::

            \eta_\mathrm{square} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

            \eta_\mathrm{upper-triangle} =
            \int_{t_1}^{t_1+\Delta} \int_{0}^{t'-t_1} C(t'-t'') dt'' dt'

            \eta_\mathrm{rectangle} =
            \int_{t_1}^{t_2} \int_{0}^{\Delta} C(t'-t'') dt'' dt'

        for `shape` either ``'square'``, ``'upper-triangle'``,
        or ``'rectangle'``.

        """
        kwargs = {
            'epsrel': epsrel,
            'subdiv_limit': subdiv_limit,
        }

        if shape == 'upper-triangle':
            integral = self.eta_function(t1 + delta, **kwargs) \
                       - self.eta_function(t1, **kwargs)
        elif shape == 'square':
            integral = self.eta_function(t1 + delta, **kwargs) \
                       - 2.0 * self.eta_function(t1, **kwargs) \
                       + self.eta_function(t1 - delta, **kwargs)
        elif shape == 'rectangle':
            integral = self.eta_function(t2, **kwargs) \
                       - self.eta_function(t1, **kwargs) \
                       - self.eta_function(t2 - delta, **kwargs) \
                       + self.eta_function(t1 - delta, **kwargs)
        else:
            raise NotImplementedError("Shape '{shape}' not implemented.")

        return integral

    @lru_cache(maxsize=2 ** 10, typed=False)
    def eta_function(
            self,
            tau,
            epsrel: float | None = 2**(-26),
            subdiv_limit: int | None = 256,
        ) -> ndarray:
        r"""
        Auto-correlation function associated to the spectral density at the
        given temperature :math:`T`

        .. math::

            C(\tau) = \int_0^{\infty} J(\omega) \
                    \left[ \cos(\omega \tau) \
                            \coth\left( \frac{\omega}{2 T}\right) \
                            - i \sin(\omega \tau) \right] \mathrm{d}\omega .

        with time difference `tau` :math:`\tau`.
        """
        if self.temperature == 0.0:
            def integrand(w):
                return self._spectral_density(w) / w ** 2 * (
                    (np.exp(-1j * w * tau) - 1) + 1j * w * tau
                )
        else:
            def integrand(w):
                # this is to stop overflow
                if np.exp(-w / self.temperature) > np.finfo(float).eps:
                    inte = self._spectral_density(w) / w ** 2 \
                        * (((np.exp(-1j*tau * w) \
                            + np.exp(-(w / self.temperature - 1j*tau * w))) \
                            - np.exp(- w / self.temperature) - 1) \
                        / (1 - np.exp(-w / self.temperature)) + 1j*tau * w)
                else:
                    inte = self._spectral_density(w) / w ** 2 \
                        * (np.exp(-1j * w * tau) - 1 + 1j * w * tau)
                return inte

        integral = _complex_integral(integrand,
                                    a=0.0,
                                    b=self.cutoff,
                                    epsrel=epsrel,
                                    limit=subdiv_limit)

        if self.cutoff_type != "hard":
            integral += _complex_integral(integrand,
                                        a=self.cutoff,
                                        b=np.inf,
                                        epsrel=epsrel,
                                        limit=subdiv_limit)
        return -integral


class PowerLawSpectralDensity(SpectralDensity):
    r"""
    Correlations corresponding to the spectral density of the standard form

    .. math::

        J(\omega) = 2 \alpha \frac{\omega^\zeta}{\omega_c^{\zeta-1}} \
                    X(\omega,\omega_c)

    The environment is called 
    
    - *ohmic* if :math:`\zeta=1`,
    - *superohmic* if :math:`\zeta>1`,
    - *subohmic* if :math:`\zeta<1`.
    """

    def __init__(
            self,
            alpha: float,
            zeta: float,
            cutoff: float,
            cutoff_type: Text = 'exponential',
            temperature: Optional[float] = 0.0,
        ) -> None:
        """Create a StandardSD (spectral density) object. """

        self.alpha = alpha
        self.zeta = zeta
        self.cutoff = cutoff

        j_function = lambda w: 2.0 * self.alpha * w ** self.zeta \
                               * self.cutoff ** (1 - zeta)


        super().__init__(j_function,
                         cutoff=cutoff,
                         cutoff_type=cutoff_type,
                         temperature=temperature
        )

