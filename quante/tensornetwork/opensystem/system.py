# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-01 00:00:00

"""System-side adapter for open-system tensor-network algorithms."""

from __future__ import annotations

import numpy as np
from scipy import sparse as sps
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator

from ...generate.dynamics import LiouvillianDynamics


class System:
    """Open-system view of a local system dynamics.

    ``Dynamics`` describes the generator. ``System`` turns that generator into
    explicit local gates needed by TEMPO-style tensor contractions.
    """

    def __init__(
        self,
        dynamics: LiouvillianDynamics,
        *,
        sparse: bool = False,
    ):
        if not isinstance(dynamics, LiouvillianDynamics):
            raise TypeError("System requires a LiouvillianDynamics object")

        self.dynamics = dynamics
        self.sparse = sparse
        self._propagator_cache = {}
        self.dim = self.dynamics.dim

    @classmethod
    def from_liouvillian(cls, ham=None, jump_ops=None, *, sparse: bool = False, **kwargs):
        """Build a system from Lindblad data."""
        return cls(LiouvillianDynamics(ham=ham, jump_ops=jump_ops, **kwargs), sparse=sparse)

    @classmethod
    def from_hamiltonian(cls, ham, *, sparse: bool = False, **kwargs):
        """Build a closed local system from a Hamiltonian."""
        return cls.from_liouvillian(ham=ham, sparse=sparse, **kwargs)

    @property
    def time_dependent(self):
        return self.dynamics.time_dependent

    def explicit_generator(self, t: float | None = None):
        """Return the explicit local generator matrix."""
        matrix = self.dynamics.explicit(t)
        if isinstance(matrix, LinearOperator):
            raise TypeError("TEMPO requires an explicit local propagator")
        if self.sparse:
            return matrix if sps.issparse(matrix) else sps.csr_array(matrix)
        return matrix.toarray() if sps.issparse(matrix) else np.asarray(matrix)

    def propagator(self, dt: float, *, t: float | None = None):
        """Return ``exp(A(t) dt)`` as an explicit local gate."""
        if not self.time_dependent:
            key = round(float(dt), 14)
            if key not in self._propagator_cache:
                self._propagator_cache[key] = self._expm(self.explicit_generator(None), dt)
            return self._propagator_cache[key]
        if t is None:
            raise ValueError("t must be supplied for time-dependent system dynamics")
        return self._expm(self.explicit_generator(t), dt)

    def half_step_propagators(self, dt: float, *, t: float | None = None):
        """Return the two local half-step gates used by TEMPO."""
        if self.time_dependent:
            if t is None:
                raise ValueError("t must be supplied for time-dependent system dynamics")
            prop_1 = self.propagator(dt / 2.0, t=t)
            prop_2 = self.propagator(dt / 2.0, t=t + dt / 2.0)
            return prop_1, prop_2
        prop = self.propagator(dt / 2.0)
        return prop, prop

    def _expm(self, matrix, dt: float):
        expm_fn = sps.linalg.expm if self.sparse else expm
        return expm_fn(matrix * dt)


def as_system(system: System | LiouvillianDynamics) -> System:
    """Return ``system`` as an open-system adapter."""
    if isinstance(system, System):
        return system
    if isinstance(system, LiouvillianDynamics):
        return System(system)
    raise TypeError("system should be a System or LiouvillianDynamics object")


__all__ = ["System", "as_system"]
