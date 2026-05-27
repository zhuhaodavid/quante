# -*- coding: utf-8 -*-

import numpy as np


class TempoResult:
    """Lightweight container for reduced density matrices."""

    def __init__(self, times, states):
        """Store time points and the corresponding reduced density matrices."""
        self.times = np.asarray(times)
        self.states = np.asarray(states)

    def measure(self, obs=None, *, real=False):
        """Measure an observable on stored states.

        ``obs=None`` returns the stored density matrices. Matrix observables use
        ``trace(rho @ obs)``. Callable observables are called as ``obs(t, rho)``.
        """
        if obs is None:
            return self.states
        if callable(obs):
            vals = np.array([obs(t, rho) for t, rho in zip(self.times, self.states)])
        elif isinstance(obs, list):
            vals = np.array([[np.trace(rho @ ob) for ob in obs] for rho in self.states])
        else:
            obs = np.asarray(obs)
            vals = np.array([np.trace(rho @ obs) for rho in self.states])
        vals = vals.real if real else np.real_if_close(vals)
        return self.times, vals

    def state(self, ind=-1):
        """Return one stored density matrix by index."""
        return self.states[ind]
