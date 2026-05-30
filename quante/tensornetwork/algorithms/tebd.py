# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-30 00:00:00
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-31 00:40:55

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from ...linalg.evolve.evolve_engine import EvolveEngineBase
from ..networks.mps import MPS


class TEBD(EvolveEngineBase):
    """Time-evolving block decimation for MPS states.

    Parameters
    ----------
    hamiltonian
        Operator with a ``trotter_gates`` method.
    psi0
        Initial MPS state.
    ts
        Target times. If omitted, pass ``tau`` and ``steps``.
    tau, steps
        Uniform time step and number of steps used to construct ``ts``.
    L
        System size. Defaults to ``len(psi0)``.
    """

    def __init__(
        self,
        hamiltonian,
        psi0: MPS,
        ts=None,
        *,
        tau: float | complex | None = None,
        steps: int | None = None,
        L: int | None = None,
        start_time: float = 0.0,
        order: str = "2",
        evolve_type: str = "time",
        N_step: int = 1,
        pauli: bool = False,
        copy_state: bool = True,
        direction: str | None = None,
        svd_alg: str = "svd",
        trunc_para: tuple[int | None, float | None, float | None] = (None, None, None),
        updateS: bool = False,
        normalize: bool = False,
        gate_range: int = 2,
        unitary_gate: bool = False,
    ):
        if ts is None:
            if tau is None or steps is None:
                raise ValueError("Either ts or both tau and steps should be provided")
            ts = start_time + tau * np.arange(1, steps + 1)
        super().__init__(ts, start_time=start_time)

        if not hasattr(hamiltonian, "trotter_gates"):
            raise TypeError("hamiltonian should provide a trotter_gates method")
        if not isinstance(psi0, MPS):
            raise TypeError("psi0 should be an MPS")

        self.hamiltonian = hamiltonian
        self.initial_state = psi0.copy() if copy_state else psi0
        self.cur_state = self.initial_state
        self.psi = self.cur_state
        self.L = len(psi0) if L is None else L

        self.order = order
        self.evolve_type = evolve_type
        self.N_step = N_step
        self.pauli = pauli
        self.gate_options = dict(
            direction=direction,
            svd_alg=svd_alg,
            trunc_para=trunc_para,
            updateS=updateS,
            normalize=normalize,
            gate_range=gate_range,
            unitary_gate=unitary_gate,
        )
        self._gates_cache = {}
        self.resolved_options = {
            "backend": type(self).__name__,
            "order": self.order,
            "evolve_type": self.evolve_type,
            "N_step": self.N_step,
            "pauli": self.pauli,
            "start_time": self.start_time,
            "gate_options": dict(self.gate_options),
        }

    @staticmethod
    def _dt_key(dt):
        dt = np.asarray(dt).item()
        if isinstance(dt, complex):
            return complex(round(dt.real, 14), round(dt.imag, 14))
        return round(float(dt), 14)

    def _gates(self, dt):
        key = self._dt_key(dt)
        if key not in self._gates_cache:
            self._gates_cache[key] = self.hamiltonian.trotter_gates(
                self.L,
                tau=dt,
                order=self.order,
                evolve_type=self.evolve_type,
                N_step=self.N_step,
                pauli=self.pauli,
            )
        return self._gates_cache[key]

    def propagate(self, state: MPS, dt: float | complex):
        positions, gates = self._gates(dt)
        for pos, gate in zip(positions, gates):
            state.apply_gate_(pos, gate, **self.gate_options)
        return state

    def _prepare_measure(self, measure=None, *, pos=None, pauli=False, logscale=False):
        if measure is None:
            return lambda t, state: state.copy()
        if callable(measure):
            return measure
        return lambda t, state: state.measure(measure, pos, pauli=pauli, logscale=logscale)

    def run(self, measure=None, *, pos=None, pauli=False, logscale=False, progressbar: bool = True):
        measure_func = self._prepare_measure(measure, pos=pos, pauli=pauli, logscale=logscale)
        return self._run(measure_func, progressbar=progressbar)

    def plot(self, measure, *, pos=None, pauli=False, logscale=False, progressbar: bool = True, ax=None, **kwargs):
        measure_func = self._prepare_measure(measure, pos=pos, pauli=pauli, logscale=logscale)
        return self._plot(measure_func, ax=ax, **kwargs)
 

TEBDEvolveEngine = TEBD


__all__ = ["TEBD", "TEBDEvolveEngine"]
