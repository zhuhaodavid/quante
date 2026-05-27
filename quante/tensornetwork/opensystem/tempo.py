# -*- coding: utf-8 -*-

from dataclasses import dataclass
import math

import numpy as np
from tqdm import tqdm

from ..networks import MPS, MPO
from .bath import Bath
from .result import TempoResult
from .system import System


@dataclass
class TempoParams:
    """Numerical parameters for TEMPO-style memory evolution."""

    dt: float
    tcut: float
    epsrel: float = 1e-6
    chi_max: int | None = None
    svd_min: float | None = None
    trunc_cut: float | None = None

    def memory_steps(self):
        """Return the number of memory cells kept by ``tcut`` and ``dt``."""
        return int(math.ceil(self.tcut / self.dt))


def _left_right_super(left, right):
    """Return the row-major superoperator for ``left @ rho @ right``."""
    return np.kron(left, right.T)


class TempoEngine:
    """Stateful reference TEMPO engine.

    The first implementation uses only the system propagator. Bath influence and
    tensor-network compression hooks are prepared for the full TEMPO backend.
    """

    def __init__(
        self,
        system,
        bath,
        rho0,
        ts,
        params,
        *,
        unique=False,
        progressbar=True,
        backend="system",
    ):
        """Initialize a stateful reference engine.

        The current version validates inputs, precomputes the system half-step
        propagator, and builds bath influence coefficients. The actual
        non-Markovian memory update is reserved for the next backend phase.
        """
        self.system = system if isinstance(system, System) else System(system)
        self.bath = bath
        if not isinstance(self.bath, Bath):
            raise TypeError("bath should be a Bath instance")
        self.rho = np.asarray(rho0, dtype=np.complex128)
        if self.rho.shape != (self.system.dim, self.system.dim):
            raise ValueError("rho0 shape is inconsistent with system dimension")
        if self.bath.dim != self.system.dim:
            raise ValueError("bath coupling dimension is inconsistent with system dimension")

        self.ts = np.asarray(ts, dtype=float)
        self.params = params
        self.unique = unique
        self.backend = backend
        self.progressbar = progressbar
        self.cur_step = 0
        self.cur_time = self.ts[0] if len(self.ts) else 0.0
        self.memory = None
        self.adt = None
        self.influence_mpo = None
        trunc_cut = params.epsrel if params.trunc_cut is None else params.trunc_cut
        self.trunc_para = (params.chi_max, params.svd_min, trunc_cut)
        self.prop_half = self.system.half_propagator(params.dt)
        self.infl_coeffs = self._build_influence()
        self.infl_mats = self._build_influence_matrices()
        self._reference_data = None
        if self.backend == "reference":
            self._init_reference()
        elif self.backend == "mps":
            self._init_mps()
        elif self.backend != "system":
            raise ValueError("backend should be 'system', 'reference', or 'mps'")

    def _build_influence(self):
        """Build bath influence coefficients for the configured memory length."""
        return self.bath.corr.coefficients(self.params.memory_steps(), self.params.dt)

    def _coupling_liouville_values(self):
        """Return coupling-basis unitary and commutator/anti-commutator values."""
        vals, unitary = np.linalg.eigh(self.bath.coupling)
        d = len(vals)
        left = np.repeat(vals, d)
        right = np.tile(vals, d)
        return vals, unitary, left - right, left + right

    def _build_influence_matrices(self):
        """Build OQuPy-style influence vectors/matrices from bath coefficients."""
        _, _, comm, acomm = self._coupling_liouville_values()
        mats = []
        for k, eta in enumerate(self.infl_coeffs):
            op = eta.real * comm + 1j * eta.imag * acomm
            if k == 0:
                mats.append(np.exp(-comm * op))
            else:
                mats.append(np.exp(-np.outer(op, comm)))
        return mats

    def _init_reference(self):
        """Initialize dense augmented-density-tensor state in coupling basis."""
        self._init_memory_basis()
        self.memory = self._reference_data["to_coupling"] @ self.rho.reshape(-1)

    def _init_mps(self):
        """Prepare the local MPS augmented density tensor."""
        self._init_memory_basis()
        rho_vec = self._reference_data["to_coupling"] @ self.rho.reshape(-1)
        self.adt = MPS.from_vector(
            rho_vec,
            phys_dim=[self.system.dim**2],
            trunc_para=self.trunc_para,
        )
        self.influence_mpo = self._build_influence_mpo()

    def _init_memory_basis(self):
        """Build the coupling-basis transforms used by TEMPO memory updates."""
        _, unitary, _, _ = self._coupling_liouville_values()
        udag = unitary.conj().T
        to_coupling = _left_right_super(udag, unitary)
        from_coupling = _left_right_super(unitary, udag)
        prop = self.system.propagator(self.params.dt)
        self._reference_data = {
            "unitary": unitary,
            "to_coupling": to_coupling,
            "from_coupling": from_coupling,
            "prop": to_coupling @ prop @ from_coupling,
        }

    def _compress_memory(self):
        """Compress the internal memory state.

        This is a no-op in the reference implementation. The tensor-network
        backend will replace it with an SVD/MPS compression step.
        """
        return self.memory

    def _step_memory_tensor(self, memory):
        """Advance a dense augmented density tensor by one TEMPO time step."""
        prop = self._reference_data["prop"]
        new_memory = np.einsum("ab,b...->ab...", prop, memory)
        new_memory *= self.infl_mats[0].reshape((-1,) + (1,) * (new_memory.ndim - 1))

        max_k = min(new_memory.ndim - 1, len(self.infl_mats) - 1)
        for k in range(1, max_k + 1):
            shape = [1] * new_memory.ndim
            shape[0] = self.infl_mats[k].shape[0]
            shape[k] = self.infl_mats[k].shape[1]
            new_memory *= self.infl_mats[k].reshape(shape)

        keep_legs = self.params.memory_steps() + 1
        if new_memory.ndim > keep_legs:
            new_memory = new_memory.sum(axis=-1)
        return new_memory

    def _rho_from_memory_tensor(self, memory):
        """Trace memory legs and transform the reduced state back."""
        rho_vec = memory.sum(axis=tuple(range(1, memory.ndim)))
        rho_vec = self._reference_data["from_coupling"] @ rho_vec
        return rho_vec.reshape(self.system.dim, self.system.dim)

    def _mps_propagators(self, step):
        """Return first/second propagators for an MPS TEMPO step."""
        prop = self._reference_data["prop"]
        return prop, np.eye(self.system.dim**2, dtype=np.complex128)

    def _build_influence_mpo(self):
        """Store reusable influence factors in MPO leg convention."""
        return [np.asarray(infl, dtype=np.complex128) for infl in self.infl_mats]

    def _expand_adt_with_propagator(self, prop):
        """Split the newest ADT leg into a propagated new leg and old leg.

        If the current first site is ``A[b, r]``, the new first two sites
        represent ``prop[a, b] * A[b, r]`` without expanding the whole ADT.
        """
        first = self.adt.data[0]
        if first.shape[0] != 1:
            raise ValueError("ADT should have an open left boundary")
        dim = first.shape[1]
        right = first.shape[2]
        new_first = np.asarray(prop, dtype=np.complex128).reshape(1, dim, dim)
        new_second = np.zeros((dim, dim, right), dtype=np.complex128)
        for i in range(dim):
            new_second[i, i, :] = first[0, i, :]
        self.adt.data = [new_first, new_second] + self.adt.data[1:]
        self.adt.Ss = [None] * (len(self.adt.data) + 1)
        self.adt.L = len(self.adt.data)
        self.adt.llim = 0
        self.adt.rlim = len(self.adt.data) - 1

    def _select_step_mpo(self, step):
        """Build the diagonal influence MPO for the current ADT length."""
        dim = self.system.dim**2
        length = len(self.adt)
        tensors = []

        first = np.zeros((1, dim, dim, dim), dtype=np.complex128)
        infl0 = self.influence_mpo[0]
        for a in range(dim):
            first[0, a, a, a] = infl0[a]
        tensors.append(first)

        for site in range(1, length):
            infl = None if site >= len(self.influence_mpo) else self.influence_mpo[site]
            if site == length - 1:
                tensor = np.zeros((dim, dim, dim, 1), dtype=np.complex128)
                for carry in range(dim):
                    for x in range(dim):
                        tensor[carry, x, x, 0] = 1.0 if infl is None else infl[carry, x]
            else:
                tensor = np.zeros((dim, dim, dim, dim), dtype=np.complex128)
                for carry in range(dim):
                    for x in range(dim):
                        tensor[carry, x, x, carry] = 1.0 if infl is None else infl[carry, x]
            tensors.append(tensor)
        return MPO(tensors)

    def _sum_adt_oldest_leg(self):
        """Sum the oldest ADT physical leg into its left neighbor."""
        if len(self.adt) <= 1:
            return
        last = self.adt.data[-1]
        vec = last.sum(axis=1)[:, 0]
        prev = self.adt.data[-2]
        self.adt.data[-2] = np.einsum("lpr,r->lp", prev, vec).reshape(
            prev.shape[0], prev.shape[1], 1
        )
        self.adt.data.pop()
        self.adt.Ss = [None] * (len(self.adt.data) + 1)
        self.adt.L = len(self.adt.data)
        self.adt.llim = 0
        self.adt.rlim = len(self.adt.data) - 1

    def _apply_first_half_step(self, prop):
        """Apply the first propagator and grow the ADT by one memory leg."""
        self._expand_adt_with_propagator(prop)

    def _apply_influence_mpo(self, mpo):
        """Apply the current influence MPO to the ADT."""
        self.adt.apply_mpo_(mpo, trunc_para=self.trunc_para, normalize=False)

    def _readout_mps(self):
        """Sum old ADT legs and return the reduced density matrix."""
        env = np.ones(self.adt.data[-1].shape[-1], dtype=np.complex128)
        for site in range(len(self.adt.data) - 1, 0, -1):
            tensor = self.adt.data[site]
            env = np.einsum("lpr,p,r->l", tensor, np.ones(tensor.shape[1]), env)
        first = self.adt.data[0]
        rho_vec = np.einsum("lar,r->a", first, env) * np.exp(self.adt.lognm)
        rho_vec = self._reference_data["from_coupling"] @ rho_vec
        return rho_vec.reshape(self.system.dim, self.system.dim)

    def run(self):
        """Advance the engine by one requested time point and return ``rho``."""
        if self.backend == "reference":
            return self._run_reference()
        if self.backend == "mps":
            return self._run_mps()
        return self._run_system()

    def _run_system(self):
        """Advance by one time point using only the system propagator."""
        if self.cur_step >= len(self.ts):
            return self.rho
        if self.cur_step == 0:
            self.cur_step += 1
            return self.rho

        dt = self.ts[self.cur_step] - self.ts[self.cur_step - 1]
        prop = self.system.propagator(dt)
        rho_vec = prop @ self.rho.reshape(-1)
        self.rho = rho_vec.reshape(self.system.dim, self.system.dim)
        self.cur_time = self.ts[self.cur_step]
        self.cur_step += 1
        self._compress_memory()
        return self.rho

    def _run_reference(self):
        """Advance by one time point with dense TEMPO memory."""
        if self.cur_step >= len(self.ts):
            return self.rho
        if self.cur_step == 0:
            self.cur_step += 1
            return self.rho

        self.memory = self._step_memory_tensor(self.memory)
        self.rho = self._rho_from_memory_tensor(self.memory)
        self.cur_time = self.ts[self.cur_step]
        self.cur_step += 1
        return self.rho

    def _run_mps(self):
        """Advance by one time point with the local MPS ADT backend."""
        if self.cur_step >= len(self.ts):
            return self.rho
        if self.cur_step == 0:
            self.cur_step += 1
            return self.rho

        prop_1, _ = self._mps_propagators(self.cur_step - 1)
        self._apply_first_half_step(prop_1)
        mpo = self._select_step_mpo(self.cur_step)
        self._apply_influence_mpo(mpo)

        keep_legs = self.params.memory_steps() + 1
        if len(self.adt) > keep_legs:
            self._sum_adt_oldest_leg()

        self.adt.canonicalize_(trunc_para=self.trunc_para, canonicalform=False)
        self.rho = self._readout_mps()
        self.cur_time = self.ts[self.cur_step]
        self.cur_step += 1
        return self.rho

    def measure(self, obs=None, *, real=False, progressbar=None):
        """Run through all time points and optionally measure an observable.

        If ``obs`` is ``None``, all density matrices are returned. If ``obs`` is
        a matrix, expectation values ``trace(rho @ obs)`` are returned. If it is
        callable, it is called as ``obs(t, rho)``.
        """
        progressbar = self.progressbar if progressbar is None else progressbar
        iterator = tqdm(self.ts, ascii=True) if progressbar else self.ts
        states = []
        vals = []
        for t in iterator:
            rho = self.run()
            if obs is None:
                states.append(rho.copy())
            elif callable(obs):
                vals.append(obs(t, rho))
            elif isinstance(obs, list):
                vals.append([np.trace(rho @ ob) for ob in obs])
            else:
                vals.append(np.trace(rho @ obs))
        if obs is None:
            return np.asarray(states)
        vals = np.asarray(vals)
        vals = vals.real if real else np.real_if_close(vals)
        return self.ts, vals

    def result(self):
        """Run the engine and return a ``TempoResult`` object."""
        return TempoResult(self.ts, self.measure(progressbar=False))


def tempo_compute(system, bath, rho0, ts, params, *, measure=None, real=False, **kwargs):
    """Convenience wrapper around ``TempoEngine``.

    Returns a ``TempoResult`` when ``measure`` is ``None``; otherwise returns
    ``(times, values)`` from ``TempoEngine.measure``.
    """
    engine = TempoEngine(system, bath, rho0, ts, params, **kwargs)
    if measure is None:
        return engine.result()
    return engine.measure(measure, real=real)
