# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-02 00:00:00

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from numpy import ndarray
import numpy as np

from .bath import Bath
from .tempo import ApplyMPOMethod, TempoParams, append_mps_
from ..networks import MPS, MPO
from ..core import tensor_operations as top
from ..core.tensor_utils import TruncationError


@dataclass
class ProcessTensor:
    """Process tensor stored as a matrix-product operator chain."""

    hilbert_space_dimension: int
    dt: float
    mpo_tensors: list[ndarray] | None = None
    cap_tensors: list[ndarray] | None = None

    def __post_init__(self):
        self.rho_dimension = self.hilbert_space_dimension ** 2
        self.trace = (
            np.identity(self.hilbert_space_dimension, dtype=complex)
            / np.sqrt(float(self.hilbert_space_dimension))
        ).reshape(-1)
        self.trace_square = self.trace ** 2
        self.mpo_tensors = [] if self.mpo_tensors is None else self.mpo_tensors
        self.cap_tensors = [] if self.cap_tensors is None else self.cap_tensors

    def __len__(self):
        return len(self.mpo_tensors)

    def set_mpo_tensor(self, step: int, tensor: ndarray):
        if step >= len(self.mpo_tensors):
            self.mpo_tensors.extend([None] * (step - len(self.mpo_tensors) + 1))
        self.mpo_tensors[step] = np.asarray(tensor, dtype=complex)

    def get_mpo_tensor(self, step: int, *, full: bool = False):
        tensor = self.mpo_tensors[step]
        if tensor is None:
            return None
        if full and tensor.ndim == 3:
            return _create_delta(tensor, [0, 1, 2, 2])
        return tensor

    def get_cap_tensor(self, step: int):
        if step < 0 or step >= len(self.cap_tensors):
            return None
        return self.cap_tensors[step]

    def get_bond_dimensions(self):
        if len(self.mpo_tensors) == 0:
            return np.array([], dtype=int)
        bond_dims = [tensor.shape[0] if tensor is not None else 0
                     for tensor in self.mpo_tensors]
        bond_dims.append(self.mpo_tensors[-1].shape[1])
        return np.array(bond_dims, dtype=int)

    def compute_caps(self):
        caps = [np.array([1.0], dtype=complex)]
        last_cap = caps[-1]
        for tensor in reversed(self.mpo_tensors):
            if tensor.ndim == 3:
                new_cap = np.tensordot(tensor, last_cap, axes=([1], [0]))
                new_cap = np.tensordot(new_cap, self.trace_square, axes=([1], [0]))
            else:
                new_cap = np.tensordot(tensor, last_cap, axes=([1], [0]))
                new_cap = np.tensordot(new_cap, self.trace, axes=([1], [0]))
                new_cap = np.tensordot(new_cap, self.trace, axes=([1], [0]))
            caps.insert(0, new_cap)
            last_cap = new_cap
        self.cap_tensors = caps
        return caps

    def as_mpo(self):
        """Return the stored PT-MPO tensors as an ``MPO`` object.

        Rank-3 process tensors use an implicit input-output delta. They are
        expanded before converting to the local MPO axis convention
        ``(left, output, input, right)``.
        """
        data = []
        for tensor in self.mpo_tensors:
            full = self.get_mpo_tensor(len(data), full=True)
            data.append(full.transpose(0, 3, 2, 1))
        return MPO(data)


class PtTempoEngine:
    """Process-tensor TEMPO calculation for a diagonal bath coupling."""

    def __init__(
        self,
        bath: Bath,
        params: TempoParams,
        end_time: float,
        *,
        start_time: float = 0.0,
        unique: bool = False,
        apply_mpo_method: ApplyMPOMethod | None = None,
    ) -> None:
        if unique:
            raise NotImplementedError("unique=True requires bath degeneracy maps")
        if end_time <= start_time:
            raise ValueError("end_time should be larger than start_time")

        self.bath = bath
        self.params = params
        self.dim = bath.dim
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.num_steps = int(round((self.end_time - self.start_time) / self.params.dt))
        if self.num_steps < 2:
            raise ValueError("end_time must span at least two time steps")

        self.dkmax = min(self.params.dkmax, self.num_steps)
        self.num_infl = min(self.num_steps, self.dkmax + 1)
        self.apply_mpo_method = (
            self.params.apply_mpo_method
            if apply_mpo_method is None
            else apply_mpo_method
        )
        self.process_tensor = ProcessTensor(self.dim, self.params.dt)
        self.step = None
        self.truncation_errors = []
        self.spectral_density = bath.spectral_density

    def initialize(self):
        scale = self.dim
        influences_mps = []
        influences_mpo = []

        for i in range(self.num_infl):
            infl = self._influence(i)
            if i == 0:
                infl = infl / scale
                infl_mps = infl.T / scale
                infl_mpo = _create_delta(infl, [1, 1, 0])
            elif i == self.num_infl - 1:
                infl_mps = np.expand_dims(infl, 2)
                infl_mpo = np.expand_dims(infl, 1)
                infl_mpo = np.expand_dims(infl_mpo, 3)
            else:
                infl_mps = _create_delta(infl / scale, [0, 1, 0])
                infl_mpo = _create_delta(infl, [0, 1, 1, 0])

            influences_mps.append(_as_mps_tensor(infl_mps, i, self.num_infl))
            influences_mpo.append(_as_mpo_tensor(infl_mpo, i, self.num_infl))

        self.mps = MPS(influences_mps)
        self.mpo = MPO(influences_mpo)
        self._compress_mps_()
        self.one_mps_tensor = np.array([[[1.0]]], dtype=complex)
        self.step = 1
        return self

    def _influence(self, dk: int) -> ndarray:
        dt = self.params.dt

        if dk == 0:
            t1 = 0.0
            t2 = None
            shape = "upper-triangle"
        elif dk < 0:
            t1 = float(self.dkmax) * dt
            t2 = float(self.dkmax + dk) * dt
            if t2 <= 0:
                return None
            shape = "rectangle"
        else:
            t1 = float(dk) * dt
            t2 = None
            shape = "square"

        eta_dk = self.spectral_density.correlation_2d_integral(
            delta=dt,
            t1=t1,
            t2=t2,
            shape=shape,
            epsrel=self.params.epsrel,
        )
        op_p = self.bath.coupling_acomm
        op_m = self.bath.coupling_comm

        if dk == 0:
            return np.diag(np.exp(-op_m * (eta_dk.real * op_m
                                           + 1j * eta_dk.imag * op_p)))
        return np.exp(-np.outer(eta_dk.real * op_m
                                + 1j * eta_dk.imag * op_p, op_m))

    def compute_step(self):
        if self.step is None:
            self.initialize()

        self.step += 1
        end_phase = self.step > self.num_steps - self.num_infl + 1

        if end_phase:
            self._shorten_mpo_()
            self._contract_mpo_right_(np.ones(self.dim ** 2) * self.dim)
        else:
            if self.dkmax is not None:
                infl = self._influence(-self.step)
                if infl is not None:
                    infl_mpo = np.expand_dims(infl, 1)
                    infl_mpo = np.expand_dims(infl_mpo, 3)
                    self._shorten_mpo_()
                    self._append_mpo_(_as_mpo_tensor(infl_mpo, 0, 1))
            append_mps_(self.mps, self.one_mps_tensor.copy())

        trunc_err = self._apply_influence_mpo_()
        self.truncation_errors.append(trunc_err)
        return self.step < self.num_steps

    def compute(self, *, progressbar: bool = True):
        if self.step is None:
            self.initialize()

        iterator = range(self.step, self.num_steps)
        if progressbar:
            from tqdm import tqdm
            iterator = tqdm(iterator, ascii=True)

        for _ in iterator:
            self.compute_step()
        return self

    def get_process_tensor(self, *, progressbar: bool = True):
        if self.step is None or self.step < self.num_steps:
            self.compute(progressbar=progressbar)
        if len(self.process_tensor) < self.num_steps:
            self.update_process_tensor()
        return self.process_tensor

    @property
    def truncation_error(self):
        err = TruncationError(0.0, 1.0)
        for trunc_err in self.truncation_errors:
            err += trunc_err
        return err

    def update_process_tensor(self):
        if self.step < self.num_steps:
            raise RuntimeError("PT-TEMPO calculation has not finished")
        for step in reversed(range(self.num_steps)):
            self.process_tensor.set_mpo_tensor(step, self.get_mpo_tensor(step))
        self.process_tensor.compute_caps()
        return self.process_tensor

    def get_mpo_tensor(self, step: int):
        if self.mps.L != self.num_steps:
            raise RuntimeError("PT-MPO tensors are available only after compute()")
        if step == 0:
            tensor = self.mps.data[0][0, :, :]
            tensor = np.expand_dims(tensor.T, 0)
        elif step == self.num_steps - 1:
            tensor = self.mps.data[-1][:, :, 0]
            tensor = np.expand_dims(tensor, 1)
        else:
            tensor = self.mps.data[step].transpose(0, 2, 1)
        return tensor * self.dim

    def _trunc_para(self):
        return (
            self.params.chi_max,
            self.params.svd_min,
            self.params.trunc_cut,
        )

    def _apply_influence_mpo_(self):
        if self.mps.L < self.mpo.L:
            raise RuntimeError(f"MPS/MPO length mismatch: {self.mps.L} < {self.mpo.L}")

        start = self.mps.L - self.mpo.L
        if start == 0:
            target = self.mps
        else:
            target = _right_partition_mps(self.mps, start)

        trunc_err = self._apply_mpo_to_mps_(target, self.mpo)

        if start != 0:
            _put_back_right_partition_(self.mps, target, start)
        return trunc_err

    def _apply_mpo_to_mps_(self, mps, mpo):
        if mps.L != mpo.L:
            raise RuntimeError(f"MPS/MPO length mismatch: {mps.L} != {mpo.L}")

        trunc_para = self._trunc_para()
        nm = self.params.normalize
        if self.apply_mpo_method == "naive":
            mps.apply_mpo_naive_(mpo)
            if mps.data[0].shape[0] == 1 and mps.data[-1].shape[-1] == 1:
                trunc_err = mps.canonicalize_(
                    trunc_para=trunc_para,
                    canonicalform=False,
                    qrnormalize=nm,
                )
            else:
                mps.data, mps.Ss, trunc_err = top._right2left_SVD(
                    mps.data,
                    mps.L,
                    trunc_para=trunc_para,
                )
                mps.llim = mps.rlim = 0
        elif self.apply_mpo_method == "density_matrix":
            if mps.data[0].shape[0] == 1 and mps.data[-1].shape[-1] == 1:
                trunc_err = mps.apply_mpo_(
                    mpo,
                    trunc_para=trunc_para,
                    normalize=nm,
                )
            else:
                trunc_err = mps.apply_mpo_zip_up(
                    mpo,
                    trunc_para=trunc_para,
                    direction="left",
                    normalize=nm,
                )
                mps.data, mps.Ss, sweep_trunc_err = top._right2left_SVD(
                    mps.data,
                    mps.L,
                    trunc_para=trunc_para,
                )
                trunc_err += sweep_trunc_err
                mps.llim = mps.rlim = 0
        elif self.apply_mpo_method == "zip_up":
            trunc_err = mps.apply_mpo_zip_up(
                mpo,
                trunc_para=trunc_para,
                direction="left",
                normalize=nm,
            )
            mps.data, mps.Ss, sweep_trunc_err = top._right2left_SVD(
                mps.data,
                mps.L,
                trunc_para=trunc_para,
            )
            trunc_err += sweep_trunc_err
            mps.Ss[0] = mps.Ss[-1] = np.array([1.], dtype=mps.dtype)
            mps.llim = mps.rlim = 0
        else:
            raise ValueError(f"apply_mpo_method {self.apply_mpo_method!r} is not supported")
        return _as_truncation_error(trunc_err)

    def _compress_mps_(self):
        trunc_err = self.mps.canonicalize_(
            trunc_para=self._trunc_para(),
            canonicalform=False,
            qrnormalize=self.params.normalize,
        )
        return _as_truncation_error(trunc_err)

    def _shorten_mpo_(self):
        self.mpo.data.pop()
        self.mpo.Ss.pop()
        self.mpo.L = len(self.mpo.data)
        self.mpo.rlim = min(self.mpo.rlim, self.mpo.L - 1)

    def _append_mpo_(self, tensor):
        if self.mpo.data[-1].shape[-1] != tensor.shape[0]:
            raise ValueError(
                "bond dimension mismatch: "
                f"mpo right bond {self.mpo.data[-1].shape[-1]} != tensor left bond {tensor.shape[0]}"
            )
        self.mpo.data.append(tensor)
        self.mpo.Ss.append(None)
        self.mpo.L = len(self.mpo.data)
        self.mpo.rlim = self.mpo.L - 1

    def _contract_mpo_right_(self, vector):
        tensor = self.mpo.data[-1]
        if tensor.shape[-1] != len(vector):
            raise ValueError(
                "right boundary mismatch: "
                f"mpo right bond {tensor.shape[-1]} != vector length {len(vector)}"
            )
        self.mpo.data[-1] = np.tensordot(tensor, vector, axes=([-1], [0]))[..., None]


def pt_tempo_compute(
    bath: Bath,
    start_time: float,
    end_time: float,
    params: TempoParams,
    *,
    unique: bool = False,
    progressbar: bool = True,
) -> ProcessTensor:
    """Compute and return a PT-TEMPO process tensor."""
    engine = PtTempoEngine(
        bath,
        params,
        end_time,
        start_time=start_time,
        unique=unique,
    )
    return engine.get_process_tensor(progressbar=progressbar)


def _create_delta(tensor, index_scrambling):
    tensor = np.asarray(tensor)
    ret_shape = tuple(tensor.shape[i] for i in index_scrambling)
    ret = np.zeros(ret_shape, dtype=tensor.dtype)

    for index in np.ndindex(tensor.shape):
        ret_index = tuple(index[i] for i in index_scrambling)
        ret[ret_index] = tensor[index]
    return ret


def _as_mps_tensor(tensor, i: int, n: int):
    tensor = np.asarray(tensor, dtype=complex)
    if tensor.ndim == 2:
        if i == 0:
            return tensor.reshape(1, *tensor.shape)
        if i == n - 1:
            return tensor.reshape(*tensor.shape, 1)
    if tensor.ndim != 3:
        raise ValueError(f"MPS tensor should have ndim 2 or 3, got {tensor.shape}")
    return tensor


def _as_mpo_tensor(tensor, i: int, n: int):
    tensor = np.asarray(tensor, dtype=complex)
    if tensor.ndim == 3:
        if i != 0:
            raise ValueError("rank-3 PT-MPO tensor is only valid at the left boundary")
        tensor = tensor.reshape(1, *tensor.shape)
    if tensor.ndim != 4:
        raise ValueError(f"MPO tensor should have ndim 3 or 4, got {tensor.shape}")
    return tensor.transpose(0, 2, 1, 3)


def _as_truncation_error(trunc_err):
    if trunc_err is None:
        return TruncationError(0.0, 1.0)
    return trunc_err


def _right_partition_mps(mps, start: int):
    data = [tensor.copy() for tensor in mps.data[start:]]
    ss = [
        s.copy() if isinstance(s, np.ndarray) else None
        for s in mps.Ss[start:start + len(data) + 1]
    ]
    return MPS(data, Ss=ss, llim=0, rlim=len(data) - 1)


def _put_back_right_partition_(mps, submps, start: int):
    for i, tensor in enumerate(submps.data):
        mps.data[start + i] = tensor
    for i, s in enumerate(submps.Ss):
        mps.Ss[start + i] = s
    mps.llim = start + submps.llim
    mps.rlim = start + submps.rlim
    mps.lognm += submps.lognm
    return mps


__all__ = ["ProcessTensor", "PtTempoEngine", "pt_tempo_compute"]
