# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-27 23:04:05
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-29 12:40:39

from dataclasses import dataclass
import math
import warnings
from typing import Callable, Literal

from numpy import ndarray
import numpy as np

from ...linalg.matops import super as opr
from .system import System
from .bath import Bath
from ..networks import MPS, MPO
from ..core import tensor_operations as top
from ..core.tensor_utils import TruncationError

ApplyMPOMethod = Literal["naive", "density_matrix", "zip_up"]

def create_delta(tensor, index_scrambling):
    tensor = np.asarray(tensor)

    ret_shape = tuple(tensor.shape[i] for i in index_scrambling)
    ret = np.zeros(ret_shape, dtype=tensor.dtype)

    idx = np.indices(tensor.shape)
    ret_idx = tuple(idx[i].ravel() for i in index_scrambling)

    ret[ret_idx] = tensor.ravel()
    return ret

@dataclass
class TempoParams:
    """Numerical parameters for TEMPO-style memory evolution."""

    dt: float
    tcut: float
    epsrel: float = 1e-6
    chi_max: int | None = None
    svd_min: float | None = None
    trunc_cut: float | None = None
    apply_mpo_method: ApplyMPOMethod = "naive"
    normalize: bool = True

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.tcut < 0:
            raise ValueError(f"tcut must be non-negative, got {self.tcut}")
        if self.epsrel <= 0:
            raise ValueError(f"epsrel must be positive, got {self.epsrel}")
        if self.chi_max is not None and self.chi_max <= 0:
            raise ValueError(f"chi_max must be positive or None, got {self.chi_max}")
        if self.svd_min is not None and self.svd_min < 0:
            raise ValueError(f"svd_min must be non-negative or None, got {self.svd_min}")
        if self.trunc_cut is not None and self.trunc_cut < 0:
            raise ValueError(f"trunc_cut must be non-negative or None, got {self.trunc_cut}")
        if self.apply_mpo_method not in ("naive", "density_matrix", "zip_up"):
            raise ValueError(f"apply_mpo_method {self.apply_mpo_method!r} is not supported")

    @property
    def dkmax(self) -> int:  # noqa: A003
        """Return the number of memory cells kept by ``tcut`` and ``dt``."""
        return int(math.ceil(self.tcut / self.dt))


class TempoEngine:
    """Class representing the entire TEMPO tensornetwork as introduced in
    [Strathearn2018].
    """
    def __init__(
            self,
            system: System,
            bath: Bath,
            params: TempoParams,
            initial_state: ndarray,
            times: float,
        ) -> None:
        self.bath = bath
        self.system = system
        self.dim = system.dim
        self.initial_state = initial_state
        self.params = params
        self.times = times
        self.dts = np.insert(np.diff(times), 0, times[0])
        self.cur_step = 0
        self.cur_layer = 0
        self.spectral_density = bath.spectral_density
        self.truncation_errors = []
        self.last_truncation_error = TruncationError(0.0, 1.0)
    
        self.initialize()
    
    def initialize(self):
        """Initialize the TEMPO tensornetwork."""
        if self.bath.unitary is None:
            self.super_u = self.super_u_dag = None
        else:
            self.super_u = opr.left_right_super(
                self.bath.unitary,
                self.bath.unitary.conj().T
            )
            self.super_u_dag = opr.left_right_super(
                self.bath.unitary.conj().T,
                self.bath.unitary
            )
        
        influences = []
        for i in range(self.params.dkmax+1):
            infl = self._influence(i)
            if i == 0:
                infl_four_legs = create_delta(infl, [1, 0, 0, 1])
                if self.super_u is not None:
                    tmp = np.dot(np.moveaxis(infl_four_legs, 1, -1),
                            self.super_u_dag)
                    tmp = np.moveaxis(tmp, -1, 1)
                    tmp = np.dot(tmp, self.super_u.T)
                    infl_four_legs = tmp
            else:
                infl_four_legs = create_delta(infl, [1, 0, 0, 1])

            influences.append(infl_four_legs)
        
        self.mps = MPS([self.initial_state.reshape(1, -1, 1)])
        self.mpo = MPO(list(reversed(influences)))

        
    def _influence(self, dk: int) -> ndarray:
        """Compute the influence of the bath on the system at time ``dkmax*dt``."""
        dt = self.params.dt
        dkmax = self.params.dkmax
        
        if dk == 0:
            t1 = 0
            t2 = None
            shape = "upper-triangle"
        elif dk < 0:
            t1 = float(dkmax) * dt
            return None
        else:
            t1 = float(dk) * dt
            t2 = None
            shape = "square"
        
        eta_dk = self.spectral_density.correlation_2d_integral(
            delta=dt,
            t1=t1,
            t2=t2,
            shape=shape,
            epsrel=self.params.epsrel
        )
        op_p = self.bath.coupling_acomm
        op_m = self.bath.coupling_comm

        if dk == 0:
            infl = np.diag(np.exp(-op_m*(eta_dk.real*op_m \
                                        + 1j*eta_dk.imag*op_p)))
        else:
            infl = np.exp(-np.outer(eta_dk.real*op_m \
                                + 1j*eta_dk.imag*op_p, op_m))
        return infl
    
    def run(self):
        """Advance the TEMPO network by one time step."""
        if self.cur_step >= len(self.times):
            warnings.warn(
                f"t {self.times[-1]} has been reached, dt = {self.dts[-1]} will be used"
            )
            dt = self.dts[-1]
        else:
            dt = self.dts[self.cur_step]
        
        trunc_err = TruncationError(0.0, 1.0)
        if not np.isclose(dt, 0.0): 
            trunc_err = self.update_mps(dt)
            self.cur_layer += 1
        self.last_truncation_error = trunc_err
        self.truncation_errors.append(trunc_err)
        self.cur_step += 1

    @property
    def truncation_error(self):
        """Return the accumulated TEMPO truncation error."""
        err = TruncationError(0.0, 1.0)
        for trunc_err in self.truncation_errors:
            err += trunc_err
        return err

    def update_mps(self, dt: float):
        """Update the MPS by applying the TEMPO evolution.
        
                |      |      |      |      |
                ◻——————◻——————◻——————◻——————◻
                |      |      |      |   
                |      |      |      ◻   
                |      |      |      |   
                ◻——————◻——————◻——————◻
        """
        prop_1, prop_2 = self.system.get_propagators(dt)
        mpo = self._current_mpo()

        self._apply_first_propagator_(prop_1)
        self._align_mps_to_mpo_(mpo)

        trunc_err = self._apply_influence_mpo_(mpo)
        self._append_second_propagator_(prop_2)
        return trunc_err

    def _current_mpo(self):
        if self.cur_layer <= self.params.dkmax:
            mpo = MPO(self.mpo.data[-self.cur_layer-1:])
        else:
            mpo = self.mpo.copy()
            infl = self._influence(self.params.dkmax - self.cur_layer)
            if infl is not None:
                infl_tensor = create_delta(infl, [1, 0, 0, 1])
                mpo.data.append(infl_tensor)
                mpo.data.pop(0)
        sum_west_(mpo)
        return mpo

    def _apply_first_propagator_(self, prop_1):
        self.mps.apply_gate_(self.mps.L-1, gate=prop_1, gate_range=1)

    def _align_mps_to_mpo_(self, mpo):
        if self.mps.L > mpo.L:
            sum_north_(self.mps, self.mps.L - mpo.L)
        assert mpo.L == self.mps.L

    def _trunc_para(self):
        return (
            self.params.chi_max,
            self.params.svd_min,
            self.params.trunc_cut,
        )

    def _apply_influence_mpo_(self, mpo):
        trunc_para = self._trunc_para()
        nm = self.params.normalize
        right_boundary_shape = _fold_right_boundary_into_north_(mpo)
        if self.params.apply_mpo_method == 'naive':
            self.mps.apply_mpo_naive_(mpo)
            trunc_err = self.mps.canonicalize_(
                trunc_para=trunc_para,
                canonicalform=False,
                qrnormalize=nm,
            )
        elif self.params.apply_mpo_method == 'density_matrix':
            trunc_err = self.mps.apply_mpo_(mpo, trunc_para=trunc_para, normalize=nm)
        elif self.params.apply_mpo_method == 'zip_up':
            trunc_err = self.mps.apply_mpo_zip_up(
                mpo,
                trunc_para=trunc_para,
                direction="right",
                normalize=nm,
            )
            self.mps.data, self.mps.Ss, sweep_trunc_err = top._right2left_SVD(
                self.mps.data,
                self.mps.L,
                trunc_para=trunc_para,
            )
            trunc_err += sweep_trunc_err
            self.mps.Ss[0] = self.mps.Ss[-1] = np.array([1.], dtype=self.mps.dtype)
            self.mps.llim = self.mps.rlim = 0
        else:
            raise ValueError(f"apply_mpo_method {self.params.apply_mpo_method!r} is not supported")
    
        _unfold_right_boundary_from_north_(self.mps, right_boundary_shape)
        return _as_truncation_error(trunc_err)

    def _append_second_propagator_(self, prop_2):
        a, b = prop_2.shape
        append_mps_(self.mps, prop_2.T.reshape(b,a,1))

    def measure(
        self,
        obs: ndarray | Callable[[float, ndarray], ndarray] | None = None,
        *,
        progressbar: bool = False,
    ):
        """Run remaining steps and measure each reduced density matrix."""
        values = []
        stop_step = len(self.times)
        iterator = range(self.cur_step, stop_step)
        if progressbar:
            from tqdm import tqdm
            iterator = tqdm(iterator, ascii=True)

        for _ in iterator:
            self.run()
            rho = self.density_matrix()
            t = self.times[self.cur_step - 1]
            if obs is None:
                values.append(rho)
            elif callable(obs):
                values.append(obs(t, rho))
            else:
                values.append(np.trace(obs @ rho))
        return np.real_if_close(values)
    
    def density_matrix(self):
        """Return the density matrix of the system."""
        left = np.sum(self.mps.data[0], axis=1, keepdims=False)
        for i in range(1, self.mps.L-1):
            tmp_tensor = np.sum(self.mps.data[i], axis=1, keepdims=False)
            left = left @ tmp_tensor
        
        right = self.mps.data[-1]
        a, b, c = right.shape
        rho = (left @ right.reshape(a,b)).reshape(self.dim, self.dim)
        rho *= np.exp(self.mps.lognm)
        return rho


def sum_west_(mpo):
    mpo.data[0] = np.sum(mpo.data[0], axis=0, keepdims=True)

def _as_truncation_error(trunc_err):
    if trunc_err is None:
        return TruncationError(0.0, 1.0)
    return trunc_err

def _fold_right_boundary_into_north_(mpo):
    """Fold the right MPO boundary leg into the north/output leg."""
    left, north, south, right = mpo.data[-1].shape
    mpo.data[-1] = (
        mpo.data[-1]
        .swapaxes(2, 3)
        .reshape(left, north * right, south, 1)
    )
    return north, right

def _unfold_right_boundary_from_north_(mps, shape):
    """Undo ``_fold_right_boundary_into_north_`` on the right MPS tensor."""
    north, right = shape
    mps.data[-1] = mps.data[-1].reshape(-1, north, right)
    return mps

def append_mps_(mps, tensor):
    """Append one MPS tensor to the right boundary in place."""
    tensor = np.asarray(tensor)
    if tensor.ndim != 3:
        raise ValueError(f"MPS tensor should have ndim=3, got shape {tensor.shape}")
    if mps.data[-1].shape[-1] != tensor.shape[0]:
        raise ValueError(
            "bond dimension mismatch: "
            f"mps right bond {mps.data[-1].shape[-1]} != tensor left bond {tensor.shape[0]}"
        )

    mps.data.append(tensor)
    mps.Ss.append(None)
    mps.L = len(mps.data)
    if mps.llim is None or mps.rlim is None:
        mps.llim = 0
    mps.rlim = mps.L - 1
    return mps

def append_mpo_(mpo, tensor):
    """Append one MPO tensor to the right boundary in place."""
    tensor = np.asarray(tensor)
    if tensor.ndim != 4:
        raise ValueError(f"MPO tensor should have ndim=4, got shape {tensor.shape}")
    if mpo.data[-1].shape[-1] != tensor.shape[0]:
        raise ValueError(
            "bond dimension mismatch: "
            f"mpo right bond {mpo.data[-1].shape[-1]} != tensor left bond {tensor.shape[0]}"
        )

    mpo.data.append(tensor)
    mpo.Ss.append(None)
    mpo.L = len(mpo.data)
    if mpo.llim is None or mpo.rlim is None:
        mpo.llim = 0
    mpo.rlim = mpo.L - 1
    return mpo

def sum_north_(mps, l):
    """
    l = 2
                ^      ^
                |      |      |      |   
                ◻——————◻——————◻——————◻
    """
    if l == 0:
        return mps
    
    left = np.sum(mps.data[0], axis=1, keepdims=False)
    mps.data.pop(0)
    mps.Ss.pop()
    for i in range(1, l):
        tmp_tensor = np.sum(mps.data[0], axis=1, keepdims=False)
        left = left @ tmp_tensor
        mps.data.pop(0)
        mps.Ss.pop(0)
    a, b = left.shape
    tmp = top._full_contract_two(left.reshape(a,1,b), mps.data[0])
    c, *d, e = mps.data[0].shape
    mps.data[0] = tmp.reshape(a, -1, e)
    mps.L = len(mps.data)
    if mps.is_canonical_form():
        mps.llim = mps.rlim = 0
    else:
        mps.llim = 0
        mps.rlim = max(mps.rlim - l, 0)
    

            



            



        
    



