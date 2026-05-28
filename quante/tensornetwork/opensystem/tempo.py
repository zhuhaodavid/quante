# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-27 23:04:05
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-28 02:44:42

from dataclasses import dataclass
import math
import warnings

from numpy import ndarray
import numpy as np

from ...linalg.matops import super as opr
from .system import System
from .bath import Bath
from ..networks import MPS, MPO
from ..core import tensor_operations as top

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
    apply_mpo_method: str = "naive"
    normalize: bool = True

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
        self.spetral_density = bath.spectral_density
    
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
        for i in range(self.params.dkmax):
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
            shape = "rectangle"
        else:
            t1 = float(dk) * dt
            t2 = None
            shape = "square"
        
        eta_dk = self.spetral_density.correlation_2d_integral(
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
        """Run the TEMPO tensornetwork evolution."""
        if self.cur_step >= len(self.times):
            warnings.warn(
                f"t {self.times[-1]} has been reached, dt = {self.dts[-1]} will be used"
            )
            dt = self.dts[-1]
        else:
            dt = self.dts[self.cur_step]
        
        if not np.isclose(dt, 0.0): 
            self.update_mps(dt)
            self.cur_layer += 1
        self.cur_step += 1
    
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

        self.mps.apply_gate_(self.mps.L-1, gate=prop_1, gate_range=1)

        if self.mps.L > mpo.L:
            sum_north_(self.mps, self.mps.L - mpo.L)
        assert mpo.L == self.mps.L

        trunc_para = (
            self.params.chi_max,
            self.params.svd_min,
            self.params.trunc_cut,
        )
        nm = self.params.normalize
        if self.params.apply_mpo_method == 'naive':
            self.mps.apply_mpo_naive_(mpo)
            a, b = prop_2.shape
            append_mps_(self.mps, prop_2.T.reshape(b,a,1))
            self.mps.canonicalize_(trunc_para=trunc_para, canonicalform=False, qrnormalize=nm)
        elif self.params.apply_mpo_method == 'density_matrix':
            append_mps_(self.mps, np.ones(1).reshape(1,1,1))
            a, b = prop_2.shape
            append_mpo_(mpo, prop_2.T.reshape(b,a,1,1))
            self.mps.apply_mpo_(mpo, trunc_para=trunc_para, normalize=nm)
    
    def get_rho(self):
        """Return the density matrix of the system."""
        left = np.sum(self.mps.data[0], axis=1, keepdims=False)
        for i in range(1, self.mps.L-1):
            tmp_tensor = np.sum(self.mps.data[i], axis=1, keepdims=False)
            left = left @ tmp_tensor
        
        right = self.mps.data[-1]
        a, b, c = right.shape
        rho = left @ right.reshape(a,b).reshape(self.dim, self.dim)
        rho *= np.exp(self.mps.lognm)
        return rho


def sum_west_(mpo):
    mpo.data[0] = np.sum(mpo.data[0], axis=0, keepdims=True)

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
    

            



            



        
    



