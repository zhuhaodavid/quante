# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-30 19:55:16
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-31 00:49:08


from scipy import sparse as sps
from scipy.sparse.linalg import LinearOperator
from scipy.integrate import solve_ivp

import numpy as _np
import warnings as _warnings
from typing import TYPE_CHECKING, Callable, Literal, TypeAlias, get_args
from functools import lru_cache
from tqdm import tqdm

from ...measure.expect import expect
from .eigen_evolve import (
    _project_to_eigenbasis,
    _reconstruct_from_eigenbasis,
)

if TYPE_CHECKING:
    from ...generate.dynamics import Dynamics

MatrixLikeInput: TypeAlias = _np.ndarray | sps.sparray | sps.spmatrix | LinearOperator
MatrixRole = Literal["hamiltonian", "generator"]
Observable = (
    None
    | _np.ndarray
    | sps.sparray
    | sps.spmatrix
    | list[sps.sparray]
    | list[sps.spmatrix]
    | list[_np.ndarray]
    | Callable[[float, _np.ndarray], _np.ndarray]
)
ODEMethod = Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
EvolveMethod = Literal[
    "eig-cpu",
    "eig-cuda:0",
    "mul-cpu",
    "mul-cuda:0",
    "RK45",
    "RK23",
    "DOP853",
    "Radau",
    "BDF",
    "LSODA",
]

__all__ = [
    "EvolveEngineBase",
    "ArrayEvolveEngine",
    "ExpmMulEvolveEngine",
    "EigenEvolveEngine",
    "OdeEvolveEngine",
    "evolve_and_measure",
    "make_evolve_engine",
]

_ODE_METHODS = set(get_args(ODEMethod))


def _method_family(method: str) -> Literal["eig", "mul", "ode"]:
    if method.startswith("eig-"):
        return "eig"
    if method.startswith("mul-"):
        return "mul"
    if method in _ODE_METHODS:
        return "ode"
    raise ValueError(f"Unknown evolution method: {method!r}")


def _method_device(method: str) -> str:
    family = _method_family(method)
    if family in ("eig", "mul"):
        return method.split("-", 1)[1]
    return "cpu"


def _trace_sentinel_to_unset(traceA):
    if isinstance(traceA, float) and _np.isnan(traceA):
        from ...generate.dynamics import _TRACE_UNSET

        return _TRACE_UNSET
    return traceA


class EvolveEngineBase:
    """Base class for time-stepped evolution."""

    def __init__(self, ts, *, start_time: float = 0.0):
        self.tlist = _np.asarray(ts)
        self.start_time = start_time
        self.cur_time = start_time
        self.dts = _np.insert(_np.diff(self.tlist), 0, self.tlist[0] - start_time)
        self.cur_step = 0

    @property
    def finished(self):
        return self.cur_step >= len(self.tlist)

    def _next_dt(self):
        if self.finished:
            raise StopIteration(
                f"Evolution has already reached the end of tlist at t={self.cur_time}."
            )
        return self.dts[self.cur_step]

    def step(self):
        dt = self._next_dt()
        if dt != 0:
            self.cur_state = self.propagate(self.cur_state, dt)
            self.cur_time += dt
        self.cur_step += 1
        return self.cur_state

    def propagate(self, state, dt: float):
        raise NotImplementedError
    
    def _run(self, measure_func, progressbar):
        res = []
        tlist = self.tlist[self.cur_step:]
        t_iter = tqdm(tlist, ascii=True) if progressbar else tlist
        for t in t_iter:
            state = self.step()
            res.append(measure_func(t, state))
        try:
            return _np.real_if_close(res)
        except (TypeError, ValueError):
            return res
    
    def _plot(self, measure_func, ax=None, *args, **kwargs):
        from ...basicfun import DynamicPlot
        tlist = self.tlist[self.cur_step:]
        dp = DynamicPlot(tlist, ax, **kwargs)
        for t in tlist:
            state = self.step()
            res = _np.real_if_close(measure_func(t, state))
            if isinstance(res, _np.ndarray) and res.ndim >= 1 and len(res) == 1:
                res = res[0]
            dp.append(res)
        res = dp.data
        return res


class ArrayEvolveEngine(EvolveEngineBase):
    """Evolution base for array/vector states."""

    device = "cpu"
    pkg = _np
    herm = False
    scale = 1.0

    def __init__(
        self,
        dynamics: "Dynamics",
        *,
        ts,
        method: str,
        device: str | None,
        start_time: float = 0.0,
        options: dict | None = None,
    ):
        from ...generate.dynamics import Dynamics

        super().__init__(ts, start_time=start_time)
        if not isinstance(dynamics, Dynamics):
            raise TypeError("ArrayEvolveEngine requires a Dynamics object")
        self.dynamics = dynamics
        self.device = _method_device(method) if device is None else device
        self.method = method
        self.options = {} if options is None else dict(options)
        self.resolved_options = {
            "backend": type(self).__name__,
            "dynamics_type": type(self.dynamics).__name__,
            "time_dependent": self.dynamics.time_dependent,
            "method": self.method,
            "device": self.device,
            "start_time": self.start_time,
        }

    def _check_initial_state_shape(self, init_state):
        state = _np.asarray(init_state)
        expected_size = self.dynamics.shape[0]
        if state.size != expected_size:
            raise ValueError(
                f"init_state has size {state.size}, but dynamics requires size {expected_size}"
            )
        return state

    @staticmethod
    def _infer_state_dtype(state, operator_dtype=None):
        return _np.result_type(state.dtype, operator_dtype or state.dtype)

    @staticmethod
    def _to_cpu_state(state, dtype):
        return state.reshape(-1, 1).astype(dtype)

    @staticmethod
    def _to_gpu_state(state, dtype, *, device: str):
        from ...bridge.torch_utils import totc
        import torch as tc

        torch_dtype = tc.complex128 if _np.issubdtype(dtype, _np.complexfloating) else tc.float64
        return totc(state.reshape(-1, 1), device=device, dtype=torch_dtype)

    def _initialize_state(self, init_state=None):
        if init_state is not None:
            self.initial_state = init_state
        self.psi = self.initial_state
        self.cur_state = self.psi
        self._update_resolved_options(state_dtype=str(getattr(self.cur_state, "dtype", None)))

    def _update_resolved_options(self, **options):
        self.resolved_options.update(options)

    def _prepare_measure(self, measure):
        if measure is None:
            return lambda t, state: state.reshape(-1)
        if isinstance(measure, (sps.sparray, sps.spmatrix, list, _np.ndarray)):
            if self.device != "cpu":
                from ...bridge.torch_utils import totc

                measure = totc(measure, device=self.device)
            dim = getattr(self.dynamics, "dim", None)
            if self.dynamics.space == "density":
                return lambda t, state: expect(measure, state.reshape(dim, dim), isdm=True)
            return lambda t, state: expect(measure, state.reshape(self.dynamics.shape[0], -1), isdm=False)
        if callable(measure):
            return measure
        raise ValueError("measure should be a list of sparse matrices or a function")

    def run(self, measure: Observable = None, *, progressbar: bool = True):
        measure_func = self._prepare_measure(measure)
        return self._run(measure_func, progressbar=progressbar)

    def plot(self, measure: Observable, ax=None, **kwargs):
        measure_func = self._prepare_measure(measure)
        return self._plot(measure_func, ax=ax, **kwargs)


class ExpmMulEvolveEngine(ArrayEvolveEngine):
    def __init__(
        self,
        dynamics: "Dynamics",
        init_state,
        ts,
        *,
        method: str = "mul-cpu",
        device: str | None = None,
        start_time: float = 0.0,
        backend_options: dict | None = None,
    ):
        super().__init__(
            dynamics,
            ts=ts,
            method=method,
            device=device,
            start_time=start_time,
            options=backend_options,
        )
        if _method_family(method) != "mul":
            raise ValueError(f"ExpmMulEvolveEngine requires a mul method, got {method!r}")
        if self.dynamics.time_dependent:
            raise ValueError("time-dependent dynamics require an ODE method")
        matrix = self._prepare_operator(compute_traceA=True)
        if self.device == "cpu":
            self._prepare_cpu(matrix, init_state)
        else:
            self._prepare_gpu(matrix, init_state)
        self._initialize_state()

    def _prepare_operator(self, *, compute_traceA: bool = True):
        try:
            matrix = self.dynamics.explicit()
        except TypeError:
            matrix = self.dynamics
        self.scale = 1.0
        self.traceA = self.dynamics.traceA if compute_traceA else None
        self.herm = self.dynamics.herm
        self.operator_source = self.dynamics.operator_source
        self._update_resolved_options(backend_options={
            "matrix_type": type(matrix).__name__,
            "traceA": self.traceA,
            "herm": self.herm,
            "scale": self.scale,
            "operator_source": self.operator_source,
        })
        return matrix

    def _prepare_cpu(self, matrix, init_state):
        self.matrix = matrix if isinstance(matrix, LinearOperator) else _to_csr(matrix)
        state = self._check_initial_state_shape(init_state)
        dtype = self._infer_state_dtype(state, self.dynamics.dtype)
        self.initial_state = self._to_cpu_state(state, dtype)
        self.pkg = _np

    def _prepare_gpu(self, matrix, init_state):
        if isinstance(matrix, LinearOperator):
            raise TypeError("GPU evolve methods do not support LinearOperator generators")
        from ...bridge.torch_utils import totc
        import torch as tc

        self.matrix = totc(matrix, device=self.device)
        state = self._check_initial_state_shape(init_state)
        dtype = self._infer_state_dtype(state, self.dynamics.dtype)
        self.initial_state = self._to_gpu_state(state, dtype, device=self.device)
        self.pkg = tc

    @lru_cache(maxsize=2)
    def _evolve_engine(self, dt):
        if self.device == "cpu":
            from .nbfuc.expm_mul_core import _evolve_engine

            return _evolve_engine(
                self.matrix,
                scale=self.scale,
                t=dt,
                traceA=self.traceA,
                herm=self.herm,
            )
        from ...bridge.torch_utils.linalg.expm_multiply import evolve_engine

        return evolve_engine(dt * self.matrix, scale=self.scale, herm=self.herm)

    def propagate(self, state, dt: float):
        return self._evolve_engine(round(dt, 14))(state)


class EigenEvolveEngine(ArrayEvolveEngine):
    def __init__(
        self,
        dynamics: "Dynamics",
        init_state,
        ts,
        *,
        method: str = "eig-cpu",
        device: str | None = None,
        start_time: float = 0.0,
        eigval=None,
        eigvec=None,
        backend_options: dict | None = None,
    ):
        backend_options = {} if backend_options is None else dict(backend_options)
        if eigval is None and "eigval" in backend_options:
            eigval = backend_options.pop("eigval")
        if eigvec is None and "eigvec" in backend_options:
            eigvec = backend_options.pop("eigvec")
        super().__init__(
            dynamics,
            ts=ts,
            method=method,
            device=device,
            start_time=start_time,
            options=backend_options,
        )
        if _method_family(method) != "eig":
            raise ValueError(f"EigenEvolveEngine requires an eig method, got {method!r}")
        if self.dynamics.time_dependent:
            raise ValueError("time-dependent dynamics require an ODE method")
        self._prepare_operator()
        self._prepare_eigensystem(eigval=eigval, eigvec=eigvec)
        self._initialize_state(self._prepare_state(init_state))
        self.projected_initial_state = _project_to_eigenbasis(
            self.pkg, self.eigenvectors, self.initial_state, self.herm
        )

    def _prepare_operator(self):
        matrix = self.dynamics.explicit()
        if isinstance(matrix, LinearOperator):
            raise TypeError("eig methods do not support LinearOperator generators")
        self.herm = self.dynamics.herm
        self.operator_source = self.dynamics.operator_source
        if self.device == "cpu":
            self.matrix = _to_csr(matrix)
            self.pkg = _np
        else:
            from ...bridge.torch_utils import totc
            import torch as tc

            self.matrix = totc(matrix, device=self.device)
            self.pkg = tc
        self._update_resolved_options(
            backend_options={
                "matrix_type": type(self.matrix).__name__,
                "herm": self.herm,
                "scale": 1.0,
                "operator_source": self.operator_source,
            }
        )

    def _prepare_state(self, state):
        state = self._check_initial_state_shape(state)
        dtype = self._infer_state_dtype(state, self.dynamics.dtype)
        if self.device == "cpu":
            return self._to_cpu_state(state, dtype)
        return self._to_gpu_state(state, dtype, device=self.device)

    def _prepare_eigensystem(self, *, eigval=None, eigvec=None):
        if (eigval is None) != (eigvec is None):
            raise ValueError("eigval and eigvec should be passed together")
        if eigval is not None:
            self.eigenvalues = self._prepare_eigen_array(eigval)
            self.eigenvectors = self._prepare_eigen_array(eigvec)
            self._update_resolved_options(eigensystem="provided")
            return
        mat = self.matrix.toarray() if self.device == "cpu" else self.matrix.to_dense()
        eigf = self.pkg.linalg.eigh if self.herm else self.pkg.linalg.eig
        self.eigenvalues, self.eigenvectors = eigf(mat)
        self._update_resolved_options(eigensystem="computed")

    def _prepare_eigen_array(self, value):
        if self.device == "cpu":
            return _np.asarray(value)
        from ...bridge.torch_utils import totc

        return totc(value, device=self.device)

    def state_at_time(self, t: float):
        return _reconstruct_from_eigenbasis(
            self.pkg,
            self.eigenvalues,
            self.eigenvectors,
            t - self.start_time,
            self.projected_initial_state,
            1.0,
        )

    def propagate(self, state, dt: float):
        state = self._prepare_state(state)
        return _reconstruct_from_eigenbasis(
            self.pkg,
            self.eigenvalues,
            self.eigenvectors,
            dt,
            _project_to_eigenbasis(self.pkg, self.eigenvectors, state, self.herm),
            1.0,
        )


class OdeEvolveEngine(ArrayEvolveEngine):
    def __init__(
        self,
        dynamics: "Dynamics",
        init_state,
        ts,
        *,
        method: str = "RK45",
        device: str | None = None,
        start_time: float = 0.0,
        ivp_options: dict | None = None,
    ):
        super().__init__(
            dynamics,
            ts=ts,
            method=method,
            device=device,
            start_time=start_time,
            options=None,
        )
        ivp_options = self._prepare_ivp_options(ivp_options)
        if _method_family(method) != "ode":
            raise ValueError(f"OdeEvolveEngine requires an ODE method, got {method!r}")
        if self.device != "cpu":
            raise ValueError("ODE methods currently run on the CPU")
        self.ivp_options = ivp_options
        self.method, self.ivp_method = "ode", method
        self._update_resolved_options(ivp_options={"method": self.ivp_method, **dict(ivp_options)})
        self._initialize_state(self._prepare_state(init_state))

    @staticmethod
    def _prepare_ivp_options(ivp_options: dict | None):
        options = dict(rtol=1e-9, atol=1e-12)
        if ivp_options is not None:
            options.update(ivp_options)
        return options

    def _prepare_state(self, state):
        state = self._check_initial_state_shape(state)
        dtype = self._infer_state_dtype(state, self.dynamics.dtype)
        return self._to_cpu_state(state, dtype)

    def _rhs(self, t, state):
        return self.dynamics.matvec_at(t, state).reshape(-1)

    def propagate(self, state, dt: float):
        start = self.cur_time
        stop = self.cur_time + dt
        sol = solve_ivp(
            self._rhs,
            (start, stop),
            state.flatten(),
            t_eval=[stop],
            method=self.ivp_method,
            **self.ivp_options,
        )
        if not sol.success:
            raise RuntimeError(f"ODE solver failed with message: {sol.message}")
        return sol.y


def _to_csr(matrix):
    if isinstance(matrix, LinearOperator):
        return matrix
    try:
        return matrix.tocsr()
    except AttributeError:
        return sps.csr_array(matrix)


class MeasureError(Exception):
    """Custom exception for measurement errors."""


def evolve_and_measure(
    matrix: MatrixLikeInput,
    inistate: _np.ndarray,
    tlist: _np.ndarray,
    *,
    herm: bool | None = None,
    measure: Observable = None,
    method: EvolveMethod = "mul-cpu",
    matrix_role: MatrixRole = "hamiltonian",
    traceA=_np.nan,
    start_time: float = 0.0,
    backend_options: dict | None = None,
    progressbar: bool = True,
    ivp_options: dict | None = None,
):
    """Evolve a matrix-defined dynamics and evaluate measurements."""
    return make_evolve_engine(
        matrix,
        inistate,
        tlist,
        method=method,
        matrix_role=matrix_role,
        start_time=start_time,
        backend_options=backend_options,
        ivp_options=ivp_options,
        herm=herm,
        traceA=traceA,
    ).run(measure=measure, progressbar=progressbar)


def make_evolve_engine(
    matrix: MatrixLikeInput,
    init_state,
    ts,
    *,
    method: str = "mul-cpu",
    herm: bool | None = None,
    device: str | None = None,
    matrix_role: MatrixRole = "hamiltonian",
    traceA=_np.nan,
    start_time: float = 0.0,
    backend_options: dict | None = None,
    ivp_options: dict | None = None,
):
    from ...generate.dynamics import as_dynamics

    device = _method_device(method) if device is None else device
    family = _method_family(method)
    dynamics = as_dynamics(
        matrix,
        matrix_role=matrix_role,
        is_sparse=None,
        allow_dynamics=False,
        herm=herm,
        traceA=_trace_sentinel_to_unset(traceA),
    )
    if family == "mul":
        return ExpmMulEvolveEngine(
            dynamics,
            init_state,
            ts,
            method=method,
            device=device,
            start_time=start_time,
            backend_options=backend_options,
        )
    if family == "eig":
        return EigenEvolveEngine(
            dynamics,
            init_state,
            ts,
            method=method,
            device=device,
            start_time=start_time,
            backend_options=backend_options,
        )
    if family == "ode":
        return OdeEvolveEngine(
            dynamics,
            init_state,
            ts,
            method=method,
            device=device,
            start_time=start_time,
            ivp_options=ivp_options,
        )
    raise ValueError(f"Unknown evolution method: {method!r}")
