# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-29 16:02:00

"""Time-aware linear operators for quantum dynamics."""

from __future__ import annotations

from typing import Callable, Literal, Sequence
import warnings as _warnings

import numpy as np
from numpy import ndarray
from scipy import sparse as sps
from scipy.sparse import csr_array, csr_matrix, dia_array, dia_matrix
from scipy.sparse.linalg import LinearOperator

from ...linalg import matops as opr
from ...linalg.matops.sparse_mul import dot_parallel

DynamicsSpace = Literal["state", "density"]
MatrixRole = Literal["hamiltonian", "generator"]
MatrixLike = ndarray | sps.sparray | sps.spmatrix | LinearOperator
MaybeTimedMatrix = MatrixLike | Callable[[float], MatrixLike]
_TRACE_UNSET = object()


def as_dynamics(
    obj: Dynamics | MaybeTimedMatrix,
    *,
    matrix_role: MatrixRole = "hamiltonian",
    dim: int | None = None,
    is_sparse: bool | None = None,
    allow_dynamics: bool = True,
    herm: bool | None = None,
    space: DynamicsSpace = "state",
    traceA=_TRACE_UNSET,
) -> Dynamics:
    """Return ``obj`` as a state-space dynamics object."""
    if isinstance(obj, Dynamics):
        if not allow_dynamics:
            raise TypeError(
                "This entry point expects matrix-like input. Pass Dynamics "
                "objects directly to the concrete evolve engine."
            )
        return obj
    if matrix_role == "generator":
        return GeneratorDynamics(
            obj,
            dim=dim,
            is_sparse=is_sparse,
            herm=False if herm is None else herm,
            space=space,
            traceA=traceA,
        )
    if matrix_role != "hamiltonian":
        raise ValueError(f"Unknown matrix_role: {matrix_role!r}")
    return HamiltonianDynamics(
        obj,
        dim=dim,
        is_sparse=is_sparse,
        herm=herm,
        traceA=traceA,
    )


class Dynamics(LinearOperator):
    """Base class for linear dynamics ``d/dt y = A(t) y``.

    Time-independent dynamics can be used as a normal ``LinearOperator``.
    Time-dependent dynamics require ``matvec_at(t, y)`` or ``matmat_at(t, Y)``.
    """

    dim: int
    time_dependent: bool = False
    herm: bool = False
    scale = 1.0
    operator_source = "generator"
    space: DynamicsSpace = "state"

    def __init__(self, shape, dtype, *, time_dependent: bool = False, traceA=_TRACE_UNSET):
        self.time_dependent = time_dependent
        self._traceA = traceA
        super().__init__(dtype=np.dtype(dtype), shape=shape)

    def _matvec(self, x):
        if self.time_dependent:
            raise ValueError("time-dependent dynamics requires matvec_at(t, x)")
        return self.matvec_at(None, x)

    def _matmat(self, X):
        if self.time_dependent:
            raise ValueError("time-dependent dynamics requires matmat_at(t, X)")
        return self.matmat_at(None, X)

    def matvec_at(self, t: float | None, x):
        raise NotImplementedError

    def matmat_at(self, t: float | None, X):
        return np.column_stack([self.matvec_at(t, X[:, i]) for i in range(X.shape[1])])

    def explicit(self, t: float | None = None):
        raise TypeError(f"{type(self).__name__} has no explicit matrix representation")

    @property
    def traceA(self):
        if self._traceA is not _TRACE_UNSET:
            return self._traceA
        if self.time_dependent:
            self._traceA = None
            return None
        trace = self._trace_from_structure()
        if trace is not _TRACE_UNSET:
            self._traceA = trace
            return self._traceA
        try:
            explicit = self.explicit()
            trace = getattr(explicit, "trace", None)
            self._traceA = None if trace is None else trace() if callable(trace) else trace
            return self._traceA
        except Exception:
            self._traceA = None
            return None

    def _trace_from_structure(self):
        return _TRACE_UNSET


class HamiltonianDynamics(Dynamics):
    """State-space real-time Hamiltonian dynamics ``dpsi/dt = -1j H psi``."""

    scale = -1j
    operator_source = "hamiltonian"
    space = "state"

    def __init__(
        self,
        ham: MaybeTimedMatrix,
        *,
        herm: bool | None = None,
        is_sparse: bool | None = None,
        traceA=_TRACE_UNSET,
        dim: int | None = None,
    ) -> None:
        self.is_sparse = is_sparse
        self.ham = _format_maybe_timed_operator(ham, self.is_sparse)
        self.dim = _infer_dim(ham, dim=dim)
        self._herm = herm
        dtype = np.result_type(np.asarray(self.scale).dtype, _operator_dtype(self.ham) or np.float64)
        super().__init__((self.dim, self.dim), dtype, time_dependent=_is_timed_operator(ham), traceA=traceA)

    @property
    def herm(self) -> bool:
        if self._herm is None:
            _warnings.warn(
                "HamiltonianDynamics.herm is not specified; using herm=True. "
                "Pass herm=False if the Hamiltonian-like operator is not Hermitian.",
                stacklevel=2,
            )
            return True
        return self._herm

    def matvec_at(self, t: float | None, x):
        return self.scale * _matvec(_evaluate(self.ham, t), x)

    def matmat_at(self, t: float | None, X):
        return self.scale * _left_matmul(_evaluate(self.ham, t), X)

    def explicit(self, t: float | None = None):
        ham = _evaluate(self.ham, t)
        if isinstance(ham, LinearOperator):
            raise TypeError("LinearOperator cannot be converted to an explicit generator")
        return self.scale * ham

    def _trace_from_structure(self):
        trace = _operator_trace(self.ham)
        return _TRACE_UNSET if trace is _TRACE_UNSET else self.scale * trace


class GeneratorDynamics(Dynamics):
    """Generic vector-space dynamics defined by a generator ``A``."""

    scale = 1.0
    operator_source = "generator"

    def __init__(
        self,
        generator: MaybeTimedMatrix,
        *,
        dim: int | None = None,
        traceA=_TRACE_UNSET,
        herm: bool = False,
        is_sparse: bool | None = None,
        space: DynamicsSpace = "state",
    ) -> None:
        if space not in ("state", "density"):
            raise ValueError(f"Unknown dynamics space: {space!r}")
        self.space = space
        self.is_sparse = is_sparse
        self.A = _format_maybe_timed_operator(generator, self.is_sparse)
        self.dim = _infer_dim(generator, dim=dim)
        self.herm = herm
        dtype = _operator_dtype(self.A) or np.complex128
        super().__init__((self.dim, self.dim), dtype, time_dependent=_is_timed_operator(generator), traceA=traceA)

    def matvec_at(self, t: float | None, x):
        return _matvec(_evaluate(self.A, t), x)

    def matmat_at(self, t: float | None, X):
        return _left_matmul(_evaluate(self.A, t), X)

    def explicit(self, t: float | None = None):
        matrix = _evaluate(self.A, t)
        if isinstance(matrix, LinearOperator):
            raise TypeError("LinearOperator cannot be converted to an explicit generator")
        return matrix

    def _trace_from_structure(self):
        return _operator_trace(self.A)


class LiouvillianDynamics(Dynamics):
    """Density-space Lindblad/Liouvillian dynamics on vectorized density matrices."""

    scale = 1.0
    herm = False
    operator_source = "liouvillian"
    space = "density"

    def __init__(
        self,
        ham: MaybeTimedMatrix | None = None,
        jump_ops: Sequence[MaybeTimedMatrix] | None = None,
        *,
        traceA=_TRACE_UNSET,
        dim: int | None = None,
        is_sparse: bool | None = None,
    ) -> None:
        self.is_sparse = is_sparse
        self.ham = None if ham is None else _format_maybe_timed_operator(ham, self.is_sparse)
        self.jump_ops = [] if jump_ops is None else [
            _format_maybe_timed_operator(op, self.is_sparse) for op in jump_ops
        ]
        if ham is None and not self.jump_ops:
            raise ValueError("LiouvillianDynamics requires ham or at least one jump operator")
        ref = ham if ham is not None else self.jump_ops[0]
        self.dim = _infer_dim(ref, dim=dim)
        time_dependent = _is_timed_operator(ham) or any(_is_timed_operator(op) for op in self.jump_ops)
        if not time_dependent:
            self.jump_ops = [jump for jump in self.jump_ops if not _is_zero_sparse(jump)]
        self._ham_eff = None
        super().__init__((self.dim * self.dim, self.dim * self.dim), np.complex128, time_dependent=time_dependent, traceA=traceA)

    @property
    def ham_eff(self):
        if self.time_dependent:
            raise ValueError("time-dependent LiouvillianDynamics has no cached ham_eff")
        return self._ham_eff_at(None)

    def _ham_eff_at(self, t: float | None):
        if not self.time_dependent and self._ham_eff is not None:
            return self._ham_eff
        ham = None if self.ham is None else _evaluate(self.ham, t)
        jumps = self._jumps_at(t)
        jump_sum = None
        for jump in jumps:
            jump_dagger = jump.H if isinstance(jump, LinearOperator) else jump.conj().T
            term = _compose(jump_dagger, jump)
            jump_sum = term if jump_sum is None else jump_sum + term
        if ham is None:
            ham_eff = -0.5j * jump_sum if jump_sum is not None else 0
        elif jump_sum is None:
            ham_eff = ham
        else:
            ham_eff = ham - 0.5j * jump_sum
        if not self.time_dependent:
            self._ham_eff = ham_eff
        return ham_eff

    def _jumps_at(self, t: float | None):
        if not self.time_dependent:
            return self.jump_ops
        return [
            jump for jump in (_evaluate(op, t) for op in self.jump_ops)
            if not _is_zero_sparse(jump)
        ]

    def matvec_at(self, t: float | None, x):
        rho = np.asarray(x).reshape(self.dim, self.dim)
        jumps = self._jumps_at(t)
        ham_eff = self._ham_eff_at(t)
        ham_eff_dagger = ham_eff.H if isinstance(ham_eff, LinearOperator) else ham_eff.conj().T
        out = -1j * (_left_matmul(ham_eff, rho) - _right_matmul(rho, ham_eff_dagger))
        for jump in jumps:
            jump_dagger = jump.H if isinstance(jump, LinearOperator) else jump.conj().T
            out += _left_matmul(jump, _right_matmul(rho, jump_dagger))
        return out.reshape(-1)

    def matmat_at(self, t: float | None, X):
        return np.column_stack([self.matvec_at(t, X[:, i]) for i in range(X.shape[1])])

    def _rmatvec(self, x):
        if self.time_dependent:
            raise ValueError("time-dependent dynamics requires an explicit adjoint at a fixed time")
        return self.rmatvec_at(None, x)

    def rmatvec_at(self, t: float | None, x):
        rho = np.asarray(x).reshape(self.dim, self.dim)
        jumps = self._jumps_at(t)
        ham_eff = self._ham_eff_at(t)
        out = -1j * (_left_matmul(ham_eff.T, rho) - _right_matmul(rho, ham_eff.conj()))
        for jump in jumps:
            out += _left_matmul(jump.T, _right_matmul(rho, jump.conj()))
        return out.reshape(-1)

    def explicit(self, t: float | None = None):
        jumps = self._jumps_at(t)
        ham_eff = self._ham_eff_at(t)
        if any(isinstance(op, LinearOperator) for op in [ham_eff, *jumps]):
            raise TypeError("LinearOperator terms cannot be converted to an explicit Liouvillian")
        eye = sps.eye(self.dim) if _any_sparse([ham_eff, *jumps]) else np.eye(self.dim)
        sparse = _any_sparse([ham_eff, *jumps])
        liouvillian = -1j * (
            opr.left_right_super(ham_eff, eye, sparse=sparse)
            - opr.left_right_super(eye, ham_eff.conj().T, sparse=sparse)
        )
        jump_sum = _sum_jump(jumps, self.dim) if sparse else None
        if jump_sum is not None:
            liouvillian = liouvillian + jump_sum
        elif jumps:
            for jump in jumps:
                liouvillian = liouvillian + opr.left_right_super(jump, jump.conj().T, sparse=False)
        return liouvillian

    def liouvillian(self, t: float | None = None):
        return self.explicit(t=t)

    def _trace_from_structure(self):
        if self.time_dependent:
            return _TRACE_UNSET
        try:
            jumps = self._jumps_at(None)
            ham_eff = self._ham_eff_at(None)
            ham_trace = _operator_trace(ham_eff)
            if ham_trace is _TRACE_UNSET:
                return _TRACE_UNSET
            trace = 2 * self.dim * np.imag(ham_trace)
            for jump in jumps:
                jump_trace = _operator_trace(jump)
                if jump_trace is _TRACE_UNSET:
                    return _TRACE_UNSET
                trace += abs(jump_trace) ** 2
            return trace.item() if isinstance(trace, np.ndarray) else trace
        except Exception:
            return _TRACE_UNSET

def _operator_dtype(operator):
    if operator is None or _is_timed_operator(operator):
        return None
    return getattr(operator, "dtype", None)


def _operator_trace(operator):
    if operator is None or _is_timed_operator(operator) or isinstance(operator, LinearOperator):
        return _TRACE_UNSET
    trace = getattr(operator, "trace", None)
    if trace is None:
        return _TRACE_UNSET
    return trace() if callable(trace) else trace


def _infer_dim(operator: MaybeTimedMatrix, *, dim: int | None = None) -> int:
    if _is_timed_operator(operator):
        if dim is None:
            raise ValueError("dim must be supplied when operator is time-dependent")
        return dim
    if operator.shape[0] != operator.shape[1]:
        raise ValueError(f"operator must be square, got shape {operator.shape}")
    operator_dim = operator.shape[0]
    if dim is not None and dim != operator_dim:
        raise ValueError(f"dim={dim} is inconsistent with operator shape {operator.shape}")
    return operator_dim


def _evaluate(operator: MaybeTimedMatrix, t: float | None):
    if _is_timed_operator(operator):
        if t is None:
            raise ValueError("t must be supplied for time-dependent dynamics")
        return operator(t)
    return operator


def _is_timed_operator(operator) -> bool:
    return callable(operator) and not isinstance(operator, LinearOperator)


def _format_maybe_timed_operator(operator: MaybeTimedMatrix, is_sparse: bool | None):
    if _is_timed_operator(operator):
        return lambda t: _format_operator(operator(t), is_sparse)
    return _format_operator(operator, is_sparse)


def _format_operator(operator: MatrixLike, is_sparse: bool | None):
    if isinstance(operator, LinearOperator):
        return operator
    if is_sparse is None:
        return operator if sps.issparse(operator) else np.asarray(operator)
    if is_sparse:
        return operator if sps.issparse(operator) else sps.csr_array(operator)
    return operator.toarray() if sps.issparse(operator) else np.asarray(operator)


def _matvec(operator, vector):
    if isinstance(operator, LinearOperator):
        return operator.matvec(vector)
    if sps.issparse(operator):
        return _sparse_dot(operator, vector)
    return operator @ vector


def _left_matmul(operator, value):
    if isinstance(operator, LinearOperator):
        return operator.matmat(value)
    if sps.issparse(operator):
        return _sparse_dot(operator, value)
    return operator @ value


def _right_matmul(value, operator):
    if isinstance(operator, LinearOperator):
        try:
            return operator.T.matmat(np.asarray(value).T).T
        except Exception as exc:
            raise TypeError(
                "Right multiplication by a LinearOperator requires a working transpose action"
            ) from exc
    if sps.issparse(operator):
        return _sparse_dot(operator.T.tocsr(), np.asarray(value).T).T
    return value @ operator


def _sparse_dot(operator, value):
    if isinstance(operator, (csr_array, csr_matrix, dia_array, dia_matrix)):
        try:
            return dot_parallel(operator, value)
        except (ModuleNotFoundError, ImportError):
            return operator @ value
    return operator @ value


def _compose(left, right):
    if isinstance(left, LinearOperator) or isinstance(right, LinearOperator):
        return sps.linalg.aslinearoperator(left) @ sps.linalg.aslinearoperator(right)
    return left @ right


def _sum_jump(jump_ops, dim: int):
    if not jump_ops:
        return None
    if not all(sps.issparse(jump) for jump in jump_ops):
        return None
    from ..basis.basis_class_nb import coodiaglists2csr

    row_result = []
    col_result = []
    ele_result = []
    for jump in jump_ops:
        tmp = sps.kron(jump, jump.conj()).tocoo()
        row_result.append(tmp.row)
        col_result.append(tmp.col)
        ele_result.append(tmp.data)
    return coodiaglists2csr(
        row_result=row_result,
        col_result=col_result,
        ele_result=ele_result,
        diag=None,
        n_row=dim * dim,
        index_type=np.int32,
        dtype=np.complex128,
    )


def _is_zero_sparse(operator):
    return sps.issparse(operator) and operator.nnz == 0


def _any_sparse(operators):
    return any(sps.issparse(op) for op in operators if op is not None and not isinstance(op, LinearOperator))
