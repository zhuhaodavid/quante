# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-29 16:02:00

"""Base classes and shared helpers for dynamics objects."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from numpy import ndarray
from scipy import sparse as sps
from scipy.sparse import csr_array, csr_matrix, dia_array, dia_matrix
from scipy.sparse.linalg import LinearOperator

DynamicsSpace = Literal["state", "density"]
MatrixRole = Literal["hamiltonian", "generator"]
MatrixLike = ndarray | sps.sparray | sps.spmatrix | LinearOperator
MaybeTimedMatrix = MatrixLike | Callable[[float], MatrixLike]
_TRACE_UNSET = object()


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
            from ...linalg.matops.sparse_mul import dot_parallel

            return dot_parallel(operator, value)
        except (ModuleNotFoundError, ImportError):
            return operator @ value
    return operator @ value


def _compose(left, right):
    if isinstance(left, LinearOperator) or isinstance(right, LinearOperator):
        return sps.linalg.aslinearoperator(left) @ sps.linalg.aslinearoperator(right)
    return left @ right


def _is_zero_sparse(operator):
    return sps.issparse(operator) and operator.nnz == 0


def _any_sparse(operators):
    return any(sps.issparse(op) for op in operators if op is not None and not isinstance(op, LinearOperator))
