# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-29 16:02:00

"""Hamiltonian and generic generator dynamics."""

from __future__ import annotations

import warnings as _warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator

from .base import (
    Dynamics,
    DynamicsSpace,
    MatrixRole,
    MaybeTimedMatrix,
    _TRACE_UNSET,
    _evaluate,
    _format_maybe_timed_operator,
    _infer_dim,
    _is_timed_operator,
    _left_matmul,
    _matvec,
    _operator_dtype,
    _operator_trace,
)


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
