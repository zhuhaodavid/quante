# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-29 16:02:00

"""Liouvillian dynamics for density matrices."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse as sps
from scipy.sparse.linalg import LinearOperator

from .base import (
    Dynamics,
    MaybeTimedMatrix,
    _TRACE_UNSET,
    _any_sparse,
    _compose,
    _evaluate,
    _format_maybe_timed_operator,
    _infer_dim,
    _is_timed_operator,
    _is_zero_sparse,
    _left_matmul,
    _operator_trace,
    _right_matmul,
)


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
        from ...linalg import matops as opr

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
