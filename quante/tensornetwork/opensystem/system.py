# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator


def as_matrix(obj, *, basis=None, pauli=False, sparse=False):
    """Convert a supported object into a matrix-like object.

    Matrix-like inputs are returned unchanged. Objects with a ``to_matrix``
    method, such as ``SpinOper``, are converted using the provided basis.
    """
    if obj is None:
        return None
    if isinstance(obj, (np.ndarray, sps.spmatrix, sps.sparray, LinearOperator)):
        return obj
    if hasattr(obj, "to_matrix"):
        if basis is None:
            raise ValueError("basis is required when converting operator objects")
        return obj.to_matrix(basis=basis, pauli=pauli, sparse=sparse)
    return np.asarray(obj)


def _to_dense(mat):
    """Return a dense NumPy array for dense, sparse, or linear-operator input."""
    if isinstance(mat, LinearOperator):
        return mat.to_matrix()
    if sps.issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def liouvillian_from_ham(ham, jump_ops=None):
    """Build the Liouville-space generator from Hilbert-space operators.

    The vectorization convention matches NumPy's default reshape order. For a
    density matrix ``rho``, the returned matrix ``L`` satisfies
    ``(L @ rho.reshape(-1)).reshape(d, d) == d rho / dt``.
    """
    ham = as_matrix(ham)
    sparse = sps.issparse(ham) or any(sps.issparse(lo) for lo in (jump_ops or []))
    xp = sps if sparse else np
    eye = xp.eye(ham.shape[0], format="csr") if sparse else np.eye(ham.shape[0])

    res = -1j * (xp.kron(ham, eye) - xp.kron(eye, ham.T))
    for lo in jump_ops or []:
        lo = as_matrix(lo)
        lodlo = lo.conj().T @ lo
        res = res + xp.kron(lo, lo.conj())
        res = res - 0.5 * xp.kron(lodlo, eye)
        res = res - 0.5 * xp.kron(eye, lodlo.T)
    return res


class System:
    """Thin wrapper for system Liouvillians and propagators."""

    def __init__(
        self,
        ham=None,
        *,
        jump_ops=None,
        lindbladian=None,
        basis=None,
        pauli=False,
        sparse=False,
    ):
        """Store a Hamiltonian or prebuilt Liouvillian.

        Parameters are intentionally close to the existing operator APIs in
        ``quante.generate.operas``. This class only normalizes inputs and
        provides propagators needed by TEMPO.
        """
        if (ham is None) == (lindbladian is None):
            raise ValueError("exactly one of ham or lindbladian should be supplied")
        if lindbladian is not None and jump_ops is not None:
            raise ValueError("jump_ops can only be used with ham")

        self.basis = basis
        self.pauli = pauli
        self.ham = as_matrix(ham, basis=basis, pauli=pauli, sparse=sparse)
        self.jump_ops = [
            as_matrix(lo, basis=basis, pauli=pauli, sparse=sparse)
            for lo in (jump_ops or [])
        ]
        self.lindbladian = self._init_lindbladian(
            lindbladian, basis=basis, pauli=pauli, sparse=sparse
        )

        if self.ham is not None:
            self.dim = self.ham.shape[0]
        elif isinstance(self.lindbladian, LinearOperator):
            self.dim = int(round(self.lindbladian.shape[0] ** 0.5))
        else:
            self.dim = int(round(self.lindbladian.shape[0] ** 0.5))

    def _init_lindbladian(self, lindbladian, *, basis=None, pauli=False, sparse=False):
        """Normalize a prebuilt Lindbladian input to matrix or LinearOperator form."""
        if lindbladian is None:
            return None
        if isinstance(lindbladian, (np.ndarray, sps.spmatrix, sps.sparray, LinearOperator)):
            return lindbladian
        if hasattr(lindbladian, "to_linearoperator"):
            return lindbladian.to_linearoperator(basis, pauli)
        if hasattr(lindbladian, "to_matrix"):
            if basis is None:
                raise ValueError("basis is required when converting Lindblian objects")
            return lindbladian.to_matrix(basis=basis, pauli=pauli, sparse=sparse)
        return np.asarray(lindbladian)

    def liouvillian(self):
        """Return and cache the Liouville-space generator."""
        if self.lindbladian is not None:
            return self.lindbladian
        self.lindbladian = liouvillian_from_ham(self.ham, self.jump_ops)
        return self.lindbladian

    def propagator(self, dt, *, dense=True):
        """Return the one-step Liouville-space propagator ``exp(L * dt)``."""
        L = self.liouvillian()
        if not dense:
            if sps.issparse(L):
                return sps.linalg.expm(L * dt)
            if isinstance(L, LinearOperator):
                raise ValueError("dense=False is not supported for LinearOperator")
        return scipy.linalg.expm(_to_dense(L) * dt)

    def half_propagator(self, dt):
        """Return ``exp(L * dt / 2)`` for split-step TEMPO updates."""
        return self.propagator(dt / 2)
