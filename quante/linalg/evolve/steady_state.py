# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-30 18:25:58
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-30 18:46:52


from typing import Literal, TYPE_CHECKING

import numpy as np
from scipy import sparse as sps
from scipy.sparse.linalg import LinearOperator, eigsh, spsolve, svds

if TYPE_CHECKING:
    from ...generate.dynamics import LiouvillianDynamics

SteadyStateMethod = Literal["direct", "eig", "svd"]

__all__ = ["steady_state"]


def steady_state(
    generator,
    method: SteadyStateMethod = "svd",
):
    """Return the normalized steady state of a Liouvillian-like generator.

    Parameters
    ----------
    generator : sparse matrix | LinearOperator | LiouvillianDynamics
        Density-space generator.
    method : {"direct", "eig", "svd"}, optional
        Numerical method used to find the null vector.
    """
    if method not in ("direct", "eig", "svd"):
        raise ValueError("method should be 'direct' or 'eig' or 'svd'")
    operator = _as_density_generator(generator, explicit=True)
    dim = _physical_dim(operator, generator)
    if method == "direct":
        return _steady_state_direct(_as_sparse_matrix(operator), dim)
    if method == "eig":
        return _steady_state_eig(operator, dim)
    return _steady_state_svd(_as_sparse_matrix(operator), dim)


def _as_density_generator(generator, *, explicit: bool):
    from ...generate.dynamics import LiouvillianDynamics

    if isinstance(generator, LiouvillianDynamics):
        if explicit:
            return generator.explicit()
        return generator
    return generator


def _physical_dim(operator, original):
    dim = getattr(original, "dim", None)
    if dim is not None:
        return dim
    size = operator.shape[0]
    dim = int(np.sqrt(size))
    if size != dim * dim:
        raise ValueError("generator shape is not a square of a physical dimension")
    return dim


def _as_sparse_matrix(operator):
    if isinstance(operator, LinearOperator):
        raise TypeError("This steady_state method needs an explicit sparse matrix")
    return operator.tocsr() if sps.issparse(operator) else sps.csr_array(operator)


def _steady_state_direct(L_mat, dim: int):
    size = L_mat.shape[0]
    weight = _constraint_weight(L_mat)

    eye_row = sps.lil_array((size, size), dtype=np.complex128)
    eye_row[0, :] = (sps.eye(dim, format="lil") * weight).reshape(1, -1)
    L_mat_aug = L_mat + eye_row.tocsr()

    rhs = np.zeros((size, 1), dtype=np.complex128)
    rhs[0, 0] = weight
    rho = spsolve(L_mat_aug, rhs).reshape(dim, dim)
    return rho / np.trace(rho)


def _steady_state_eig(operator, dim: int):
    size = operator.shape[0]

    def LdagL_matvec(x):
        return _rmatvec(operator, _matvec(operator, x))

    linop = LinearOperator((size, size), matvec=LdagL_matvec, dtype=np.complex128)
    _, vec = eigsh(linop, k=1, which="SM")
    rho = vec[:, 0].reshape(dim, dim)
    return rho / np.trace(rho)


def _matvec(operator, x):
    if isinstance(operator, LinearOperator):
        return operator.matvec(x)
    return operator @ x


def _rmatvec(operator, x):
    if isinstance(operator, LinearOperator):
        return operator.rmatvec(x)
    return operator.conj().T @ x


def _steady_state_svd(L_mat, dim: int):
    _, _, vh = svds(L_mat, k=1, which="SM")
    rho = vh[0].reshape(dim, dim)
    return rho / np.trace(rho)


def _constraint_weight(L_mat):
    data = getattr(L_mat, "data", None)
    if data is None or len(data) == 0:
        return 1.0
    weight = np.mean(np.abs(data))
    return 1.0 if weight == 0 else weight
