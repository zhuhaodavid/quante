# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-27 22:28:53
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-27 22:31:10

import numpy as np
from numpy import ndarray
from scipy import sparse as sps

from .kron import eye, kron


def commutator(operator: ndarray, *, sparse: bool | None = None):
    """Construct commutator superoperator from operator. """
    sparse = sps.issparse(operator) if sparse is None else sparse
    dim = operator.shape[0]
    identity = eye(dim, sparse=sparse, stype="csr", dtype=operator.dtype)
    stype = "csr" if sparse else None
    return kron(operator, identity, stype=stype) - kron(identity, operator.T, stype=stype)


def acommutator(operator: ndarray, *, sparse: bool | None = None):
    """Construct anti-commutator superoperator from operator. """
    sparse = sps.issparse(operator) if sparse is None else sparse
    dim = operator.shape[0]
    identity = eye(dim, sparse=sparse, stype="csr", dtype=operator.dtype)
    stype = "csr" if sparse else None
    return kron(operator, identity, stype=stype) + kron(identity, operator.T, stype=stype)

def left_right_super(
        left_operator: ndarray,
        right_operator: ndarray,
        *,
        sparse: bool | None = None):
    """Construct left and right acting superoperator from operators. """
    sparse = (
        sps.issparse(left_operator) or sps.issparse(right_operator)
        if sparse is None
        else sparse
    )
    return kron(left_operator, right_operator.T, stype="csr" if sparse else None)


