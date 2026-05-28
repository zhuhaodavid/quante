# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-27 22:28:53
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-27 22:31:10

import numpy as np
from numpy import ndarray

def commutator(operator: ndarray) -> ndarray:
    """Construct commutator superoperator from operator. """
    dim = operator.shape[0]
    return np.kron(operator, np.identity(dim)) \
            - np.kron(np.identity(dim), operator.T)

def acommutator(operator: ndarray) -> ndarray:
    """Construct anti-commutator superoperator from operator. """
    dim = operator.shape[0]
    return np.kron(operator, np.identity(dim)) \
            + np.kron(np.identity(dim), operator.T)

def left_right_super(
        left_operator: ndarray,
        right_operator: ndarray) -> ndarray:
    """Construct left and right acting superoperator from operators. """
    return np.kron(left_operator, right_operator.T)


