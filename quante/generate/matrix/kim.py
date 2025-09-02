# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 20:37:42
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-02 18:31:22

import numpy as _np

from ...linalg.matops import kron_power, exp

def KIM_Hk(b:float, L:int):
    cosb, sinb = _np.cos(b), _np.sin(b)
    exp_sx = _np.array([[cosb, -1j*sinb], [-1j*sinb, cosb]])
    return exp_sx if L == 1 else kron_power(exp_sx, L)


def KIM_Hi(J:float, h:_np.ndarray, L:int):
    from .nbfuc.kim_nb import _Hi_model
    assert len(h) == L
    hammat = _Hi_model(J, h, L)
    return exp(hammat, -1j)


def KIM_matrix(b:float, J:float, h:_np.ndarray, L:int):
    return KIM_Hk(b, L) * KIM_Hi(J, h, L)

