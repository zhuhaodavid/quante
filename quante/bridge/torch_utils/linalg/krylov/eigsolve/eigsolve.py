# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:19:37
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-28 19:19:29

import numpy as np

def eigsolve(A, x0, howmany=1, which='LM', **kwargs):
    from .arnoldi import Arnoldi
    alg = Arnoldi(**kwargs)
    assert which in ['LM', 'LR', 'SR', 'LI', 'SI']
    return alg.eigsolve(A, x0, howmany, which)

EIGSORT = {
    # "name": (sortfunction,  if_revert)
    "LM": (abs, True),
    "LR": (np.real, True),
    "SR": (np.real, False),
    "LI": (np.imag, True),
    "SI": (np.imag, False)
}

