# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 20:42:42
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-02 18:41:40

import numpy as _np

from ....linalg.matops import kron
from ..random import _cue, _rand_simple_complex

def random_phase_model(L, q, ε, seed=None):
    """随机相位模型，
    
    但 ε 不太确定，这里用 ε**2 似乎才能与文献对的比较好（q=3 -> ε_c=0.25）
    
    """
    # get W1
    Ui = [_cue(q, seed=seed) for _ in range(L)]
    W1 = kron(*Ui)
    
    # get W2
    phi = [ε**2*_rand_simple_complex(q, seed) for _ in range(L-1)]
    dim = q**L
    W2 = _np.zeros((dim, dim), dtype=complex)
    
    for i in range(dim):
        res = 0
        for j in range(L-1):
            an = _spin_at_i(state=i, pos=j, L=L)
            anp1 = _spin_at_i(state=i, pos=j+1, L=L)
            res += phi[j][an, anp1]
        W2[i,i] = _np.exp(1j * res)
    return W2 @ W1

def _spin_at_i(state, pos, L):
    idf = 1 << (L - pos - 1)
    return 0 if state & idf == 0 else 1


