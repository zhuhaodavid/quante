# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-11 16:21:17
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-12 21:25:25

from ......linalg.usenumba.numba_settings import njit, config, numba_cache_dir
import numpy as np

config.CACHE_DIR = numba_cache_dir

@njit("int64[:](int64, int64, int64)")  # @njit
def index2state(index, L, local_dim):
    state = np.zeros(L, dtype=np.int64)
    index_inv = local_dim**L - index - 1
    for i in range(L):
        state[i] = index_inv % local_dim
        index_inv //= local_dim
    return state[::-1]

config.CACHE_DIR = numba_cache_dir

@njit("int64(int64[:], int64, int64)")  # @njit
def state2index(state, L, local_dim):
    index = 0
    for i in range(L):
        index = index * local_dim + state[i]
    return local_dim**L - index - 1
