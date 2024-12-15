# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-12 22:04:17
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-15 20:18:24

from ......linalg.usenumba.numba_settings import njit, config, numba_cache_dir
import numpy as np

config.CACHE_DIR = numba_cache_dir
@njit("i8[:,:](i8, i8, i8, i8)")
def generate_all_states2(L, Ns, local_dim, Nup): # todo 有么有更好的方法？
    all_states = np.zeros((Ns, L), dtype=np.int64)
    stack = [(0, np.zeros(L, dtype=np.int64), 0)]  # 栈模拟递归 (current_sum, current_state, depth)
    count = 0
    
    while len(stack) > 0:
        current_sum, current_state, depth = stack.pop()
        
        # 当满足条件时，保存状态
        if current_sum == Nup and depth == L:
            all_states[count] = current_state
            count += 1
            continue
        
        # 如果超过限制，直接跳过
        if current_sum > Nup or depth >= L or current_sum + (local_dim-1) * (L-depth) < Nup:
            continue

        # 遍历每个可能的值，尝试生成新的状态
        for i in range(local_dim):
            new_state = current_state.copy()
            new_state[depth] = i
            stack.append((current_sum + i, new_state, depth + 1))

    return all_states[:count]
