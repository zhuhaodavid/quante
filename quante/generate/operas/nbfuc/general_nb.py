# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:23:37
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:24:06

from ....basicfun.utils_numba import njit, numba_cache_dir, config

config.CACHE_DIR = numba_cache_dir
@njit
def _quick_merge(res_pos, res_coef):
    total_len = len(res_pos)
    cur_coef = res_coef[0]
    cur_pos = 0
    prev_pos = res_pos[0]  # 引入局部变量存储上一个位置
    for i in range(1, total_len):
        tmp = res_pos[i]
        if (tmp == prev_pos).all():
            cur_coef += res_coef[i]
        else:
            res_pos[cur_pos] = prev_pos
            res_coef[cur_pos] = cur_coef
            cur_pos += 1
            cur_coef = res_coef[i]
            prev_pos = tmp
            
    res_pos[cur_pos] = res_pos[total_len-1]
    res_coef[cur_pos] = cur_coef
    mask = res_coef[:cur_pos + 1] != 0  # Remove zero coefficients
    return res_pos[:cur_pos+1][mask], res_coef[:cur_pos+1][mask]
