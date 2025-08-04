# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-11 16:10:37
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 20:15:16

from ....basicfun.utils_numba import njit, types, config, numba_cache_dir
import numpy as _np

# 'Tuple((f8,i8))(i8[:],i8[:],f8,i8[:])', 
@njit(inline='always')
def operateon(opnm, posn, S, sb):
    """
    Apply the operator opnm to the state s.
    """
    opco = 1.
    for i in range(len(opnm)-1,-1,-1):
        oi = opnm[i]
        pi = posn[i]
        if oi == 0:
            if sb[pi] == 0:
                return _np.nan
            sb[pi] -= 1
            m = sb[pi] - S
            opco *= (S * (S + 1) - m * (m + 1))**0.5
        elif oi == 1:
            if sb[pi] == _np.int64(2 * S):
                return _np.nan
            m = sb[pi] - S
            opco *= (S * (S + 1) - m * (m + 1))**0.5
            sb[pi] += 1
        elif oi == 3:
            opco *= 2*(sb[pi] - S)
            sb = sb
        else:
            sb = sb
    return opco


config.CACHE_DIR = numba_cache_dir
@njit("i8(i8[:,:], i8[:])")
def findstate(s_list, target):
    # 初始化左右指针
    Ns, L = s_list.shape
    left, right = 0, Ns - 1
    
    while left <= right:
        mid = (left + right) // 2
        # 比较当前中间行和目标行
        s_list_mid = s_list[mid]
        
        for i in range(L):
            if s_list_mid[i] > target[i]:
                left = mid + 1
                break
            elif s_list_mid[i] < target[i]:
                right = mid - 1
                break
        else:
            return mid
    
    # 如果没有找到，返回 -1
    return -1
