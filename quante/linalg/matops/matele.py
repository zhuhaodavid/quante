# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:06:57
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:43:18

import numpy as _np

def uptrig(mat):
    """返回矩阵的对角元和上三角矩阵元"""
    from .nbfuc.matele_nb import uptri2list
    return uptri2list(mat)

def uptrig_inv(lis):
    """将 uptrig 上三角矩阵元素重新组装为上三角矩阵"""
    from .nbfuc.matele_nb import list2uptri
    return list2uptri(lis)

def uptrigindex(row_indx:list, col_indx:list, dim:int) -> _np.ndarray:
    """uptrig 将 (i,j) 指标变为生成的 list 的指标"""
    from .nbfuc.matele_nb import _uptrigindex
    return _uptrigindex(row_indx, col_indx, dim)

def uptrigindex_inv(indices:list, dim:int) -> _np.ndarray:
    """uptrig 生成的 list 的指标变为上三角矩阵的指标"""
    from .nbfuc.matele_nb import _uptrigindex_inv
    return _uptrigindex_inv(indices, dim)

