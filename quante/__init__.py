# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2023-09-15 13:40:25
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-07 16:46:29
"""Quante - 一个用于量子物理的Python库

用于处理张量网络、精确对角化、保存工具、日志工具等的工具库。

提供功能：
  1. 更好的日志记录、保存（hdf5）、随机工具
  2. 自适应特征值、SVD方法
  3. 量子态、算符、张量网络生成器

关于 python 计算效率的说明
  1. 首先要确认是否真的需要调优，不要在没有意义的优化上花费时间，优先保证简单易用！
  2. 80/20法则：百分之二十的代码产生了百分之八十的时间消耗。
  2. 对于简单数据处理的调优使用 numba，对于复杂数据结构的调优请改用 julia。
"""
# 关于 python 的效率问题，参考 https://www.cnblogs.com/traditional/tag/Cython/ 中的解释


from . import basicfun   # 提供关于日志和储存的一些基本功能
from . import quantity

from . import solvable_models
from . import linalg
from . import generate
from . import tensor
