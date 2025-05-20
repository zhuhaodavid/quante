# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 18:34:51
# @Last Modified by:   hzhu
# @Last Modified time: 2025-05-20 12:13:44

# symmetry 中只提供了一维自旋链相关的函数
from .symmetry.basis_wrapped import spin_basis

# quspin 中实现了一维、二维，自旋、玻色、费米以及其他自定义基矢的生成
from .quspin.basis_1d_wrapped import quspin_fermion_basis, quspin_spinful_fermion_basis, quspin_boson_basis, quspin_spin_basis
