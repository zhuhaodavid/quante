# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:31:03
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-28 22:13:12

from . import grad_krylov

# the following codes are based on KrylovKit.jl
# todo: 
# 1. implement eigsolve for real non-symmetric matrices and hermitian matrices
# 2. implement ortho method like ClassicalGramSchmidtIR,ModifiedGramSchmidtIR
# 3. implement svdsolve
# 4. add support for torch-cuda
# 5. implement in tensor-network language

from .eigsolve.eigsolve import eigsolve
