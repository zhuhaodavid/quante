# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-01 11:19:29
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-01 15:35:00
#

# import quante as qt
# import scipy as sp
# import torch as tc

# L = 15
# ham = qt.generate.operas.spin.heisenberg_operator(L, j=(1, 1, 1))
# basis = qt.generate.basis.spin_basis(L=L)
# mat1 = tc.tensor(ham.to_matrix(basis, sparse=True).toarray())
# with qt.basicfun.Timer():
#     engs1 = tc.linalg.eigh(mat1)
# print(engs1)
