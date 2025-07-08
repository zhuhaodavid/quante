# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-01 11:19:29
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-08 16:48:02
#

# import quante as qt
# import numpy as np
# from quante.bridge.tenpy_utils import tenpy_tebd_GS_, tenpy_tebd_params_imag_time, tenpy_model_tebd
# import tenpy

# g = 1.0
# op = qt.generate.operas.spin
# L = 10
# ham = - op.sum(op.xx(i, i+1) for i in range(L-1)) - g * op.sum(op.z(i) for i in range(L))

# M = tenpy_model_tebd(
#     L=L, oper=ham, pauli=True, conserve='None', bc_MPS='finite',
# )
# psi = M.product_state(["up"] * L)

# tebd_params = tenpy_tebd_params_imag_time(
#         delta_tau_list=[0.1, 0.01, 0.001, 1.e-4, 1.e-5],
#         order=2,
#         chi_max=30,
#         svd_min=1.e-10,
#         max_error_E=1.e-6,
#     )

# tenpy_tebd_GS_(psi, M, tebd_params)
# E = np.sum(M.bond_energies(psi))  # M.bond_energies() works only a for NearestNeighborModel
# print("E = {E:.13f}".format(E=E))
# E_exact = qt.generate.operas.heisenberg_operator(L=L, j=(-1,0,0), h=-1.0).gdenergy(pauli=True)
# print("Analytic result: E (per site) = {E:.13f}".format(E=E_exact))
# print("relative error: ", abs((E - E_exact) / E_exact))