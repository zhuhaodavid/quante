# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-04 19:45:36
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-06 13:25:22

import unittest
import quante as qt
import numpy as np

try:
    import tenpy
    from quante.generate.tenpy_bridge import (
        tenpy_model_tebd,
        tenpy_tebd_GS_,
        tenpy_tebd_params_imag_time, 
    )
    tenpy_available = True
except ImportError:
    tenpy_available = False


class TestSpinHalfProj(unittest.TestCase):

    @unittest.skipIf(not tenpy_available, "tenpy is not available")
    def test_imag_time_tebd_finite(self):
        # build model
        op = qt.generate.operas.spin
        L = 10
        g = 1.0
        ham = - op.sum(op.xx(i, i+1) for i in range(L-1)) - g * op.sum(op.z(i) for i in range(L))
        M = tenpy_model_tebd(
            L=L, oper=ham, pauli=True, conserve='None', bc_MPS='finite',
        )

        # # set the initial state
        product_state = ["up"] * M.lat.N_sites
        psi = tenpy.networks.MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        
        # set tebd parameters
        tebd_params = tenpy_tebd_params_imag_time(
            delta_tau_list=[0.1, 0.01, 0.001, 1.e-4, 1.e-5],
            order=2,
            N_steps=10,
            chi_max=30,
            svd_min=1.e-10,
            max_error_E=1.e-6,
        )
        eng = tenpy_tebd_GS_(psi, M, tebd_params)
        E = np.sum(M.bond_energies(psi))  # M.bond_energies() works only a for NearestNeighborModel
        E_exact = qt.generate.operas.heisenberg_operator(L=L, j=(-1,0,0), h=-1.0).gdenergy(pauli=True)
        self.assertAlmostEqual(E, E_exact, places=5)


    @unittest.skipIf(not tenpy_available, "tenpy is not available")
    def test_imag_time_tebd_infinite(self):
        # build model
        g = 1.0
        L = 2
        op = qt.generate.operas.spin
        ham = - op.sum(op.xx(i, i+1) for i in range(L)) - g * op.sum(op.z(i) for i in range(L))
        M = tenpy_model_tebd(
            L=L, oper=ham, pauli=True, conserve='None', bc_MPS='infinite',
        )

        # # set the initial state
        product_state = ["up"] * M.lat.N_sites
        psi = tenpy.networks.MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
        
        # set tebd parameters
        tebd_params = tenpy_tebd_params_imag_time(
            delta_tau_list=[0.1, 0.01, 0.001, 1.e-4, 1.e-5],
            order=2,
            N_steps=10,
            chi_max=30,
            svd_min=1.e-10,
            max_error_E=1.e-6,
        )
        eng = tenpy_tebd_GS_(psi, M, tebd_params)
        
        # expectation values
        E = np.mean(M.bond_energies(psi))  # M.bond_energies() works only a for NearestNeighborModel
        E_exact = qt.generate.operas.heisenberg_operator(L=np.inf, j=(-1,0,0), h=-1.0).gdenergy(pauli=True)
        self.assertAlmostEqual(E, E_exact, places=4)

