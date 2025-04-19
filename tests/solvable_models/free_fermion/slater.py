# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-04-19 12:02:23
# @Last Modified by:   hzhu
# @Last Modified time: 2025-04-19 18:01:56

import unittest
import quante as qt
op = qt.generate.operas
from quante.solvable_models.gaussian_state import SlaterState
import numpy as np

class TestSlaterState(unittest.TestCase):
    def test_particle_number(self):
        L = 10
        J, γ = 1, 0.

        builder = op.SpinBuilder()
        for i in range(L-1):
            builder += "+-", [i+1, i], (J+γ)/2
            builder += "+-", [i, i+1], (J-γ)/2
        ham = builder.build()

        model = ham.jw_transfer()
        h = model.single_particle_ham(L)
        state = SlaterState.from_product_state("01"*(L//2))
        tlist = np.linspace(0, 60, 100)
        result = []
        for s in state.evolve(h, tlist):
            result.append(s.particle_number())
        result = np.abs(result)

        obs = [op.z(i) + 0.5 for i in range(L)]
        state = qt.generate.state.neel(L, down_first=True)
        result1 = ham.evolve(state, tlist, measure=obs, pauli=False)

        self.assertTrue(np.all(np.abs(result1 - result) < 1e-10))
    
    def test_entanglement(self):
        L = 10
        J, γ = 1, 0.

        builder = qt.generate.operas.SpinBuilder()
        for i in range(L-1):
            builder += "+-", [i+1, i], (J+γ)/2
            builder += "+-", [i, i+1], (J-γ)/2
        ham = builder.build()

        model = ham.jw_transfer()
        h = model.single_particle_ham(L)
        state = SlaterState.from_product_state("01"*(L//2))
        tlist = np.linspace(0, 800, 500)
        result = []
        for s in state.evolve(h, tlist):
            result.append(s.entanglement(range(L//2)))
        result = np.array(result)

        basis = qt.generate.basis.spin_basis(L)
        obs = [op.z(i) + 0.5 for i in range(L)]
        state = qt.generate.state.neel(L, down_first=True)
        obs = lambda state: qt.quantity.entanglement_entropy(state, L, L//2, basis=basis)
        result1 = ham.evolve(state, tlist, measure=obs, pauli=False, basis=basis)

        self.assertTrue(np.all(np.abs(result1 - result) < 1e-10))

    def test_2fermionstate(self):
        L = 8
        J, γ = 1, 0.

        builder = op.SpinBuilder()
        for i in range(L-1):
            builder += "+-", [i+1, i], (J+γ)/2
            builder += "+-", [i, i+1], (J-γ)/2
        ham = builder.build()

        model = ham.jw_transfer()
        h = model.single_particle_ham(L)
        state_str = "10"*(L//2) + "1"*(L%2)

        state = SlaterState.from_product_state(state_str, spin=True)
        tlist = np.linspace(0, 60, 100)
        result = []
        s = state
        for s in state.evolve(h, tlist):
            result.append(s.particle_number())
        # result = np.abs(result)

        psi1 = s._tovector()
        rho1 = s._todensirtmatrix()
        rhoA1 = s.reduced_density_matrix(range(L//2))

        state = qt.generate.state.product_state(state_str)
        basis = qt.generate.basis.spin_basis(L=L)
        mat = ham.to_matrix(basis)
        psi2 = qt.linalg.expm(mat, -1j*60) @ state

        rhoA2 = qt.linalg.partial_trace(psi2, [2]*L, range(L//2))
        rho2 = psi2 @ psi2.conj().T

        self.assertTrue(np.all(np.abs(psi1 - psi2) < 1e-10))
        self.assertTrue(np.all(np.abs(rho1 - rho2) < 1e-10))
        self.assertTrue(np.all(np.abs(rhoA1 - rhoA2) < 1e-10))


if __name__ == "__main__":
    unittest.main()

