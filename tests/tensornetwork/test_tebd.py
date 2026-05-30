# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-30 00:00:00
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-30 00:00:00

import unittest

import numpy as np

import quante as qt
import quante.tensornetwork as qtn


class TestTEBD(unittest.TestCase):
    def test_tebd_matches_manual_gate_loop(self):
        L = 4
        tau = 0.1
        steps = 5
        ham = qt.generate.operas.heisenberg_operator(L)
        psi_manual = qtn.MPS.from_product_state(["up", "down"] * (L // 2), dtype=np.complex128)
        psi_engine = psi_manual.copy()

        gates = ham.trotter_gates(L, tau=tau, pauli=False)
        manual_res = []
        for _ in range(steps):
            for pos, gate in zip(*gates):
                psi_manual.apply_gate_(pos, gate)
            manual_res.append(psi_manual.measure("z", [3]).item().real)

        engine = qtn.TEBD(ham, psi_engine, tau=tau, steps=steps, pauli=False)
        engine_res = engine.run(measure="z", pos=[3], progressbar=False).reshape(-1)

        self.assertTrue(np.allclose(engine_res, manual_res))
        self.assertTrue(np.allclose(engine.cur_state.to_vector(), psi_manual.to_vector()))


if __name__ == "__main__":
    unittest.main()
