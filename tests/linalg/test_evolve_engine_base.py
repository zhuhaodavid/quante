# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-30 00:00:00
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-30 00:00:00

import unittest

from quante.linalg.evolve.evolve_engine import EvolveEngineBase


class AddDtEngine(EvolveEngineBase):
    def __init__(self, ts):
        super().__init__(ts)
        self.cur_state = 0.0

    def propagate(self, state, dt):
        return state + dt


class TestEvolveEngineBase(unittest.TestCase):
    def test_step_after_tlist_raises(self):
        engine = AddDtEngine([0.1, 0.2])
        self.assertFalse(engine.finished)
        self.assertAlmostEqual(engine.step(), 0.1)
        self.assertFalse(engine.finished)
        self.assertAlmostEqual(engine.step(), 0.2)
        self.assertTrue(engine.finished)
        with self.assertRaises(StopIteration):
            engine.step()


if __name__ == "__main__":
    unittest.main()
