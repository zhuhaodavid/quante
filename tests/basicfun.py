# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-02-06 14:13:35
# @Last Modified by:   hzhu
# @Last Modified time: 2025-02-06 14:15:15

import unittest
from quante.basicfun import profile

class TestProfileDecorator(unittest.TestCase):
    def test_profile_enabled(self):
        @profile(on=True)
        def test_func():
            a = 1
            b = 2
            c = a + b
            return c

        result = test_func()
        self.assertEqual(result, 3)

    def test_profile_disabled(self):
        @profile(on=False)
        def test_func():
            a = 1
            b = 2
            c = a + b
            return c

        result = test_func()
        self.assertEqual(result, 3)

    def test_profile_save(self):
        @profile(on=True, save=True)
        def test_func():
            a = 1
            b = 2
            c = a + b
            return c

        result = test_func()
        self.assertEqual(result, 3)

    def test_profile_output_unit(self):
        @profile(on=True, output_unit=1e-6)
        def test_func():
            a = 1
            b = 2
            c = a + b
            return c

        result = test_func()
        self.assertEqual(result, 3)

if __name__ == "__main__":
   unittest.main()