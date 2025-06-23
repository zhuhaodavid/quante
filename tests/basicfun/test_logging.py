# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-23 15:17:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-23 15:21:59

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




