# -*- coding: utf-8 -*-
"""Compatibility exports for NumPy tensor-network utilities.

The NumPy implementation keeps the actual implementations in
``tensor_utils.py``.  This module is intentionally kept as a lightweight
compatibility layer for code that imports ``core_utils``.
"""

from .tensor_utils import *  # noqa: F401,F403

