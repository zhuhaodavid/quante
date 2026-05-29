# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:31:48
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-29 12:57:48

import builtins
from .utils_logging import println, tqdm
from .utils_profile import Timer

__all__ = ['tqdm']

builtins.show = println
builtins.Timer = Timer

from functools import partial
builtins.tqdm = partial(tqdm, ascii=True)
