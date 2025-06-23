# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:31:48
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-23 16:51:34

import builtins
from .utils_logging import println
from .utils_profile import Timer

__all__ = ['tqdm']

builtins.show = println
builtins.Timer = Timer

from tqdm import tqdm
from functools import partial
builtins.tqdm = partial(tqdm, ascii=True)
