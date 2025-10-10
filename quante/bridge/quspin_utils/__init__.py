# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-07 12:03:27
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 20:05:28

import os
try:
    os.environ["KMP_DUPLICATE_LIB_OK"] = (
        "True"  # uncomment this line if omp error occurs on OSX for python 3
    )
    os.environ["OMP_NUM_THREADS"] = "1"  # set number of OpenMP threads to run in parallel
    os.environ["MKL_NUM_THREADS"] = "1"  # set number of MKL threads to run in parallel
except:
    pass


from .quspin_example import *
from .quspin_extension_wrap import *
