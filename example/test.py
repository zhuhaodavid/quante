# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2024-12-30 14:31:20
# @Last Modified by:   dzwang
# @Last Modified time: 2025-04-19 17:08:24
import numpy as np
import quante as qt
from quante.basicfun import println
dtype = np.float64


N = 2
mps = qt.tensor.networks.MPS.generate_W_state(N, type="dual", dtype=dtype) 
mps_vec = mps.to_vector()
dir_vec = qt.generate.state.w(L=N).astype(mps.dtype).squeeze() # one down spin
print(np.allclose(mps_vec, dir_vec))


