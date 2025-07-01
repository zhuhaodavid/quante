# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-26 17:28:00
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-26 17:34:01

#%%

import numpy as np
from quante.measure.curve_fit import fit
import matplotlib.pyplot as plt

xs = np.logspace(1, 5, 5)
ys = xs

# xs = np.linspace(1, 10, 10)
# ys = 3 * xs + 7 + 0.1 * np.random.normal(0, 1, xs.shape)
# f, c = fit(xs, ys)
# print(c)
# nxs = np.linspace(1, 10, 100)
# nys = f(nxs)
plt.plot(xs, ys, 'o', label='data')
plt.xscale('log')
plt.yscale('log')
# plt.plot(nxs, nys, label='fit')

