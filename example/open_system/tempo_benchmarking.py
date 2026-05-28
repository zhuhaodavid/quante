# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-27 20:36:46
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-28 02:37:26

import numpy as np
import matplotlib.pyplot as plt

import quante as qt
import quante.tensornetwork.opensystem as qto
from quante.generate.matrix import pauli_matrix

sigma_x = pauli_matrix("X")
sigma_z = pauli_matrix("Z")

up = pauli_matrix("u")
rho0 = up @ up.conj().T

Omega = 1.0
alpha = 0.3
omega_cutoff = 5.0

system = qto.System(0.5 * Omega * sigma_x)
correlation = qto.PowerLawSpectralDensity(alpha=alpha, zeta=1.0, cutoff=omega_cutoff)
bath = qto.Bath(0.5 * sigma_z, correlation)
params = qto.TempoParams(dt=0.1, tcut=3.0, epsrel=1e-4, trunc_cut=1e-4, apply_mpo_method='density_matrix')
ts = np.arange(0.0, 15.0 + params.dt / 2, params.dt)

tempo_engine = qto.TempoEngine(
    system,
    bath,
    params,
    rho0,
    ts,
)
for i in range(len(ts)):
    tempo_engine.run()


# times, sz = qto.tempo_compute(
#     system,
#     bath,
#     rho0,
#     ts,
#     params,
#     measure=0.5 * sigma_z,
#     real=True,
#     backend="mps",
#     progressbar=True,
# )


