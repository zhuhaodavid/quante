# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-06 11:12:44

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from quante.generate.solvable.bethe_ansatz.xxz_z import (
    plot_xxz_pbc_finite_energy_vs_h,
)


L = 100
Delta = 0.7
h_values = np.linspace(-4.0, 4.0, 801)


fig, ax, data = plot_xxz_pbc_finite_energy_vs_h(
    L,
    Delta,
    h_values,
    pauli=True,
)

output_path = Path(__file__).with_name("xxz_sector_field_energy.png")
# fig.savefig(output_path, dpi=200)
if plt.get_backend().lower() != "agg":
    plt.show()

print("sector zero-field energies")
for n_down, magnetization, energy in zip(
    data["n_downs"],
    data["magnetizations"],
    data["zero_field_energies"],
):
    print(f"N_down={n_down:2d}, Sz={magnetization:3.0f}, E0={energy:.12f}")
print(f"saved figure: {output_path}")
