# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-06 11:12:44

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from quante.generate.solvable.bethe_ansatz import solve_xxz_state, xxz_energy
from quante.generate.solvable.bethe_ansatz.xxz import xxz_energy_from_rapidities


L = 100
Delta = 0.7
h_values = np.linspace(-4.0, 4.0, 801)


def sector_zero_field_energy(L, delta, n_down):
    """Lowest zero-field XXZ energy in the sector with ``n_down`` down spins."""
    n_down = int(n_down)
    n_ba = min(n_down, L - n_down)

    if n_ba == 0:
        return xxz_energy_from_rapidities([], L, delta, pauli=True)

    state = solve_xxz_state(
        L,
        delta,
        M=n_ba,
        raise_error=True,
    )
    return xxz_energy(state, pauli=True)


n_downs = np.arange(L + 1)
magnetizations = L - 2 * n_downs
zero_field_energies = np.array([
    sector_zero_field_energy(L, Delta, n_down)
    for n_down in n_downs
])
sector_energies = (
    zero_field_energies[:, None]
    - h_values[None, :] * magnetizations[:, None]
)
ground_energy = np.min(sector_energies, axis=0)
ground_sector = n_downs[np.argmin(sector_energies, axis=0)]

fig, ax = plt.subplots(figsize=(7, 3.6))
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(n_downs.min(), n_downs.max())

for n_down, magnetization, energy_h in zip(n_downs, magnetizations, sector_energies):
    ax.plot(
        h_values,
        energy_h,
        color=cmap(norm(n_down)),
        linewidth=0.75,
        alpha=0.55,
    )

ax.plot(
    h_values,
    ground_energy,
    color="black",
    linewidth=2.2,
    label="ground-state envelope",
)

switch_idx = np.flatnonzero(np.diff(ground_sector) != 0)
for idx in switch_idx:
    h_mid = 0.5 * (h_values[idx] + h_values[idx + 1])
    ax.axvline(h_mid, color="0.45", linestyle=":", linewidth=0.8)

ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$E_N^{(0)} - h (L - 2N)$")
ax.set_title(fr"XXZ sectors with field, $L={L}$, $\Delta={Delta}$")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
colorbar = fig.colorbar(sm, ax=ax, pad=0.015)
colorbar.set_label(r"$N_\downarrow$")
fig.tight_layout()
output_path = Path(__file__).with_name("xxz_sector_field_energy.png")
fig.savefig(output_path, dpi=200)
if plt.get_backend().lower() != "agg":
    plt.show()

print("sector zero-field energies")
for n_down, magnetization, energy in zip(n_downs, magnetizations, zero_field_energies):
    print(f"N_down={n_down:2d}, Sz={magnetization:3d}, E0={energy:.12f}")
print(f"saved figure: {output_path}")
