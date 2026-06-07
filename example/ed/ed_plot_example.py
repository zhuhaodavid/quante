# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-08 00:00:00

"""Plot data saved by ``ed_example.py``.

The calculation example saves ordinary nested HDF5 groups/datasets:

    metadata/
    parameters/
    scalars/
    data/

This script converts that nested structure back into one flat plotting
dictionary.  It also keeps a small legacy reader for files where
``parameters`` and ``scalars`` were stored as HDF5 attrs.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib.pyplot as plt
import numpy as np

import quante as qt


DEFAULT_INPUT = Path(__file__).with_name("xxz_dynamic_correlation.h5")


def _as_python_scalar(value):
    value = np.asarray(value)
    if value.shape == ():
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return item
    return value


def flatten_dynamic_results(saved):
    """Flatten the v2 ``metadata/parameters/scalars/data`` save format."""
    results = {}
    for group_name in ("metadata", "parameters", "scalars", "data"):
        if group_name in saved:
            for key, value in saved[group_name].items():
                results[key] = _as_python_scalar(value)
    return results


def load_dynamic_results(path):
    """Load new ``save_hdf5`` data, with compatibility for the old attrs file."""
    saved = qt.basicfun.load_hdf5(str(path), data="/")
    if "data" in saved and "scalars" in saved and "parameters" in saved:
        return flatten_dynamic_results(saved)

    results = dict(saved)
    with h5py.File(path, "r") as h5:
        for group_name in ("parameters", "scalars"):
            if group_name in h5:
                for key, value in h5[group_name].attrs.items():
                    results[key] = value
    return results


def plot_dynamic_results(results, output_path):
    tlist = np.asarray(results["tlist"], dtype=float)
    profiles = np.asarray(results["profiles"], dtype=float)
    distances = np.asarray(results["distances"], dtype=float)
    phase = np.asarray(results["phase_unwrapped"], dtype=float)
    phase_mask = np.asarray(results["phase_fit_mask"], dtype=bool)
    phase_signal = np.asarray(results["phase_signal"], dtype=np.complex128)

    L = int(results["L"])
    delta = float(results["Delta"])
    v_ba = float(results["v_ba"])
    v_gap = float(results["v_gap"])
    v_phase = float(results["v_phase"])
    q_phase = float(results["phase_q_primary"])

    fig, (ax, ax_spec) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.6),
        gridspec_kw={"height_ratios": [3.0, 1.35]},
    )
    mesh = ax.pcolormesh(tlist, distances, profiles.T, shading="auto", cmap="RdBu_r")
    fig.colorbar(mesh, ax=ax, pad=0.015, label=r"$C^{zz}(r,t)$")
    ax.plot(tlist, v_ba * tlist, color="black", linewidth=1.8, label=fr"BA $v={v_ba:.3f}$")
    ax.plot(
        tlist,
        v_gap * tlist,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
        label=fr"ED gap $v={v_gap:.3f}$",
    )
    ax.plot(
        tlist,
        v_phase * tlist,
        color="tab:green",
        linestyle=":",
        linewidth=1.3,
        label=fr"phase $v={v_phase:.3f}$",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$r=|i-j|$")
    ax.set_ylim(0, L // 2)
    ax.set_title(fr"XXZ $C^{{zz}}(r,t)$, $L={L}$, $\Delta={delta}$")
    ax.legend(frameon=False, loc="upper left")

    ax_spec.plot(tlist, phase, color="tab:blue", linewidth=1.2, label=r"$\arg S(q,t)$")
    ax_spec.plot(
        tlist,
        phase[0] - v_ba * q_phase * tlist,
        color="black",
        linewidth=1.2,
        label="BA",
    )
    ax_spec.plot(
        tlist,
        phase[0] - v_phase * q_phase * tlist,
        color="tab:green",
        linestyle=":",
        linewidth=1.2,
        label="phase fit",
    )
    ax_spec.plot(
        tlist[phase_mask],
        phase[phase_mask],
        ".",
        color="tab:red",
        markersize=3,
        label="fit window",
    )
    ax_spec.set_xlabel(r"$t$")
    ax_spec.set_ylabel(r"$\arg S(q,t)$")

    ax_spec_twin = ax_spec.twinx()
    ax_spec_twin.plot(
        tlist,
        np.abs(phase_signal),
        color="0.65",
        linewidth=0.9,
        alpha=0.8,
    )
    ax_spec_twin.set_ylabel(r"$|S(q,t)/S(q,0)|$")
    ax_spec_twin.tick_params(axis="y", colors="0.35")

    ax_spec.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="HDF5 file produced by ed_example.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="PNG output path. Defaults to the input path with .png suffix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".png")
    results = load_dynamic_results(input_path)
    plot_dynamic_results(results, output_path)
    print(f"saved figure: {output_path}")


if __name__ == "__main__":
    main()
