# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-07 13:28:50

"""XXZ dynamic correlation from ED.

Benchmark on this machine for ``L=24, Delta=0.7`` using
``quante.generate.operas`` + ``to_matrix`` in the ``Nup=L//2`` sector:

    dim = 2704156, nnz = 36564892
    basis + oper.to_matrix                  3.354 s
    eigsh(csr)                             12.523 s
    eigsh(LinearOperator + dot_parallel)   11.510 s
    quante.linalg.krylov.eigsolve          15.770 s
    scipy csr -> torch sparse csr cuda      2.479 s
    quante eigsolve(torch sparse cuda)      1.099 s

Benchmark mode still compares ARPACK ``eigsh`` with ``dot_parallel`` and
``eigsolve``.  The dynamic calculation uses the CUDA sparse ``eigsolve`` path
by default.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import h5py
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator, eigsh

import quante as qt
from quante.basicfun import Timer
from quante.generate import operas as op
from quante.linalg import make_evolve_engine
from quante.linalg.krylov.eigsolve.eigsolve import eigsolve
from quante.linalg.matops.sparse_mul import dot_parallel


L = int(os.environ.get("XXZ_L", 26))
Delta = float(os.environ.get("XXZ_DELTA", 0.7))
TOL = float(os.environ.get("XXZ_TOL", 1e-10))
MAXITER = int(os.environ.get("XXZ_MAXITER", 300))
KRYLOVDIM = int(os.environ.get("XXZ_KRYLOVDIM", 40))
T_MAX = float(os.environ.get("XXZ_TMAX", 3.0))
N_TIME = int(os.environ.get("XXZ_NTIME", 121))
J_SITE = int(os.environ.get("XXZ_JSITE", L // 2))
RIDGE_FIT_T_MAX = float(os.environ.get("XXZ_RIDGE_FIT_TMAX", 2.0))
MODE = os.environ.get("XXZ_MODE", "dynamic")
EVOLVE_METHOD = os.environ.get("XXZ_EVOLVE_METHOD", "mul-cuda:0")
OUTPUT_STEM = os.environ.get("XXZ_OUTPUT_STEM", "xxz_dynamic_correlation")
SCAN_KBLOCKS = os.environ.get("XXZ_SCAN_KBLOCKS", "1") != "0"


def xxz_sound_velocity(delta):
    r"""Bethe-ansatz spin velocity for the Pauli-matrix XXZ convention.

    The usual spin-1/2 result for
    ``H=sum Sx Sx + Sy Sy + Delta Sz Sz`` is
    ``pi sin(eta) / (2 eta)`` with ``Delta=cos(eta)``.  This example uses
    ``pauli=True``, i.e. ``S = sigma / 2``, so the Hamiltonian and velocity
    are larger by a factor of four.
    """
    eta = np.arccos(delta)
    return 4.0 * np.pi * np.sin(eta) / (2.0 * eta)


def build_xxz_matrix(L, delta):
    basis = qt.generate.basis.spin_basis(L=L, Nup=L // 2)
    ham = op.heisenberg_operator(
        L,
        j=(1.0, 1.0, delta),
        cyclic=True,
    )
    hmat = ham.to_matrix(basis=basis, pauli=True, sparse=True).tocsr()
    return basis, hmat


def build_xxz_matrix_kblock(L, delta, kblock):
    basis = qt.generate.basis.spin_basis(L=L, Nup=L // 2, kblock=kblock)
    ham = op.heisenberg_operator(
        L,
        j=(1.0, 1.0, delta),
        cyclic=True,
    )
    hmat = ham.to_matrix(basis=basis, pauli=True, sparse=True).tocsr()
    return basis, hmat


def parallel_linear_operator(mat):
    mat = mat.tocsr()

    def matvec(x):
        return dot_parallel(mat, x)

    return LinearOperator(
        mat.shape,
        matvec=matvec,
        rmatvec=matvec,
        dtype=mat.dtype,
    )


def eigsh_ground_state_parallel(hmat):
    evals, evecs = eigsh(
        parallel_linear_operator(hmat),
        k=1,
        which="SA",
        tol=TOL,
        maxiter=MAXITER,
    )
    psi0 = evecs[:, 0].astype(np.complex128)
    psi0 /= np.linalg.norm(psi0)
    return float(evals[0]), psi0


def eigsolve_ground_state_cuda(hmat, *, return_vector=True):
    import torch
    from quante.bridge.torch_utils.linalg.sparse import to_csr

    hmat_cuda = to_csr(hmat, device="cuda:0")
    if np.iscomplexobj(hmat):
        x0 = (
            torch.randn(hmat.shape[0], dtype=torch.float64, device="cuda:0")
            + 1j * torch.randn(hmat.shape[0], dtype=torch.float64, device="cuda:0")
        ).to(torch.complex128)
    else:
        x0 = torch.randn(hmat.shape[0], dtype=torch.float64, device="cuda:0")
    values, vectors, info = eigsolve(
        hmat_cuda,
        x0=x0,
        howmany=1,
        which="SR",
        isherm=True,
        tol=TOL,
        maxiter=MAXITER,
        krylovdim=KRYLOVDIM,
        verbosity=0,
    )
    torch.cuda.synchronize()
    energy = float(np.real(values[0]))
    if not return_vector:
        del hmat_cuda, x0, values, vectors
        torch.cuda.empty_cache()
        return energy, info

    if vectors.ndim == 1:
        vector = vectors
    elif vectors.shape[0] == hmat.shape[0]:
        vector = vectors[:, 0]
    else:
        vector = vectors[0]
    psi0 = vector.detach().cpu().numpy().astype(np.complex128).reshape(-1)
    psi0 /= np.linalg.norm(psi0)
    del hmat_cuda, x0, values, vectors
    torch.cuda.empty_cache()
    return energy, psi0, info


def sz_values_for_basis(basis, site, *, pauli=False):
    """Diagonal entries of ``S_site^z`` in a quante spin-half basis."""
    states = np.asarray(basis.s_list, dtype=np.int64)
    bits = (states >> (basis.L - 1 - site)) & 1
    values = 1.0 - 2.0 * bits
    if not pauli:
        values *= 0.5
    return values.astype(np.float32)


def connected_correlation_profile(psi0, psi_t, sz_values, j_site, one_point):
    source = np.conj(psi0) * psi_t
    corr = sz_values @ source
    corr -= one_point * one_point[j_site]
    return np.real_if_close(corr)


class CorrelationMeasurer:
    def __init__(self, psi0, sz_values, j_site, *, device_method):
        self.j_site = j_site
        self.use_torch = device_method.startswith("mul-cuda")
        if self.use_torch:
            import torch

            device = device_method.split("-", 1)[1]
            self.torch = torch
            self.psi0 = torch.as_tensor(psi0, dtype=torch.complex128, device=device)
            self.sz_values = torch.as_tensor(sz_values, dtype=torch.float64, device=device)
            one_point = self.sz_values @ torch.abs(self.psi0) ** 2
            self.one_point = one_point
        else:
            self.psi0 = psi0
            self.sz_values = sz_values
            self.one_point = sz_values @ np.abs(psi0) ** 2

    def initial_state(self):
        return self.sz_values[self.j_site] * self.psi0

    def profile(self, psi_t):
        if self.use_torch:
            psi_t = psi_t.reshape(-1)
            source = self.torch.conj(self.psi0) * psi_t
            corr = []
            for sz_i in self.sz_values:
                corr.append(self.torch.sum(source * sz_i))
            corr = self.torch.stack(corr)
            corr -= self.one_point.to(corr.dtype) * self.one_point[self.j_site]
            return corr.detach().cpu().numpy()
        return connected_correlation_profile(
            self.psi0,
            psi_t,
            self.sz_values,
            self.j_site,
            self.one_point,
        )


def fold_by_distance(corr, j_site):
    L = len(corr)
    max_distance = L // 2
    folded = np.empty(max_distance + 1, dtype=float)
    folded[0] = np.real(corr[j_site])
    for distance in range(1, max_distance + 1):
        right = corr[(j_site + distance) % L]
        left = corr[(j_site - distance) % L]
        if distance == L - distance:
            folded[distance] = np.real(right)
        else:
            folded[distance] = 0.5 * np.real(right + left)
    return folded


def estimate_velocity_from_structure_factor(tlist, correlations, q):
    dt = tlist[1] - tlist[0]
    L = correlations.shape[1]
    phase = np.exp(-1j * q * np.arange(L))
    signal = correlations @ phase
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    spectrum = np.fft.fft(signal * window)
    omega = 2.0 * np.pi * np.fft.fftfreq(len(signal), d=dt)
    mask = omega > 0
    omega_pos = omega[mask]
    weight = np.abs(spectrum[mask])
    peak = int(np.argmax(weight))
    omega_peak = float(omega_pos[peak])
    return omega_peak / q, omega_pos, weight


def estimate_velocity_from_ray_score(tlist, profiles, v_ba, *, fit_t_max=2.0):
    distances = np.arange(profiles.shape[1], dtype=float)
    max_distance = distances[-1]
    velocities = np.linspace(0.5 * v_ba, 1.5 * v_ba, 301)
    scores = np.zeros_like(velocities)

    for index, velocity in enumerate(velocities):
        samples = []
        for time, profile in zip(tlist, profiles):
            ray_distance = velocity * time
            if time <= 0.25 or time >= fit_t_max:
                continue
            if ray_distance < 1.0 or ray_distance > max_distance - 1.0:
                continue
            weight = np.abs(profile).astype(float)
            weight[0] = 0.0
            norm = np.max(weight)
            if norm <= 0.0:
                continue
            samples.append(np.interp(ray_distance, distances, weight / norm))
        if samples:
            scores[index] = float(np.mean(samples))

    best = int(np.argmax(scores))
    return float(velocities[best]), velocities, scores


def scan_kblock_ground_energies(L, delta):
    energies = np.empty(L, dtype=float)
    dimensions = np.empty(L, dtype=np.int64)
    for kblock in range(L):
        basis_q, hmat_q = build_xxz_matrix_kblock(L, delta, kblock)
        dimensions[kblock] = basis_q.Ns
        energies[kblock], info = eigsolve_ground_state_cuda(hmat_q, return_vector=False)
        print(
            f"kblock={kblock:2d}, dim={basis_q.Ns:7d}, "
            f"E0(k)={energies[kblock]:.12f}, ops={info.numops}"
        )
    return energies, dimensions


def kblock_gap_velocity(L, kblock_energies):
    ground_kblock = int(np.argmin(kblock_energies))
    ground_energy = float(kblock_energies[ground_kblock])
    left_kblock = (ground_kblock - 1) % L
    right_kblock = (ground_kblock + 1) % L
    if kblock_energies[left_kblock] <= kblock_energies[right_kblock]:
        excitation_kblock = left_kblock
    else:
        excitation_kblock = right_kblock
    excitation_energy = float(kblock_energies[excitation_kblock])
    q = 2.0 * np.pi / L
    return {
        "ground_kblock": ground_kblock,
        "excitation_kblock": excitation_kblock,
        "ground_energy": ground_energy,
        "excitation_energy": excitation_energy,
        "q": q,
        "velocity": (excitation_energy - ground_energy) / q,
    }


def estimate_velocity_from_momentum_gap(L, delta, e0):
    q = 2.0 * np.pi / L
    basis_q, hmat_q = build_xxz_matrix_kblock(L, delta, 1)
    evals_q = eigsh(
        parallel_linear_operator(hmat_q),
        k=1,
        which="SA",
        tol=TOL,
        maxiter=MAXITER,
        return_eigenvectors=False,
    )
    e_q = float(evals_q[0])
    return (e_q - e0) / q, e_q


def make_preferred_evolve_engine(hmat, psi, tlist):
    shifted_hmat = hmat - sps.identity(hmat.shape[0], format="csr") * 0.0
    return make_evolve_engine(
        shifted_hmat,
        psi,
        tlist,
        method=EVOLVE_METHOD,
        matrix_role="hamiltonian",
        herm=True,
    ), EVOLVE_METHOD


def to_numpy_state(state):
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    return np.asarray(state).reshape(-1)


def dynamic_parameters(*, used_method=None):
    return {
        "L": L,
        "Delta": Delta,
        "TOL": TOL,
        "MAXITER": MAXITER,
        "KRYLOVDIM": KRYLOVDIM,
        "T_MAX": T_MAX,
        "N_TIME": N_TIME,
        "J_SITE": J_SITE,
        "RIDGE_FIT_T_MAX": RIDGE_FIT_T_MAX,
        "MODE": MODE,
        "EVOLVE_METHOD": EVOLVE_METHOD,
        "USED_EVOLVE_METHOD": used_method or "",
        "SCAN_KBLOCKS": SCAN_KBLOCKS,
        "PAULI": True,
        "CYCLIC": True,
        "SECTOR": "Nup=L//2",
    }


def save_dynamic_results(results, output_path):
    with h5py.File(output_path, "w") as h5:
        h5.attrs["description"] = "XXZ connected dynamic Sz-Sz correlation from ED"
        h5.attrs["created_unix_time"] = time.time()
        for key, value in results["parameters"].items():
            h5.attrs[key] = value

        scalars = h5.create_group("scalars")
        for key in [
            "e0",
            "e0_per_site",
            "e_q",
            "q_min",
            "v_ba",
            "v_gap",
            "v_fft",
            "v_ray",
            "gap_ground_kblock",
            "gap_excitation_kblock",
        ]:
            scalars.attrs[key] = results[key]

        h5.create_dataset("tlist", data=results["tlist"])
        h5.create_dataset("distances", data=results["distances"])
        h5.create_dataset("correlations", data=results["correlations"], compression="gzip")
        h5.create_dataset("profiles", data=results["profiles"], compression="gzip")
        h5.create_dataset("omega", data=results["omega"])
        h5.create_dataset("spectrum", data=results["spectrum"])
        h5.create_dataset("ray_velocities", data=results["ray_velocities"])
        h5.create_dataset("ray_scores", data=results["ray_scores"])
        if "kblock_energies" in results:
            h5.create_dataset("kblock_energies", data=results["kblock_energies"])
            h5.create_dataset("kblock_dimensions", data=results["kblock_dimensions"])


def plot_dynamic_results(results, output_path):
    tlist = results["tlist"]
    profiles = results["profiles"]
    distances = results["distances"]
    omega = results["omega"]
    spectrum = results["spectrum"]
    q_min = results["q_min"]
    v_ba = results["v_ba"]
    v_gap = results["v_gap"]
    v_fft = results["v_fft"]
    v_ray = results["v_ray"]

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
        v_ray * tlist,
        color="tab:purple",
        linestyle="-.",
        linewidth=1.3,
        label=fr"ray score $v={v_ray:.3f}$",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$r=|i-j|$")
    ax.set_ylim(0, L // 2)
    ax.set_title(fr"XXZ $C^{{zz}}(r,t)$, $L={L}$, $\Delta={Delta}$")
    ax.legend(frameon=False, loc="upper left")

    ax_spec.plot(omega, spectrum, color="tab:blue", linewidth=1.2)
    ax_spec.axvline(v_ba * q_min, color="black", linewidth=1.4, label="BA")
    ax_spec.axvline(v_gap * q_min, color="tab:orange", linestyle="--", linewidth=1.4, label="ED gap")
    ax_spec.axvline(v_fft * q_min, color="tab:green", linestyle=":", linewidth=1.2, label="FFT peak")
    ax_spec.set_xlim(0, min(np.max(omega), 4.0 * v_ba * q_min))
    ax_spec.set_xlabel(r"$\omega$")
    ax_spec.set_ylabel(r"$|S(q,\omega)|$")
    ax_spec.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(output_path, dpi=200)
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def run_dynamic():
    if L % 2 != 0:
        raise ValueError("This example expects even L so that Nup=L//2.")

    timer = Timer()
    print(f"L={L}, Delta={Delta}, dim sector Nup={L // 2}")

    basis, hmat = build_xxz_matrix(L, Delta)
    print(f"dim={basis.Ns}, nnz={hmat.nnz}")
    timer.mark("basis + oper.to_matrix")

    print("solving ground state with eigsolve(torch sparse cuda) ...")
    e0, psi0, eigsolve_info = eigsolve_ground_state_cuda(hmat)
    print(f"E0={e0:.12f}, E0/L={e0 / L:.12f}")
    print(eigsolve_info)
    timer.mark("ground-state eigsolve cuda")

    print("building Sz diagonals ...")
    sz_values = np.array([
        sz_values_for_basis(basis, site, pauli=True)
        for site in range(L)
    ])
    psi = sz_values[J_SITE] * psi0
    timer.mark("Sz diagonals")

    tlist = np.linspace(0.0, T_MAX, N_TIME)
    shifted_hmat = hmat - e0 * sps.identity(hmat.shape[0], format="csr")
    engine, used_method = make_preferred_evolve_engine(shifted_hmat, psi, tlist)
    print(f"evolution method: {used_method}")
    timer.mark("evolve engine")
    measurer = CorrelationMeasurer(
        psi0,
        sz_values,
        J_SITE,
        device_method=used_method,
    )
    timer.mark("correlation measurer")

    correlations = []
    profiles = []
    for step, time_value in enumerate(tlist):
        psi = engine.step()
        corr = measurer.profile(psi)
        corr = np.asarray(corr, dtype=np.complex128)
        correlations.append(corr)
        profiles.append(fold_by_distance(corr, J_SITE))
        if step % max(1, len(tlist) // 10) == 0:
            print(f"evolved t={time_value:.3f}")
            timer.mark(f"evolution step {step}/{len(tlist) - 1}")
    correlations = np.asarray(correlations)
    profiles = np.asarray(profiles)
    timer.mark("time evolution")

    v_ba = xxz_sound_velocity(Delta)
    if SCAN_KBLOCKS:
        print("scanning all kblocks for finite-size velocity ...")
        kblock_energies, kblock_dimensions = scan_kblock_ground_energies(L, Delta)
        gap_info = kblock_gap_velocity(L, kblock_energies)
        v_gap = gap_info["velocity"]
        e_q = gap_info["excitation_energy"]
        gap_ground_kblock = gap_info["ground_kblock"]
        gap_excitation_kblock = gap_info["excitation_kblock"]
        print(
            "momentum gap from "
            f"k0={gap_ground_kblock} to k={gap_excitation_kblock}"
        )
    else:
        print("solving k=2pi/L block for finite-size velocity ...")
        v_gap, e_q = estimate_velocity_from_momentum_gap(L, Delta, e0)
        kblock_energies = None
        kblock_dimensions = None
        gap_ground_kblock = 0
        gap_excitation_kblock = 1
    timer.mark("momentum-gap solver")
    v_ray, ray_velocities, ray_scores = estimate_velocity_from_ray_score(
        tlist,
        profiles,
        v_ba,
        fit_t_max=RIDGE_FIT_T_MAX,
    )
    q_min = 2.0 * np.pi / L
    v_fft, omega, spectrum = estimate_velocity_from_structure_factor(
        tlist,
        correlations,
        q_min,
    )
    print(f"BA velocity: {v_ba:.8f}")
    print(f"ED gap velocity: {v_gap:.8f}  (E(q)-E0={e_q - e0:.12f})")
    print(f"ED gap kblocks: {gap_ground_kblock} -> {gap_excitation_kblock}")
    print(f"ED S(q=2pi/L,t) velocity: {v_fft:.8f}")
    print(f"ED ray-score velocity: {v_ray:.8f}")

    distances = np.arange(profiles.shape[1])
    results = {
        "parameters": dynamic_parameters(used_method=used_method),
        "e0": e0,
        "e0_per_site": e0 / L,
        "e_q": e_q,
        "q_min": q_min,
        "v_ba": v_ba,
        "v_gap": v_gap,
        "v_fft": v_fft,
        "v_ray": v_ray,
        "gap_ground_kblock": gap_ground_kblock,
        "gap_excitation_kblock": gap_excitation_kblock,
        "tlist": tlist,
        "distances": distances,
        "correlations": correlations,
        "profiles": profiles,
        "omega": omega,
        "spectrum": spectrum,
        "ray_velocities": ray_velocities,
        "ray_scores": ray_scores,
    }
    if kblock_energies is not None:
        results["kblock_energies"] = kblock_energies
        results["kblock_dimensions"] = kblock_dimensions

    stem = Path(__file__).with_name(OUTPUT_STEM)
    h5_path = stem.with_suffix(".h5")
    save_dynamic_results(results, h5_path)
    print(f"saved hdf5: {h5_path}")
    timer.mark("save hdf5")

    output_path = stem.with_suffix(".png")
    plot_dynamic_results(results, output_path)
    print(f"saved figure: {output_path}")
    timer.mark("plot/save")


def run_benchmark():
    timer = Timer()
    print(f"L={L}, Delta={Delta}, tol={TOL}, maxiter={MAXITER}")
    basis, hmat = build_xxz_matrix(L, Delta)
    print(f"dim={basis.Ns}, nnz={hmat.nnz}")
    timer.mark("basis + oper.to_matrix")

    evals, evecs = eigsh(hmat, k=1, which="SA", tol=TOL, maxiter=MAXITER)
    e_csr, v_csr = float(evals[0]), evecs[:, 0]
    timer.mark("eigsh(csr)")
    print(f"eigsh(csr) E0={e_csr:.15f}, E0/L={e_csr / L:.15f}")

    e_parallel, v_parallel = eigsh_ground_state_parallel(hmat)
    timer.mark("eigsh(LinearOperator dot_parallel)")
    print(
        "eigsh(parallel) "
        f"E0={e_parallel:.15f}, diff={e_parallel - e_csr:.3e}, "
        f"overlap={abs(np.vdot(v_csr, v_parallel)):.12f}"
    )

    x0 = np.random.default_rng(0).normal(size=hmat.shape[0])
    values, _, info = eigsolve(
        hmat,
        x0=x0,
        howmany=1,
        which="SR",
        isherm=True,
        tol=TOL,
        maxiter=MAXITER,
        krylovdim=KRYLOVDIM,
        verbosity=1,
    )
    e_qk = float(np.real(values[0]))
    timer.mark("quante eigsolve")
    print(f"eigsolve E0={e_qk:.15f}, diff={e_qk - e_csr:.3e}")
    print(info)

    try:
        import torch
        from quante.bridge.torch_utils.linalg.sparse import to_csr

        hmat_cuda = to_csr(hmat, device="cuda:0")
        torch.cuda.synchronize()
        timer.mark("to torch sparse csr cuda")

        x0_cuda = torch.randn(hmat.shape[0], dtype=torch.float64, device="cuda:0")
        values_cuda, vectors_cuda, info_cuda = eigsolve(
            hmat_cuda,
            x0=x0_cuda,
            howmany=1,
            which="SR",
            isherm=True,
            tol=TOL,
            maxiter=MAXITER,
            krylovdim=KRYLOVDIM,
            verbosity=1,
        )
        torch.cuda.synchronize()
        e_cuda = float(np.real(values_cuda[0]))
        timer.mark("quante eigsolve(torch sparse cuda)")
        print(f"eigsolve(cuda) E0={e_cuda:.15f}, diff={e_cuda - e_csr:.3e}")
        print(f"eigsolve(cuda) vector device={vectors_cuda.device}")
        print(info_cuda)
    except Exception as err:
        timer.mark("quante eigsolve(torch sparse cuda) failed")
        print(f"eigsolve(cuda) failed: {type(err).__name__}: {err}")


if __name__ == "__main__":
    if MODE == "benchmark":
        run_benchmark()
    elif MODE == "dynamic":
        run_dynamic()
    else:
        raise ValueError("XXZ_MODE should be 'dynamic' or 'benchmark'")
