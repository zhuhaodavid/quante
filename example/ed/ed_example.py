# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-06-07 14:11:17
# @Last Modified by:   hzhu
# @Last Modified time: 2026-06-08 01:05:27

"""XXZ ED toolkit example for future AI-assisted Quante use.

Purpose
-------
This file is a compact, executable catalogue of Quante exact-diagonalization
patterns.  It is intentionally more explicit than a minimal physics script:
future AI agents should be able to copy small pieces from here when they need
to use this library correctly.

What to learn from this file
----------------------------
1. Build symmetry-resolved spin bases and XXZ Hamiltonians with
   ``qt.generate.basis.spin_basis`` and ``quante.generate.operas``.
2. Compare ground-state solvers:
   SciPy CSR ARPACK, SciPy ``LinearOperator`` + ``dot_parallel``, Quante
   Krylov ``eigsolve``, and CUDA sparse tensors.
3. Construct diagonal local observables directly from a basis when doing so is
   cheaper than materializing operator matrices.
4. Use ``make_evolve_engine`` for real-time dynamics and measure connected
   correlation functions on CPU or CUDA.
5. Estimate a finite-size velocity from momentum-sector gaps, and estimate a
   dynamic velocity from the spatial Fourier transform ``C(r,t) -> S(q,t)``
   using phase fitting instead of a short-time FFT.
6. Save results with ``quante.basicfun.save_hdf5`` as ordinary nested
   groups/datasets.  The matching loader is ``qt.basicfun.load_hdf5(path, "/")``.

Plotting is intentionally kept out of this file.  Use ``ed_plot_example.py``
to load the saved HDF5 data and make diagnostic figures. 

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

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator, eigsh

import quante as qt
from quante.basicfun import Timer, save_hdf5
from quante.generate import operas as op
from quante.linalg import make_evolve_engine
from quante.linalg.krylov.eigsolve.eigsolve import eigsolve
from quante.linalg.matops.sparse_mul import dot_parallel


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

L = int(os.environ.get("XXZ_L", 26))
Delta = float(os.environ.get("XXZ_DELTA", 0.7))
TOL = float(os.environ.get("XXZ_TOL", 1e-10))
MAXITER = int(os.environ.get("XXZ_MAXITER", 300))
KRYLOVDIM = int(os.environ.get("XXZ_KRYLOVDIM", 40))
T_MAX = float(os.environ.get("XXZ_TMAX", 3.0))
N_TIME = int(os.environ.get("XXZ_NTIME", 121))
J_SITE = int(os.environ.get("XXZ_JSITE", L // 2))
PHASE_FIT_T_MIN = float(os.environ.get("XXZ_PHASE_FIT_TMIN", 0.5))
PHASE_FIT_T_MAX_RAW = os.environ.get("XXZ_PHASE_FIT_TMAX")
PHASE_FIT_T_MAX = (
    None
    if PHASE_FIT_T_MAX_RAW in (None, "", "none", "None")
    else float(PHASE_FIT_T_MAX_RAW)
)
PHASE_FIT_M_LIST = tuple(
    int(item.strip())
    for item in os.environ.get("XXZ_PHASE_FIT_M_LIST", "1,2,3,4").split(",")
    if item.strip()
)
MODE = os.environ.get("XXZ_MODE", "dynamic")
EVOLVE_METHOD = os.environ.get("XXZ_EVOLVE_METHOD", "mul-cuda:0")
OUTPUT_STEM = os.environ.get("XXZ_OUTPUT_STEM", "xxz_dynamic_correlation")
SCAN_KBLOCKS = os.environ.get("XXZ_SCAN_KBLOCKS", "1") != "0"
RESULT_SCHEMA = "ed_dynamic_results_v2"


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Ground-state solvers and sparse-matrix backends
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Local observables and real-time correlation measurement
# ---------------------------------------------------------------------------

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


def signed_r_grid(L, j_site):
    """Signed displacement grid in the same site order as ``correlations``."""
    sites = np.arange(L)
    return ((sites - j_site + L // 2) % L) - L // 2


# ---------------------------------------------------------------------------
# Momentum-space velocity estimators
# ---------------------------------------------------------------------------

def structure_factor_time_from_correlation(correlations, q, j_site, *, normalize=True):
    r"""Convert site-resolved ``C_ij(t)`` to ``S(q,t)=sum_r exp(-iqr) C(r,t)``."""
    correlations = np.asarray(correlations, dtype=np.complex128)
    L = correlations.shape[1]
    r = signed_r_grid(L, j_site)
    signal = correlations @ np.exp(-1j * q * r)
    if normalize:
        if abs(signal[0]) < 1e-14:
            raise ValueError("S(q,0) is too small to normalize.")
        signal = signal / signal[0]
    return signal


def estimate_phase_frequency(tlist, signal, *, tmin=0.5, tmax=None, amp_cut=0.05):
    r"""Estimate ``omega`` from ``S(q,t) ~ exp(-i omega t)`` by phase fitting."""
    tlist = np.asarray(tlist, dtype=float)
    signal = np.asarray(signal, dtype=np.complex128)
    phase = np.unwrap(np.angle(signal))
    amp = np.abs(signal)

    mask = tlist >= tmin
    if tmax is not None:
        mask &= tlist <= tmax
    mask &= amp > amp_cut * np.max(amp)

    if np.count_nonzero(mask) < 3:
        return np.nan, np.nan, phase, mask

    slope, intercept = np.polyfit(
        tlist[mask],
        phase[mask],
        deg=1,
        w=amp[mask],
    )
    return abs(float(-slope)), float(intercept), phase, mask


def estimate_velocity_from_correlation_phase(
    tlist,
    correlations,
    L,
    j_site,
    m,
    *,
    tmin=0.5,
    tmax=None,
):
    q = 2.0 * np.pi * m / L
    signal = structure_factor_time_from_correlation(
        correlations,
        q,
        j_site,
        normalize=True,
    )
    omega, intercept, phase, mask = estimate_phase_frequency(
        tlist,
        signal,
        tmin=tmin,
        tmax=tmax,
    )
    return {
        "m": m,
        "q": q,
        "omega": omega,
        "v": omega / q,
        "intercept": intercept,
        "signal": signal,
        "phase": phase,
        "mask": mask,
    }


def estimate_vs_from_many_m(
    tlist,
    correlations,
    L,
    j_site,
    m_list=(1, 2, 3, 4),
    *,
    tmin=0.5,
    tmax=None,
):
    rows = []
    for m in m_list:
        item = estimate_velocity_from_correlation_phase(
            tlist,
            correlations,
            L,
            j_site,
            m,
            tmin=tmin,
            tmax=tmax,
        )
        if np.isfinite(item["omega"]):
            rows.append(item)

    qs = np.array([item["q"] for item in rows], dtype=float)
    omegas = np.array([item["omega"] for item in rows], dtype=float)
    velocities = np.array([item["v"] for item in rows], dtype=float)

    if len(qs) == 0:
        return {
            "rows": rows,
            "qs": qs,
            "omegas": omegas,
            "velocities": velocities,
            "v_linear": np.nan,
            "v_cubic": np.nan,
            "a_cubic": np.nan,
        }

    v_linear = float(np.sum(qs * omegas) / np.sum(qs ** 2))
    if len(qs) >= 2:
        coeff, *_ = np.linalg.lstsq(
            np.column_stack([qs, qs ** 3]),
            omegas,
            rcond=None,
        )
        v_cubic = float(coeff[0])
        a_cubic = float(coeff[1])
    else:
        v_cubic = np.nan
        a_cubic = np.nan

    return {
        "rows": rows,
        "qs": qs,
        "omegas": omegas,
        "velocities": velocities,
        "v_linear": v_linear,
        "v_cubic": v_cubic,
        "a_cubic": a_cubic,
    }


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


# ---------------------------------------------------------------------------
# Time evolution and HDF5 result schema
# ---------------------------------------------------------------------------

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
        "PHASE_FIT_T_MIN": PHASE_FIT_T_MIN,
        "PHASE_FIT_T_MAX": "" if PHASE_FIT_T_MAX is None else PHASE_FIT_T_MAX,
        "PHASE_FIT_M_LIST": ",".join(str(m) for m in PHASE_FIT_M_LIST),
        "MODE": MODE,
        "EVOLVE_METHOD": EVOLVE_METHOD,
        "USED_EVOLVE_METHOD": used_method or "",
        "SCAN_KBLOCKS": SCAN_KBLOCKS,
        "PAULI": True,
        "CYCLIC": True,
        "SECTOR": "Nup=L//2",
    }


def save_dynamic_results(results, output_path):
    scalar_keys = [
        "e0",
        "e0_per_site",
        "e_q",
        "q_min",
        "v_ba",
        "v_gap",
        "v_phase",
        "v_phase_linear",
        "v_phase_cubic",
        "phase_a_cubic",
        "phase_q_primary",
        "gap_ground_kblock",
        "gap_excitation_kblock",
    ]
    array_keys = [
        "tlist",
        "distances",
        "signed_distances",
        "correlations",
        "profiles",
        "phase_qs",
        "phase_omegas",
        "phase_velocities",
        "phase_signal",
        "phase_unwrapped",
        "phase_fit_mask",
    ]

    data = {
        "metadata": {
            "description": "XXZ connected dynamic Sz-Sz correlation from ED",
            "created_unix_time": time.time(),
            "schema": RESULT_SCHEMA,
        },
        "parameters": results["parameters"],
        "scalars": {key: results[key] for key in scalar_keys},
        "data": {key: results[key] for key in array_keys},
    }
    if "kblock_energies" in results:
        data["data"]["kblock_energies"] = results["kblock_energies"]
        data["data"]["kblock_dimensions"] = results["kblock_dimensions"]

    save_hdf5(str(output_path), data=data, group="/", mode="w")


# ---------------------------------------------------------------------------
# Executable modes
# ---------------------------------------------------------------------------

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
    q_min = 2.0 * np.pi / L
    phase_fit = estimate_vs_from_many_m(
        tlist,
        correlations,
        L,
        J_SITE,
        PHASE_FIT_M_LIST,
        tmin=PHASE_FIT_T_MIN,
        tmax=PHASE_FIT_T_MAX,
    )
    if phase_fit["rows"]:
        primary_phase = phase_fit["rows"][0]
        v_phase = float(primary_phase["v"])
        phase_q_primary = float(primary_phase["q"])
        phase_signal = primary_phase["signal"]
        phase_unwrapped = primary_phase["phase"]
        phase_fit_mask = primary_phase["mask"]
    else:
        v_phase = np.nan
        phase_q_primary = q_min
        phase_signal = np.full_like(tlist, np.nan, dtype=np.complex128)
        phase_unwrapped = np.full_like(tlist, np.nan, dtype=float)
        phase_fit_mask = np.zeros_like(tlist, dtype=bool)
    print(f"BA velocity: {v_ba:.8f}")
    print(f"ED gap velocity: {v_gap:.8f}  (E(q)-E0={e_q - e0:.12f})")
    print(f"ED gap kblocks: {gap_ground_kblock} -> {gap_excitation_kblock}")
    print("ED phase velocity from C(r,t) -> S(q,t):")
    for item in phase_fit["rows"]:
        print(
            f"  m={item['m']:2d}, q={item['q']:.8f}, "
            f"omega={item['omega']:.10f}, v={item['v']:.10f}"
        )
    print(f"ED phase linear fit velocity: {phase_fit['v_linear']:.8f}")
    print(f"ED phase cubic fit velocity: {phase_fit['v_cubic']:.8f}")

    distances = np.arange(profiles.shape[1])
    signed_distances = signed_r_grid(L, J_SITE)
    results = {
        "parameters": dynamic_parameters(used_method=used_method),
        "e0": e0,
        "e0_per_site": e0 / L,
        "e_q": e_q,
        "q_min": q_min,
        "v_ba": v_ba,
        "v_gap": v_gap,
        "v_phase": v_phase,
        "v_phase_linear": phase_fit["v_linear"],
        "v_phase_cubic": phase_fit["v_cubic"],
        "phase_a_cubic": phase_fit["a_cubic"],
        "phase_q_primary": phase_q_primary,
        "gap_ground_kblock": gap_ground_kblock,
        "gap_excitation_kblock": gap_excitation_kblock,
        "tlist": tlist,
        "distances": distances,
        "signed_distances": signed_distances,
        "correlations": correlations,
        "profiles": profiles,
        "phase_qs": phase_fit["qs"],
        "phase_omegas": phase_fit["omegas"],
        "phase_velocities": phase_fit["velocities"],
        "phase_signal": phase_signal,
        "phase_unwrapped": phase_unwrapped,
        "phase_fit_mask": phase_fit_mask,
    }
    if kblock_energies is not None:
        results["kblock_energies"] = kblock_energies
        results["kblock_dimensions"] = kblock_dimensions

    stem = Path(__file__).with_name(OUTPUT_STEM)
    h5_path = stem.with_suffix(".h5")
    save_dynamic_results(results, h5_path)
    print(f"saved hdf5: {h5_path}")
    timer.mark("save hdf5")
    print(f"plot with: python {Path(__file__).with_name('ed_plot_example.py')} {h5_path}")


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
