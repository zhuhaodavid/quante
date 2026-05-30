# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 18:34:50
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-30 00:14:25

import numpy as _np
from typing import TYPE_CHECKING
from scipy import sparse as sps

from ...measure.expect import expect

if TYPE_CHECKING:
    import torch as _tc

__all__ = ["evolve_from_eigensystem", "measure_from_eigensystem"]

def evolve_from_eigensystem(
    initial_state: _np.ndarray,
    eigenvalues: _np.ndarray,
    eigenstates: _np.ndarray,
    times: _np.ndarray, 
    *, 
    failback_to_CPU: bool = False,
    device_name="cpu",
    herm=True,
    ttype="real-time",
) -> _np.ndarray:
    """
    基于严格对角化的时间演化
    
    Args:
        initial_state (_np.ndarray): 初始量子态
        eigenvalues (_np.ndarray): 哈密顿量本征值
        eigenstates (_np.ndarray): 哈密顿量本征态
        times (_np.ndarray): 时间列表
    
    Returns:
        _np.ndarray: 时间演化量子态矩阵
    """
    initial_state = _np.squeeze(initial_state)
    scale = -1j if ttype == "real-time" else 1.
    try:
        import torch as tc

        device = tc.device(device_name)
        time_states = _evolve_from_eigensystem_torch(
            initial_state,
            eigenvalues,
            eigenstates,
            times,
            device,
            herm,
            scale,
        ).cpu().numpy()
    except Exception as e:
        if not failback_to_CPU:
            raise e
        time_states = _evolve_from_eigensystem_numpy(
            initial_state,
            eigenvalues,
            eigenstates,
            times,
            herm,
            scale,
        )
    return time_states


def measure_from_eigensystem(
    initial_state: _np.ndarray,
    eigenvalues: _np.ndarray,
    eigenstates: _np.ndarray,
    times: _np.ndarray,
    measure=None,
    *,
    herm=True,
    scale=-1j,
    normalize: bool = False,
):
    states = _evolve_from_eigensystem_numpy(
        initial_state,
        eigenvalues,
        eigenstates,
        times,
        herm=herm,
        scale=scale,
        shift=normalize,
    )
    if normalize:
        states /= _np.linalg.norm(states, axis=0)
    if measure is None:
        return states.T
    if isinstance(measure, (sps.sparray, sps.spmatrix, list, _np.ndarray)):
        return expect(measure, states, isdm=False).T
    return _np.array([
        measure(t, states[:, i])
        for i, t in enumerate(times)
    ])


def _evolve_from_eigensystem_numpy(
    initial_state: _np.ndarray,
    eigenvalues: _np.ndarray,
    eigenstates: _np.ndarray,
    times: _np.ndarray,
    herm: bool = True,
    scale=-1j, 
    shift=False,
) -> _np.ndarray:
    # 确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    if _np.iscomplexobj(eigenstates) or _np.iscomplexobj(initial_state):
        eigenstates = eigenstates.astype(_np.complex128)
        initial_state = initial_state.astype(_np.complex128)
    projected_state = _project_to_eigenbasis(_np, eigenstates, initial_state, herm)
    return _reconstruct_from_eigenbasis(
        _np,
        eigenvalues,
        eigenstates,
        times,
        projected_state,
        scale,
        shift=shift,
    )


def _evolve_from_eigensystem_torch(
    initial_state: '_tc.Tensor',
    eigenvalues: '_tc.Tensor',
    eigenstates: '_tc.Tensor',
    times: '_tc.Tensor',
    device,
    herm,
    scale=-1j,
    shift=False,
) -> _np.ndarray:
    """
    在 GPU 上计算初始态在不同时刻的时间演化态。
    """
    import torch as _tc
    from ...bridge.torch_utils import totc

    # 将数据从 numpy 数组转换为 GPU 上的 torch.Tensor。
    initial_state = totc(initial_state, device=device)
    eigenvalues = totc(eigenvalues, device=device)
    eigenstates = totc(eigenstates, device=device)
    times = totc(times, device=device)

    # 确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    if eigenstates.dtype == _tc.complex128 or initial_state.dtype == _tc.complex128:
        eigenstates = eigenstates.to(_tc.complex128)
        initial_state = initial_state.to(_tc.complex128)
 
    projected_state = _project_to_eigenbasis(_tc, eigenstates, initial_state, herm)
    return _reconstruct_from_eigenbasis(
        _tc,
        eigenvalues,
        eigenstates,
        times,
        projected_state,
        scale,
        shift=shift,
    )



def _project_to_eigenbasis(pkg, eigenstates, state, herm):
    if herm:
        if eigenstates.dtype == pkg.float64 and state.dtype == pkg.complex128:
            # 直接分别计算实部和虚部，避免构造复数再分解
            projected = eigenstates.T @ state.real + 1j * (eigenstates.T @ state.imag)
        else:
            projected = eigenstates.T.conj() @ state
    else:
        projected = pkg.linalg.solve(eigenstates, state)
    return projected.reshape(1, -1)

def _reconstruct_from_eigenbasis(
    pkg,
    eigenvalues,
    eigenstates,
    times,
    projected_state,
    scale,
    *,
    shift=False,
):
    if hasattr(pkg, "ndarray"):
        times = pkg.asarray(times)
    else:
        times = pkg.asarray(times)
        times = times if times.device == eigenvalues.device else times.to(eigenvalues.device)

    times_E = times.reshape(-1, 1) * eigenvalues.reshape(1, -1)
    if eigenstates.dtype == projected_state.dtype == pkg.float64 and scale == -1j:
        # Real-time evolution: exp(-i E t)
        # exp(-i E t) = cos(E t) - i sin(E t)
        real_part = pkg.cos(times_E) * projected_state
        imag_part = pkg.sin(times_E) * projected_state
        res = eigenstates @ real_part.T - 1j * (eigenstates @ imag_part.T)
    else:
        if shift:
            scaled_times_E = scale * times_E
            shift_times_E = pkg.max(pkg.real(scaled_times_E), axis=1).reshape(-1, 1)
            exp_timeE_psi = pkg.exp(scaled_times_E - shift_times_E) * projected_state
        else:
            exp_timeE_psi = pkg.exp(scale * times_E) * projected_state
        if eigenstates.dtype == pkg.float64 and exp_timeE_psi.dtype == pkg.complex128:
            # 直接分别计算实部和虚部，避免构造复数再分解
            res = eigenstates @ exp_timeE_psi.real.T + 1j * (eigenstates @ exp_timeE_psi.imag.T)
        else:
            # Imaginary-time evolution: exp(E t)
            res = eigenstates @ exp_timeE_psi.T
    return res

