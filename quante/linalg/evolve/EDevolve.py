# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 18:34:50
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 18:44:07

import numpy as _np
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 类型检查时，导入 torch
    import torch as _tc

__all__ = ['get_time_evolution_states_ED']

def Uinvpsi(pkg, eigenstates, initial_state, herm):
    # U† |psi>
    if herm:
        if eigenstates.dtype == pkg.float64 and initial_state.dtype == pkg.complex128:
            # 直接分别计算实部和虚部，避免构造复数再分解
            udagger_psi = eigenstates.T @ initial_state.real + 1j * (eigenstates.T @ initial_state.imag)
        else:
            udagger_psi = eigenstates.T.conj() @ initial_state
    else:
        udagger_psi = pkg.linalg.solve(eigenstates, initial_state)
    return udagger_psi.reshape(1,-1)

def Uexp(pkg, eigenvalues, eigenstates, times, udagger_psi, scale, shift=False):
    # Ensure correct dtype for broadcasting and computation
    if hasattr(pkg, "ndarray"):  # numpy
        times = pkg.asarray(times)
    else:  # torch
        times = pkg.asarray(times)
        times = times if times.device == eigenvalues.device else times.to(eigenvalues.device)

    # Broadcasting for time evolution
    times_E = times.reshape(-1, 1) * eigenvalues.reshape(1, -1)
    if eigenstates.dtype == udagger_psi.dtype == pkg.float64 and scale == -1j:
        # Real-time evolution: exp(-i E t)
        # exp(-i E t) = cos(E t) - i sin(E t)
        real_part = pkg.cos(times_E) * udagger_psi
        imag_part = pkg.sin(times_E) * udagger_psi
        res = eigenstates @ real_part.T - 1j * (eigenstates @ imag_part.T)
    else:
        if shift:
            scaled_times_E = scale * times_E
            shift_times_E = pkg.max(pkg.real(scaled_times_E), axis=1).reshape(-1, 1)
            exp_timeE_psi = pkg.exp(scaled_times_E - shift_times_E) * udagger_psi
        else:
            exp_timeE_psi = pkg.exp(scale * times_E) * udagger_psi
        if eigenstates.dtype == pkg.float64 and exp_timeE_psi.dtype == pkg.complex128:
            # 直接分别计算实部和虚部，避免构造复数再分解
            res = eigenstates @ exp_timeE_psi.real.T + 1j * (eigenstates @ exp_timeE_psi.imag.T)
        else:
            # Imaginary-time evolution: exp(E t)
            res = eigenstates @ exp_timeE_psi.T
    return res

# -> CPU
def _in_CPU(
    initial_state: _np.ndarray,
    eigenvalues: _np.ndarray,
    eigenstates: _np.ndarray,
    times:_np.ndarray,
    herm:bool=True,
    scale=-1j, 
    shift=False,
) -> _np.ndarray:
    # 确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    if _np.iscomplexobj(eigenstates) or _np.iscomplexobj(initial_state):
        eigenstates = eigenstates.astype(_np.complex128)
        initial_state = initial_state.astype(_np.complex128)
    udagger_psi = Uinvpsi(_np, eigenstates, initial_state, herm)  # U† |psi>
    # U exp(-iEt) U† |psi>
    return Uexp(_np, eigenvalues, eigenstates, times, udagger_psi, scale, shift=shift)

# -> GPU
def _in_GPU(
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
    from ...torch_utils import totc

    # 将数据从 numpy 数组转换为 GPU 上的 torch.Tensor。
    initial_state = totc(initial_state, device=device)
    eigenvalues = totc(eigenvalues, device=device)
    eigenstates = totc(eigenstates, device=device)
    times = totc(times, device=device)

    # 确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    if eigenstates.dtype == _tc.complex128 or initial_state.dtype == _tc.complex128:
        eigenstates = eigenstates.to(_tc.complex128)
        initial_state = initial_state.to(_tc.complex128)
 
    udagger_psi = Uinvpsi(_tc, eigenstates, initial_state, herm)  # U† |psi>
    return Uexp(_tc, eigenvalues, eigenstates, times, udagger_psi, scale, shift=shift) 
    # 将结果从 GPU 转回 CPU，并转换为 numpy 数组

def get_time_evolution_states_ED(
    initial_state: _np.ndarray,
    eigenvalues: _np.ndarray,
    eigenstates: _np.ndarray,
    times: _np.ndarray, 
    *, 
    failback_to_CPU: bool = False, 
    device_name='cpu',
    herm=True,
    ttype='real-time'
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
    scale = -1j if ttype=='real-time' else 1.
    try:
        import torch as tc
        device = tc.device(device_name)
        time_states = _in_GPU(initial_state, eigenvalues, eigenstates, times, device, herm, scale).cpu().numpy()
    except Exception as e:
        if not failback_to_CPU:
            raise e
        time_states = _in_CPU(initial_state, eigenvalues, eigenstates, times, herm, scale)
    return time_states

