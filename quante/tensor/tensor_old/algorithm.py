# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2023-09-23 15:55:53
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 09:50:55
import scipy as _scipy
import numpy as _np
import scipy.sparse as _sparse
import scipy.sparse.linalg as _sla
import numpy.linalg as _nla
from ...linalg.decomp import svd_truncate, eigh
from .mpo import MPO, add_mpo
from .mps import MPS, canonical_form_mps
from ...linalg.operations import expm, logm, kron
from ...generate.matrix import pauli_matrix
from ...linalg.operations import kron


__all__ = ["get_tensorT"]

__all__ += ["TNE_get_init_env_psi", "TNE_single_update"]

__all__ += ["TNT_get_free_energy_pre_site"]

__all__ += ["QES_get_entanglement_bath_hamiltonian", "QES_get_quantum_entanglement_simulator", "QES_get_EBH_coefficient"]

__all__ += ["TEBD_single_unitary_evolve_DM"]

__all__ += ["dissipative_boundary_driven"]

__all__ += ["canonicalize", "shift_orthogonal_center"]


__all__ += ["mpo_apply_mps"]

# ====================================
#  Get single inequivalent tensor T
# ====================================

def _get_evolution_operator(dt:float, localH:_np.ndarray) -> _np.ndarray:
    return expm(-dt*localH)

def _get_TaTb(U_dt:_np.ndarray, d2:int) -> _np.ndarray:
    """# todo 画图
    """
    d = d2 // 2
    Es, u = eigh(U_dt.reshape(d,d,d,d).transpose(0,2,1,3).reshape(d2, d2))
    Es_sqrt:_np.ndarray = _np.sqrt(Es.astype(complex))
    Ta = u * Es_sqrt
    Tb = Es_sqrt.reshape(-1, 1) * u.T.conj()
    return Ta, Tb

def _make_TaTb_Hermitian(Ta:_np.ndarray, Tb:_np.ndarray, d:int) -> tuple[_np.ndarray, _np.ndarray]:
    print("Ta:"); print(Ta)
    print("Tb: "); print(Tb)
    Ta[:,0] = Ta[:,0]*1j  
    Tb[0,:] = Tb[0,:]*-1j
    
    Ta[:,1] = Ta[:,1]*1j
    Tb[1,:] = Tb[1,:]*-1j
    
    Ta[:,2] = Ta[:,2]*1j
    Tb[2,:] = Tb[2,:]*-1j
    print("Ta:"); print(Ta)
    print("Tb: "); print(Tb)
    return Ta.reshape(d,d,-1), Tb.reshape(-1,d,d)

def _check_TaTb_and_U_dt(U_dt:_np.ndarray, Ta:_np.ndarray, Tb:_np.ndarray, d2:int) -> None:
    U_dt_ = _np.einsum("abc,cde->adbe", Ta, Tb, optimize=True).reshape(d2, -1)
    assert _np.allclose(U_dt_, U_dt), "Ta, Tb need operation."
    pass

def _contract_U_TaTb_U(U_dt_half:_np.ndarray, Ta:_np.ndarray, Tb:_np.ndarray, d2:int) -> _np.ndarray:
    """# todo 画图
    """
    d = d2 // 2
    U_dt_half = U_dt_half.reshape(d, d, d, d)
    T:_np.ndarray = _np.einsum("acbd,fbe,dgh,egij->facijh", U_dt_half, Tb, Ta, U_dt_half, optimize=True)
    return T.reshape(d2, d2, d2, d2)

def _check_T_Hermitian(T:_np.ndarray) -> None:
    assert _nla.norm(T - T.transpose(0, 2, 1, 3).conj()), "T is NOT Hermitian."

def get_tensorT(dt:float, localH:_np.ndarray) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """将演化门张量网络变为正方张量网络，仅包含单不等价四阶张量 `T` 

    Args:
        dt (float): 演化步长
        localH (_np.ndarray): 局域哈密顿量
    """
    d2, d2 = localH.shape
    U_dt = _get_evolution_operator(dt, localH)
    Ta, Tb = _get_TaTb(U_dt, d2)
    Ta, Tb = _make_TaTb_Hermitian(Ta, Tb, d2//2)
    _check_TaTb_and_U_dt(U_dt, Ta, Tb, d2)
    U_dt_half = _get_evolution_operator(dt/2, localH)
    T = _contract_U_TaTb_U(U_dt_half, Ta, Tb, d2)
    _check_T_Hermitian(T)    
    return T, Ta, Tb


# ====================================
#  Tensor Network Encoding method
# ====================================

def TNE_get_init_env_psi(D:int, env_d:int, psi_d:int, T_dtype:_np.dtype, env_pre=None, psi_pre=None) -> tuple[_np.ndarray, _np.ndarray]:
    """初始化环境张量`env`和中心张量`psi`. 将具有小虚拟指标维度的环境和中心张量作为输入可以加速收敛速度.

    Args:
        D (int): 环境张量虚拟指标维度 (中心张量虚拟指标)
        env_d (int): 环境张量物理指标维度
        psi_d (int): 中心张量物理指标维度
        T_dtype (np.dtype): 张量网络数据类型
        env_pre (np.ndarray, optional): 具有小虚拟指标维度的环境张量 Defaults to None.
        psi_pre (np.ndarray, optional): 具有小虚拟指标维度的中心张量 Defaults to None.

    Returns:
        tuple[_np.ndarray, _np.ndarray]: 初始化后的环境张量和中心张量
    """
    try:
        D_pre, _, _ = env_pre.shape
        env = _np.zeros((D, D, env_d), dtype=T_dtype)
        psi = _np.zeros((D, psi_d, D), dtype=T_dtype)
        env[:D_pre, :D_pre, :] = env_pre
        psi[:D_pre, :, :D_pre] = psi_pre
    except:
        env = _np.random.rand(D, D, env_d).astype(T_dtype)
        psi = _np.random.rand(D, psi_d, D).astype(T_dtype)
        env /= _nla.norm(env)
        psi /= _nla.norm(psi)
    return env, psi


class _effect_density_matrix(_sparse.linalg.LinearOperator):
    """
    Define env_rho_env by matrix-vector production
    .. code-block:: text
    
        .--0       1      0--.
         |         |         |
        env--2  0--T--3  2--env
         |         |         |
         1         2         1
                   1        
                   |
               0—-psi--2
        
    """
    def __init__(self, T:_np.ndarray, env:_np.ndarray) -> None:
        self.T_ = T
        self.env_ = env
        self.dtype = T.dtype
        chi, chi, d = env.shape
        D0, d,d, D1 = T.shape
        self.shape =[chi*d*chi, chi*d*chi]
        self.chi, self.d, self.D0, self.D1 = chi, d, D0, D1

    def _matvec(self, psi:_np.ndarray):
        chi, d = self.chi, self.d
        x = psi.reshape(chi, d, chi)
        x = _np.tensordot(self.env_, x, axes=(1, 0))
        x = _np.tensordot(x, self.T_, axes=([1, 2], [0, 2]))
        x = _np.tensordot(x, self.env_, axes=([1, 3], [1, 2]))
        x = _np.reshape(x, chi*d*chi)
        return x

def _update_psi(T:_np.ndarray, env:_np.ndarray, psi:_np.ndarray) -> _np.ndarray:
    """
    .. code-block:: text
    
       0          1         0                             
        |         |         |                              
       env--2  0--T--3  2--env    ------>        1        
        |         |         |     Lanczos        |      
        1         2         1                0--psi--2
    """
    D, psi_d, _ = psi.shape
    effect_rho = _effect_density_matrix(T, env)
    _, psi =_sla.eigsh(effect_rho, k=1, which='LM', return_eigenvectors=True, v0=psi)
    return psi.reshape(D, psi_d, D)

def _contract_env_A_T_A(T:_np.ndarray, env:_np.ndarray, A:_np.ndarray) -> _np.ndarray:
    """
    .. code-block:: text
    
        .       0---A---2  
                    |
        1           1
        |           |     
        env--2  0---T---3  
        |           |        
        0          1(2)
                    |
                0---A---2    
    """
    D, _, env_d = env.shape
    _, _, _, psi_d = T.shape

    A = A.reshape(D, env_d, D)
    Aup_mat = A.conj().transpose(0, 2, 1).reshape(-1, env_d)
    T_mat = T.transpose(1, 0, 2, 3).reshape(env_d, -1)

    AT_mat = (Aup_mat @ T_mat).reshape(D, D, env_d, psi_d, env_d)
    AT_mat = AT_mat.transpose(0, 2, 1, 4, 3).reshape(-1, psi_d)
    
    Adown_mat = A.transpose(1, 0, 2).reshape(psi_d, -1)
    ATA_mat = (AT_mat @ Adown_mat).reshape(D, env_d, D, env_d, D, D)
    ATA_mat = ATA_mat.transpose(0, 4, 1, 2, 5, 3).reshape(D * env_d * D, -1)

    env_vec = env.reshape(-1, D * D * env_d)
    envATA = env_vec @ ATA_mat
    return envATA.reshape(D, D, env_d)

def _update_env(T:_np.ndarray, env:_np.ndarray, psi:_np.ndarray) -> tuple[_np.ndarray, float]:
    D, _, _ = env.shape
    A, _ = _nla.qr(psi.reshape(-1, D))
    envATA = _contract_env_A_T_A(T, env, A)
    
    envATA = (envATA + envATA.transpose(1, 0, 2).conj())/2
    norm = _nla.norm(envATA)
    env = envATA / norm
    
    return env, norm

def TNE_single_update(T:_np.ndarray, env:_np.ndarray, psi:_np.ndarray) -> tuple[float, _np.ndarray, _np.ndarray]:
    """张量网络编码算法的一次更新，包括更新中心张量和更新环境张量

    Args:
        T (np.ndarray): 单不等价四阶张量
        env (np.ndarray): 环境张量
        psi (np.ndarray): 中心张量
    """
    psi = _update_psi(T, env, psi)
    env, norm = _update_env(T, env, psi)
    return norm, env, psi


# ===================================
#  Tensor Network Tailoring method
# ===================================

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # 类型检查时，导入 torch
    import torch as _tc

def TNT_get_free_energy_pre_site(T:'_tc.Tensor', envL:'_tc.Tensor', beta:float, dt:float) -> '_tc.Tensor':
    import torch as _tc
    """
    计算平均格点自由能。

    Args:
        T (tc.Tensor): 中心张量
        envL (tc.Tensor): 左环境张量
        beta (float): 倒温度
        dt (float): 演化步长

    Returns:
        tc.Tensor: 平均格点自由能
    
    
    .. code-block:: text

          0         1         0
          |         |         |
        envL--2  0--T--3  2--envR  = 𝜌(t) = exp(-𝜏H) 
          |         |         |
          1         2         1 
     
    freeE/N = -lnZ/βN 
            = -lnλ^N / βN 
            = -lnλ/β = -1/β * ln(⟨L|𝜌|R⟩/⟨L|R⟩) 
            = (ln⟨L|𝜌|R⟩ - ln⟨L|R⟩) / (-β)
    """
    k = beta / dt
    D, _, env_d = envL.shape
    _, psi_d, _, _ = T.shape

    # * Hermitaing
    envL = (envL + envL.permute(1, 0, 2).conj())/2 
    # envR = (envR + envR.permute(1, 0, 2).conj())/2

    # * ⟨L|𝜌|R⟩ 
    tem = envL.reshape(-1, env_d) @ T.reshape(env_d, -1)
    tem = tem.reshape(-1, env_d) @ envL.conj().reshape(-1, env_d).t()
    effect_rho = tem.reshape(D, D, psi_d, psi_d, D, D).permute(0, 2, 4, 1, 3, 5)
    effect_rho = effect_rho.reshape(D*psi_d*D, -1)
    assert _tc.allclose(effect_rho, effect_rho.t().conj()), "Effective density matrix is not Hermitian"
    Es = _tc.linalg.eigvalsh(effect_rho)
    log_LρR = k * _tc.log(Es[-1]) + _tc.log(_tc.sum((Es / Es[-1])**k))

    # * ⟨L|R⟩
    norm_rho = _tc.einsum("abc,dec->adbe", envL, envL.conj()).reshape(D * D, -1)
    norm_Es = _tc.linalg.eigvalsh(norm_rho)
    log_LR = k*_tc.log(norm_Es[-1]) + _tc.log(_tc.sum((norm_Es / norm_Es[-1])**k))

    # * lnZ
    lnZ = log_LρR - log_LR

    # * -1/β * lnZ
    freeE = lnZ / -(2 * beta)
    return freeE


# =================================
#   Entanglement Bath Hamiltonian
# =================================
def QES_get_entanglement_bath_hamiltonian(env:_np.ndarray, Ta:_np.ndarray, Tb:_np.ndarray, dt:float) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Output: HL, HR
    """
    d, _, env_d = Ta.shape
    D, _, env_d = env.shape
    
    # * exp(-dt*HL), exp(-dt*HR)
    rhoL = _np.einsum("abc,cde->adbe", env, Tb, optimize=True).reshape(D*d, -1)
    rhoR = _np.einsum("abc,dec->adbe", Ta, env.conj(), optimize=True).reshape(d*D, -1)
    rhoL /= _np.trace(rhoL)
    rhoR /= _np.trace(rhoR)

    # * HL, HR
    HL = logm(rhoL)
    HR = logm(rhoR)
    HL = (HL + HL.conj().transpose())/(-2*dt)
    HR = (HR + HR.conj().transpose())/(-2*dt)

    # * trace(HL) = trace(HR) = 0
    HL_d, HL_d = HL.shape
    HL:_np.ndarray = HL - _np.trace(HL)/HL_d*_np.eye(HL_d)
    HR:_np.ndarray = HR - _np.trace(HR)/HL_d*_np.eye(HL_d)

    return HL.reshape(D, d, D, d), HR.reshape(d, D, d, D)


def QES_get_quantum_entanglement_simulator(HL:_np.ndarray, HB:_np.ndarray, HR:_np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
    """构造量子纠缠模拟器哈密顿量

    Args:
        HL (np.ndarray): 左纠缠浴哈密顿量
        HB (np.ndarray): 体哈密顿量
        HR (np.ndarray): 右纠缠浴哈密顿量

    Returns:
        tuple[_np.ndarray, _np.ndarray]: 本征值, 本征态
    """
    bulk_d, _ = HB.shape
    D, d, _, _ = HL.shape
    bulk_Id = _np.eye(bulk_d//d, dtype=HB.dtype)
    env_Id  = _np.eye(D, dtype=HB.dtype)
    Hqes = kron(HL.reshape(D*d, -1), bulk_Id, env_Id) + kron(env_Id, HB, env_Id) + kron(env_Id, bulk_Id, HR.reshape(d*D, -1))  
    assert _np.allclose(Hqes, Hqes.T.conj()), "Hqes is NOT Hermitian."
    eigenvalues, eigenstates = eigh(Hqes)
    return eigenvalues, eigenstates


def QES_get_EBH_coefficient(HL:_np.ndarray, HR:_np.ndarray, chi:int, pauli=False) -> tuple[_np.ndarray, _np.ndarray]:
    """
    Output: CL, CR
    """
    d, d = HL.shape
    assert d**0.5==chi
    # * get coefficients
    CL, CR = _np.zeros((d, d), dtype=_np.complex128), _np.zeros((d, d), dtype=_np.complex128)
    spin_basis = ["I", "X", "Y", "Z"] if pauli else ["I", "x", "y", "z"]
    for i, basis1 in enumerate(spin_basis):
        for j, basis2 in enumerate(spin_basis):
            matrix_basis = pauli_matrix(basis1+basis2)
            matrix_basis = matrix_basis.astype(HL.dtype)
            norm = _nla.norm(matrix_basis)
            matrix_basis /= norm
            # * coefficient = trace / norm
            CL[i, j] = _np.trace(HL @ matrix_basis) / norm
            CR[i, j] = _np.trace(HR @ matrix_basis) / norm
    return CL, CR



# =====================================
# Density Matrix Renormalization Group
# =====================================



# =============================================================
#  Time Evolve Block Decomposition method for Density Matrix
# =============================================================

def _apply_single_gate(rho:MPO, i:int, gate:_np.ndarray) ->None:
    """
    .. code-block:: text

        .       (x)
                 │
                gate
                 |
                (b)
                 |   
        ---(a)---Wi---(d)---  
                 |
                (c)
                 |
    """
    # einsum("xb,abcd->axcd", gate, rho.Ws[i])
    Wsi = gate @ rho.Ws[i].transpose(1, 0, 2, 3)
    rho.Ws[i] = Wsi.transpose(1, 0, 2, 3)


def _one_site_nonunitary_evolve_DM(rho:MPO, i:int, gate:_np.ndarray) ->None:
    """
    """
    current_llim, current_rlim = rho.llim, rho.rlim
    target_llim, target_rlim = i, i
    rho.Ws = shift_orthogonal_center(rho.Ws, current_llim, current_rlim, target_llim, target_rlim)
    _apply_single_gate(rho, i, gate)
    rho.llim, rho.rlim = target_llim, target_rlim


def _two_site_nonunitary_evolve_DM(rho:MPO, i:int, gate:_np.ndarray, Dc:int, eps:float) ->None:
    pass


def _one_site_unitary_evolve_DM(rho:MPO, i:int, gate:_np.ndarray, Dc:int, eps:float) ->None:
    pass


def _two_site_unitary_evolve_DM(rho:MPO, i:int, gate:_np.ndarray, Dc:int, eps:float) ->None:
    r"""
    .. code-block:: text

        .            c    f
                     |    |
        a--S--b   b--Wi---Wj--h 
                     |    |
                     d    g
    """
    j = (i+1) % rho.L
    WW = rho.two_site_WW(i)
    b, c, d, f, g, h = WW.shape

    # * up time evolve
    WW_mat = WW.transpose(1, 3, 0, 2, 4, 5).reshape(c*f, -1)
    gate_WW = (gate @ WW_mat).reshape(c, f, b, d, g, h)
    gate_WW = gate_WW.transpose(2, 0, 3, 1, 4, 5)
    
    # * down time evolve"""
    gate_WW_mat = gate_WW.transpose(0, 1, 3, 5, 2, 4).reshape(-1, d*g)
    gate_mat = gate.conj().T
    gate_WW_gate = (gate_WW_mat @ gate_mat).reshape(b, c, f, h, d, g)
    gate_WW_gate = gate_WW_gate.transpose(0, 1, 4, 2, 5, 3)

    # * truncate
    gate_SWW_gate = rho.Ss[i].reshape(-1, 1) * gate_WW_gate.reshape(b, -1)
    gate_SWW_gate = gate_SWW_gate.reshape(b*c*d, -1)
    Ai, Sj, Wj = svd_truncate(gate_SWW_gate, Dc, eps)
    Sj /= _nla.norm(Sj)
    Ai = Ai.reshape(b, c, d, -1)
    Wj = Wj.reshape(-1, f, g, h)

    # * update the mpo class
    # \tb, \cc, \rc, \tg, \dia
    # Wi = _np.einsum("bcdfgh,xfgh->bcdx", gate_WW_gate, Wj.conj())
    Wi = gate_WW_gate.reshape(b*c*d, -1) @ Wj.conj().reshape(-1, f*g*h).T
    rho.Ws[i] = Wi.reshape(b, c, d, -1)
    rho.Ss[j] = Sj
    rho.Ws[j] = Wj


def TEBD_single_unitary_evolve_DM(
    rho: MPO, 
    position: list[_np.ndarray], 
    time_evolve_gates: list[_np.ndarray], 
    Dc: int, 
    eps: float, 
    Hemitring: bool = False) -> None:
    """
    Perform single TIME evolution using the TEBD (Time Evolving Block Decimation) algorithm.

    Args:
        rho (MPO): The MPO representing the Density Matrix.
        position (list[_np.ndarray]): List of positions where the gates are applied.
        time_evolve_gates (list[_np.ndarray]): List of time evolution gates for each position.
        Dc (int): Bond dimension for truncation.
        eps (float): Truncation threshold.
        Hemitring (bool, optional): Whether to apply a Hermitian ring projection at the end.
    """
    for n, site in enumerate(position):
        _two_site_unitary_evolve_DM(rho, site, time_evolve_gates[n], Dc, eps)

    if Hemitring:
        Ws = add_mpo(rho.Ws, rho.get_Ws_dagger(), alpha=0.5, beta=0.5)  # Apply Hermitian averaging
        rho = canonical_form_mpo(Ws, Dc, eps)  # Canonicalize the MPO after Hermitian averaging
        
# =====================================
#         MPO apply on MPS
# =====================================
def _apply_one_site(mpo_W:_np.ndarray, mps_W:_np.ndarray) -> _np.ndarray:
    a, b, c, d = mpo_W.shape
    e, c, f = mps_W.shape
    return _np.einsum("abcd,ecf->aebdf", mpo_W, mps_W, optimize=True).reshape(a*e, b, d*f)


def mpo_apply_mps(mpo_Ws, mps_Ws, Dc, eps):
    assert len(mpo_Ws) == len(mps_Ws), "The length of MPO and MPS should be the same."
    for i in range(len(mpo_Ws)):
        mps_Ws[i] = _apply_one_site(mpo_Ws[i], mps_Ws[i])
    return canonical_form_mps(mps_Ws, Dc, eps)




# =============
#  Disspative 
# =============

def dissipative_uniform(rho:MPO, diss_operators:list[_np.ndarray]) -> MPO:
    L = rho.L
    d = int(diss_operators[0].shape[1]**0.5)
    # r"""
    #   x   y      b 
    #    \ /       |
    #     D   D0---W---D1
    #    / \       |
    #   b   c      c
    # """
    for i in range(L):
        # rho.Bs[i] = _np.einsum("xybc,abcd->axyd", diss_operators[i].reshape(d, d, d, d), rho.Bs[i])
        D0, D1 = rho.Bs[i].shape[0], rho.Bs[i].shape[3]
        B_mat = rho.Bs[i].transpose(1, 2, 0, 3).reshape(d*d, D0*D1)
        diss_B_mat = diss_operators[i] @ B_mat
        rho.Bs[i] = diss_B_mat.reshape(d, d, D0, D1).transpose(2, 0, 1, 3)


def dissipative_boundary_driven(rho:MPO, left:_np.ndarray, right:_np.ndarray, Dc:int, eps:float) -> None:
    """
    Apply dissipative boundary-driven gates to the MPO's leftmost and rightmost tensors
    
    Args:
        rho (MPO): Density matrix
        left (_np.ndarray): Left dissipative operator
        right (_np.ndarray): Right dissipative operator
    """
    def apply_boundary_operator(W, diss_gate:_np.ndarray) -> _np.ndarray:
        r"""
        .. code-block:: text
        
            x   y       b 
             \ /        |
              D    D0---W---D1
             / \        |
            b   c       c
        updated_tensor = _np.einsum("abcd,xcdy->xaby", diss_gate.reshape(b, b, b, b), W)
        abxy -> 0123, xaby->2013
        """
        a, b, c, d = W.shape
        updated_tensor = (diss_gate @ W.transpose(1,2,0,3).reshape(b*c, -1)).reshape(b,c,a,d)
        return updated_tensor.transpose(2,0,1,3)

    rho.Ws[0] = apply_boundary_operator(rho.Ws[0], left)
    rho.Ws[-1] = apply_boundary_operator(rho.Ws[-1], right)
    rho.Ss, rho.Ws, rho.llim, rho.rlim = canonicalize(rho.Ws, Dc, eps) 



# ==========================
#   MPO and MPS operations   
# ==========================

def _QR(W:_np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
    """
    .. code-block:: text
        
        .      |                           |
              (b)                         (b)
               |            QR             |
        --(a)--⬜--(c)--    ---->    --(a)--▷--(d)--⬜--(c)--
               :                           :       :
               W                           A       S
    """
    *bonds, c = W.shape
    A, S = _nla.qr(W.reshape(-1, c))
    return A.reshape(*bonds, c), S


def _LU(W:_np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
    """
    .. code-block:: text
    
        .              |                       |         
                      (b)                     (b)        
                       |           QR          |         
        --(a)--⬜--(d)--⨞--(c)--   <---  --(a)--⬜--(c)--    
                       :                       :
               S       A                       W
    """
    a, *bonds = W.shape
    A, S = _nla.qr(W.reshape(a, -1).T)
    return S.T, A.T.reshape(a, *bonds)


def shift_orthogonal_center(Ws, current_llim, current_rlim, target_llim, target_rlim) -> tuple[_np.ndarray, _np.ndarray]:
    """
    .. code-block:: text
    
        cur_llim --> <-- cur_rlim       tar_llim --> <-- tar_rlim
            |       |       |       |       |       |       |
        ----▷-------⬜-------⨞-------⨞--------⨞-------⬜-------⨞-----
            :       :       :       :       :       :       :
           Ws[0]   Ws[1]   Ws[2]   Ws[3]   Ws[4]   Ws[5]   Ws[6]
    """
    
    for i in range(current_llim, target_llim, 1):
        A, S = _QR(Ws[i])
        Ws[i] = A
        a, *bonds = Ws[i+1].shape
        Ws[i+1] = (S @ Ws[i+1].reshape(a, -1)).reshape(a, *bonds)
        
    for i in range(current_rlim, target_rlim, -1):
        S, A = _LU(Ws[i])
        Ws[i] = A
        *bonds, c = Ws[i-1].shape
        Ws[i-1] = (Ws[i-1].reshape(-1, c) @ S).reshape(*bonds, c)

    return Ws


def _left2right_QR(Ws) -> list[_np.ndarray]:
    L = len(Ws)
    for i in range(L - 1):
        A, S = _QR(Ws[i])
        Ws[i] = A
        a, *bonds = Ws[i+1].shape
        Ws[i+1] = (S @ Ws[i+1].reshape(a, -1)).reshape(a, *bonds)
    return Ws


def _right2left_SVD_step(W:_np.ndarray, Dc:int, eps:float) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
    .. code-block:: text
    
        .                      |                       |
                              (b)                     (b)
                               |          SVD          |
        --(a)--▷--(d)--◇--(e)--⨞--(c)--  <----  --(a)--⬜--(c)--
                               :                       : 
               U       S       B                       W
    """
    a, *bonds = W.shape
    W = W.reshape(a, -1)
    u, s, v = svd_truncate(W, Dc=Dc, eps=eps)
    return u.reshape(a, -1), s, v.reshape(-1, *bonds)


def _SVD_constract_right(W:_np.ndarray, u:_np.ndarray, s:_np.ndarray)->_np.ndarray:
    """
    .. code-block:: text
    
        .      |                                      |
              (b)                                    (b)
               |                                      |
        --(a)--▷--(c)--▷--(d)--◇--(e)  ---->   --(a)--⬜--(e)--
               :                                      :  
               W       u       s                      W

    >>> tc.einsum("abc,cd,de->abe", A, U, S)
    """
    *bonds, c = W.shape
    W = (W.reshape(-1, c) @ u) * s.reshape(1, -1)
    return W.reshape(*bonds, -1)


def _right2left_SVD(Ws:list[_np.ndarray], Dc:int, eps:float) -> tuple[list[_np.ndarray], list[_np.ndarray]]:
    L = len(Ws)
    Ss = [None] * (L + 1)
    for i in range(L-1, 0, -1):
        u, Ss[i], Ws[i] = _right2left_SVD_step(Ws[i], Dc, eps)
        Ws[i-1] = _SVD_constract_right(Ws[i-1], u, Ss[i])
    a = Ws[0]
    u, Ss[0], Ws[0] = _right2left_SVD_step(Ws[0], Dc, eps)
    return Ss, Ws


def canonicalize(Ws:list[_np.ndarray], Dc:int = None, eps=None) -> tuple[list[_np.ndarray],list[_np.ndarray]]:
    assert Ws[0].shape[0] == Ws[-1].shape[-1] == 1, "正则形式只对开边界mps有定义！"
    L = len(Ws)
    Ws = _left2right_QR(Ws)
    Ss, Ws = _right2left_SVD(Ws, Dc, eps)
    Ss[0] = Ss[-1] = _np.array([1.0])
    llim, rlim = 0, 0
    return Ss, Ws, llim, rlim
    