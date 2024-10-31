# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-26 17:09:16
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 19:21:24

# todo: 实现 Arnoldi method 对角化非厄密矩阵；LanczosEvolution 计算 :math:`exp(delta H) |psi0>`

import numpy as np
import scipy.sparse.linalg as spalg
from typing import Callable

__all__ = ["lanczos_ground_state", "lanczos_evolve_state", "lanczos_arpack"]


def lanczos_ground_state(matvec:Callable[[np.ndarray], np.ndarray], psi0:np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
    """
    从初始猜测 `|psi0>` 迭代地构建 Krylov 空间的正交基计算基态：
    
    `|psi0>`, `H|psi0>`, `H^2|psi0>`, ... `H^N |psi0>`
    
    这一组向量构成 Krylov 空间，将 `H` 投影到其中并求解，得到 "Ritz" 特征值/特征向量。最后，可以使用基将解转换回原始空间。
    
    一个重要策略是在若干步之后（隐式地）重新启动算法。这里**不**进行这种操作：当我们使用这些类时，通常有一个显式的外部循环，直到收敛，例如 DMRG 中的 "sweeps"。
    
    # todo，如果 psi0 是 list[ndarray] 的结构，如何实现？
    
    示例:
    --------
    >>> dim = 10000
    >>> H = np.random.randn(dim,dim)
    >>> psi0 = np.random.randn(dim)
    >>> E0, vec = qt.linalg.lanczos_ground_state(lambda b: H @ b, psi0)
    
    备注
    -----
    Ritz 残差 `RitzRes` 根据
    http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual 计算。
    给定间隙，Ritz 残差给出了波函数误差的界限，
    ``err < (RitzRes/gap)**2``。间隙是从完整的 Lanczos 谱估计的。
    """
    paras = {
        "N_min": 2,  #  要执行的最小步数
        "N_max": 20,  # 要执行的最大步数
        "P_tol": 1.e-14,  # 来自 Ritz 残差的误差估计的容差
        "min_gap": 1.e-12,  # 用于 P_tol 标准的间隙估计的下限
        "reortho": False,  #  是否重新正交化
        "cutoff": np.finfo(psi0.dtype if not isinstance(psi0, list) else psi0[0].dtype).eps * 100,  #  如果新 Krylov 向量的范数太小，则中止的截止值
        "E_tol": np.inf  #  本征值误差容差
    }
    paras.update(kwargs)
    eng, vec, N = _lanczos_ground_state(matvec, psi0, **paras)
    # N 是迭代次数
    return eng, vec


def _lanczos_ground_state(matvec, psi0, N_min, N_max, P_tol, min_gap, reortho, cutoff, E_tol):
    bases = []  # 用 list 因为不确定会用几个本征态
    Es = np.zeros((N_max, N_max), dtype=np.float64)
    h = np.zeros((N_max + 1, N_max + 1), dtype=np.float64)
    
    # 构建 Krylov 空间
    beta = np.linalg.norm(psi0)
    
    if beta < cutoff:
        raise ValueError(f'Norm of self.psi0 too small: {beta}')

    # 因为要反复用，所以把这两个取出来，不用每次从 self 里面找
    w = psi0.copy()
    
    for k in range(N_max):
        # 计算矩阵元
        w /= beta
        bases.append(w)
        w = matvec(w)
        alpha = np.real(w.conj() @ bases[-1])
        h[k, k] = alpha
        w -= alpha * bases[-1]
        
        # 本征求解
        if k == 0:
            Es[0,0] = h[0,0]
            eigenvector = np.ones(1, np.float64)
        else:
            eng, vec = np.linalg.eigh(h[:k+1, :k+1])
            Es[k, :k+1] = eng  # 保存本征值
            eigenvector = vec[:,0]  # 保存最小值对应的本征向量
        
        # 构建下一个基矢和矩阵元
        if reortho:
            for b in bases[:-1]:
                w -= (w.conj() @ b) * b
        elif k > 0:
            w -= beta * bases[-2]
        beta = np.linalg.norm(w)
        h[k, k + 1] = h[k + 1, k] = beta
        
        # 判断是否停止
        if abs(beta) < cutoff:
            break
        
        if k + 1 < N_min:
            continue
        
        Es_k = Es[k, :]  # current energies
        RitzRes = abs(eigenvector[k]) * h[k, k + 1]
        gap = max(Es_k[1] - Es_k[0], min_gap)
        P_err = (RitzRes / gap)**2
        Delta_E0 = Es[k - 1, 0] - Es_k[0]
        
        if P_err < P_tol and Delta_E0 < E_tol:
            break
    
    E0 = Es[k, 0]
    
    if k == 0:
        return E0, psi0.copy()/np.linalg.norm(psi0), k + 1  # no better estimate available
    
    res = np.array(bases).T @ eigenvector
    resnorm = np.linalg.norm(res)
    if abs(1. - resnorm) > 1.e-5:
        print(f"Krylov 正交性不能保证，尝试设置 reortho = True")
    
    return E0, res/resnorm, k + 1


    
def lanczos_evolve_state(matvec:Callable[[np.ndarray], np.ndarray], psi0:np.ndarray, delta:np.complexfloating, **kwargs) -> np.ndarray:
    """
    从初始猜测 `|psi0>` 迭代地构建 Krylov 空间的正交基计算 `exp(delta H) |psi0>`：
    
    `|psi0>`, `H|psi0>`, `H^2|psi0>`, ... `H^N |psi0>`
    
    这一组向量构成 Krylov 空间，将 `H` 投影到其中得到矩阵三对角 `h`
    
    此时： `exp(delta h) e_0` 就对应 `exp(delta H) |psi0>`
    
    其中 `e_0 = (1, 0, 0, ...)`
    
    示例:
    --------
    >>> dim = 10000
    >>> H = np.random.randn(dim,dim)
    >>> psi0 = np.random.randn(dim)
    >>> E0, vec = qt.linalg.lanczos_ground_state(lambda b: H @ b, psi0)
    
    """
    paras = {
        "N_min": 2,  #  要执行的最小步数
        "N_max": 20,  # 要执行的最大步数
        "P_tol": 1.e-14,  # 来自 Ritz 残差的误差估计的容差
        "reortho": False,  #  是否重新正交化
        "cutoff": np.finfo(psi0.dtype if not isinstance(psi0, list) else psi0[0].dtype).eps * 100,  #  如果新 Krylov 向量的范数太小，则中止的截止值
        "normalize": None,  # 是否归一，默认为 `np.real(delta) == 0`
    }
    paras.update(kwargs)
    vec, N = _lanczos_evolve_state(matvec, psi0, delta, **paras)
    # N 是迭代次数
    return vec
    
    
def _lanczos_evolve_state(matvec, psi0, delta, N_min, N_max, P_tol, reortho, cutoff, normalize):
    bases = []
    h = np.zeros([N_max + 1, N_max + 1], dtype=np.float64)
    
    # 构建 Krylov 空间
    beta = np.linalg.norm(psi0)
    
    if beta < cutoff:
        raise ValueError(f'Norm of self.psi0 too small: {beta}')

    w = psi0.copy()
    
    for k in range(N_max):
        # 计算矩阵元
        w /= beta
        bases.append(w)
        w = matvec(w)
        
        alpha = np.real(w.conj() @ bases[-1])
        h[k, k] = alpha
        w -= alpha * bases[-1]
        
        # 本征求解
        if k == 0:
            E = h[0,0]
            exp_dE = np.exp(delta * E)
            exp_dh_e0_norm = np.abs(exp_dE)
            exp_dh_e0 = np.array([exp_dE / exp_dh_e0_norm])
        else:
            eng, vec = np.linalg.eigh(h[:k+1, :k+1])
            exp_dh_e0 = np.dot(vec, np.exp(eng * delta) * np.conj(vec[0, :]))
            
            exp_dh_e0_norm = np.linalg.norm(exp_dh_e0)
            exp_dh_e0 = exp_dh_e0 / exp_dh_e0_norm

        # 构建下一个基矢和矩阵元
        if reortho:
            for b in bases[:-1]:
                w -= (w.conj() @ b) * b
        elif k > 0:
            w -= beta * bases[-2]
        beta = np.linalg.norm(w)
        h[k, k + 1] = h[k + 1, k] = beta
        
        # 判断是否停止
        if abs(beta) < cutoff:
            break
        
        if k + 1 < N_min:
            continue
        
        if np.abs(exp_dh_e0[k]) < P_tol:
            break
    
    if k == 0:
        exp_dH_v = exp_dh_e0[0] * psi0
    else:
        exp_dH_v = np.dot(np.array(bases).T, exp_dh_e0)
        resnorm = np.linalg.norm(exp_dH_v)
        if abs(1. - resnorm) > 1.e-5:
            print(f"Krylov 正交性不能保证，尝试设置 reortho = True")
        exp_dH_v /= resnorm
        
        
    if normalize is None:
        normalize = np.real(delta) == 0.
    
    if normalize:
        return exp_dH_v, k + 1
    else:
        beta = np.linalg.norm(psi0)
        return (beta * exp_dh_e0_norm) * exp_dH_v, k + 1


def lanczos_arpack(matvec:Callable[[np.ndarray], np.ndarray], psi0:np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
    """使用 `scipy.sparse.linalg.eigsh` 计算基态
    """
    tol = kwargs.get("P_tol", 1e-14)
    ncv = kwargs.get("N_min", None)
    dim = psi0.shape[0]
    lo = spalg.LinearOperator((dim,dim), matvec=matvec)
    Es, Vs = spalg.eigsh(lo, k=1, v0=psi0, which='SA', tol=tol, ncv=ncv)
    return Es[0], Vs[:, 0]
