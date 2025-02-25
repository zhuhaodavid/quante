# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-26 17:09:16
# @Last Modified by:   hzhu
# @Last Modified time: 2025-02-24 18:34:40

# todo: 实现 Arnoldi method 对角化非厄密矩阵；LanczosEvolution 计算 :math:`exp(delta H) |psi0>`

import numpy as np
import scipy.sparse.linalg as spalg
from typing import Callable

__all__ = ["lanczos_ground_state", "lanczos_evolve_state", "lanczos_arpack", "tenpy_arnoldi", "arnoldi_ground_state"]

def lanczos_ground_state(matvec:Callable[[np.ndarray], np.ndarray], psi0:np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
    """
    从初始猜测 `|psi0>` 迭代地构建 Krylov 空间的正交基计算基态：
    
    `|psi0>`, `H|psi0>`, `H^2|psi0>`, ... `H^N |psi0>`
    
    这一组向量构成 Krylov 空间，将 `H` 投影到其中并求解，得到 "Ritz" 特征值/特征向量。最后，可以使用基将解转换回原始空间。
    
    一个重要策略是在若干步之后（隐式地）重新启动算法。这里**不**进行这种操作：当我们使用这些类时，通常有一个显式的外部循环，直到收敛，例如 DMRG 中的 "sweeps"。
    
    # todo，如果 psi0 是 list[ndarray] 的结构，如何实现？
    
    Examples
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
    
    Examples
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
    which = kwargs.get("which", 'SA')
    dim = psi0.shape[0]
    lo = spalg.LinearOperator(shape=(dim,dim), matvec=matvec, dtype=psi0.dtype) # type: ignore
    Es, Vs = spalg.eigsh(lo, k=1, v0=psi0, which=which, tol=tol, ncv=ncv)
    # k = 1 if dim < 5 else 3
    # Es, Vs = spalg.eigs(lo, k=k, v0=psi0, which='LM', tol=tol, ncv=ncv)
    # show(Es)
    return Es[0], Vs[:, 0] #+ 1e-6*np.random.randn(dim)


def arnoldi_ground_state(matvec:Callable[[np.ndarray], np.ndarray], psi0:np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
    paras = {
        "N_min": 2,  #  要执行的最小步数
        "N_max": 20,  # 要执行的最大步数
        "P_tol": 1.e-14,  # 来自 Ritz 残差的误差估计的容差
        "min_gap": 1.e-12,  # 用于 P_tol 标准的间隙估计的下限
        "cutoff": np.finfo(psi0.dtype if not isinstance(psi0, list) else psi0[0].dtype).eps * 100,  #  如果新 Krylov 向量的范数太小，则中止的截止值
        "E_tol": np.inf,  #  本征值误差容差
        "which": 'LM',
        "num_ev": 1,
        "E_shift": None
    }
    paras.update(kwargs)
    eng, vec, N = _arnoldi_ground_state(matvec, psi0, **paras)
    # N 是迭代次数
    return eng, vec


def _arnoldi_ground_state(matvec, psi0, N_min, N_max, P_tol, min_gap, cutoff, E_tol, which, num_ev, E_shift):
    Es = np.zeros((N_max, N_max), dtype=np.complex128)
    h = np.zeros((N_max + 1, N_max + 1), dtype=np.complex128)
    basis = []
    w = psi0
    norm = np.linalg.norm(w)
    for k in range(N_max):
        w /= norm
        basis.append(w)
        w = matvec(w)
        for i, v_i in enumerate(basis):
            h[i, k] = ov = v_i.conj() @ w
            w -= ov * v_i
        h[k + 1, k] = norm = np.linalg.norm(w)
        
        if k + 1 < N_min:
            continue

        # self._calc_result_krylov(k)
        if k == 0:
            Es[0, 0] = h[0, 0]
            eigenvector = np.ones([1, 1], np.complex128)
        else:
            eng, vec = np.linalg.eig(h[:k + 1, :k + 1])
            sort = argsort(eng, which)
            Es[k, :k + 1] = eng[sort]  # 保存本征值
            eigenvector = vec[:, sort]  # 保存最小值对应的本征向量

        if norm < cutoff:
            break


        Es_k = Es[k, :]  # current energies
        RitzRes = abs(eigenvector[k, 0]) * h[k + 1, k]
        gap = max(min([np.min(np.abs(Es_k[i+1:] - Es_k[i])) for i in range(num_ev)]), min_gap)
        P_err = (RitzRes / gap)**2
        Delta_E0 = Es[k - 1, 0] - Es_k[0]

        if np.abs(P_err) < P_tol and np.abs(Delta_E0) < E_tol:
            break
    
    N = k + 1
    E0 = Es[N - 1, :num_ev]
    if E_shift is not None:
        E0 -= E_shift
    if N == 1:
        return E0, [psi0.copy()], N

    psis = []
    for i in range(min(N, num_ev)):
        vf = eigenvector[:, i]
        vf = np.real_if_close(vf)
        assert N == len(vf) > 1
        assert len(basis) >= N
        
        if isinstance(psi0, list):
            psi = [p * vf[0] for p in basis[0]]
        else:
            psi = vf[0] * basis[0]

        for k in range(1, N):
            psi += vf[k] * basis[k]
        
        psi_norm = np.linalg.norm(psi)
        
        if abs(1. - psi_norm) > 1.e-5:
            print(f"poorly conditioned H matrix in Arnoldi! |psi| = {psi_norm:.2e}")
        
        psi /= psi_norm
        psis.append(psi)
        
    return E0, psis, N


def argsort(a, sort=None, **kwargs):
    """wrapper around np.argsort to allow sorting ascending/descending and by magnitude.

    Parameters
    ----------
    a : array_like
        The array to sort.
    sort : ``'m>', 'm<', '>', '<', None``
        Specify how the arguments should be sorted.

        ==================== =============================
        `sort`               order
        ==================== =============================
        ``'m>', 'LM'``       Largest magnitude first
        -------------------- -----------------------------
        ``'m<', 'SM'``       Smallest magnitude first
        -------------------- -----------------------------
        ``'>', 'LR', 'LA'``  Largest real part first
        -------------------- -----------------------------
        ``'<', 'SR', 'SA'``  Smallest real part first
        -------------------- -----------------------------
        ``'LI'``             Largest imaginary part first
        -------------------- -----------------------------
        ``'SI'``             Smallest imaginary part first
        -------------------- -----------------------------
        ``None``             numpy default: same as '<'
        ==================== =============================

    **kwargs :
        Further keyword arguments given directly to :func:`numpy.argsort`.

    Returns
    -------
    index_array : ndarray, int
        Same shape as `a`, such that ``a[index_array]`` is sorted in the specified way.
    """
    if sort is not None:
        if sort == 'm<' or sort == 'SM':
            a = np.abs(a)
        elif sort == 'm>' or sort == 'LM':
            a = -np.abs(a)
        elif sort == '<' or sort == 'SR' or sort == 'SA':
            a = np.real(a)
        elif sort == '>' or sort == 'LR' or sort == 'LA':
            a = -np.real(a)
        elif sort == 'SI':
            a = np.imag(a)
        elif sort == 'LI':
            a = -np.imag(a)
        else:
            raise ValueError("unknown sort option " + repr(sort))
    return np.argsort(a, **kwargs)


def tenpy_arnoldi(matvec, psi0:np.ndarray, **kwargs):
    """
    Examples
    --------
    >>> try:
    >>>     from tenpy.linalg.sparse import NpcLinearOperator as LO
    >>>     import tenpy.linalg.np_conserved as npc
    >>>     from tenpy.linalg.krylov_based import Arnoldi
    >>> except ImportError:
    >>>     LO = object
    >>> 
    >>> class tpprojH(LO):
    >>>     def __init__(self, dot):
    >>>         self.matvec = lambda v: npc.Array.from_ndarray(dot(v.to_ndarray()), v.legs)
    >>> 
    >>> lo = tpprojH(matvec)
    >>> tenpy_arnoldi(lo, psi0)
    """
    import tenpy.linalg.np_conserved as npc
    from tenpy.linalg.krylov_based import Arnoldi
    from tenpy.linalg.sparse import NpcLinearOperator as LO
    class tpprojH(LO):
        def __init__(self, dot):
            self.matvec = lambda v: npc.Array.from_ndarray(dot(v.to_ndarray()), v.legs)
    lo = tpprojH(matvec)
    chinfo = npc.ChargeInfo()  # the second argument is just a descriptive name
    legcharges = npc.LegCharge.from_trivial(psi0.shape[0], chinfo)
    psi = npc.Array.from_ndarray(psi0,[legcharges])
    val, vec, _ = Arnoldi(lo, psi, options=kwargs).run()
    # show(val)
    return val[0], vec[0].to_ndarray()