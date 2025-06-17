# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-26 17:09:16
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:30:54

import numpy as np
import scipy.sparse.linalg as spalg
from typing import Callable

__all__ = [
    "tenpy_arnoldi", 
    "arnoldi_ground_state"
]

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