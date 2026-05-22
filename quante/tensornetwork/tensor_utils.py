# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-22 21:30:40
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-23 00:01:09


from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse import issparse


@dataclass
class TruncationError:
    eps: float
    ov: float

    def __add__(self, other):
        return TruncationError(self.eps + other.eps, self.ov * other.ov)

    def __iadd__(self, other):
        self.eps += other.eps
        self.ov *= other.ov
        return self


def clone(tensor):
    return deepcopy(tensor)


def tonp(tensors):
    if isinstance(tensors, list):
        return [tonp(x) for x in tensors]
    if issparse(tensors):
        return tensors
    return np.asarray(tensors)


def totc(data, dtype=None):
    if isinstance(data, list):
        return [totc(x, dtype=dtype) for x in data]
    return np.asarray(data, dtype=dtype)


def promote_dtype(*datas):
    dtype = np.complex128 if any(np.iscomplexobj(x) for x in datas) else np.float64
    return [np.asarray(x, dtype=dtype) for x in datas]


def real_if_close(a, tol=100):
    return np.real_if_close(a, tol=tol)


def open_grad(tensors):
    return None


def close_grad(tensors):
    return None


class AdaptiveLRScheduler:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("NumPy backend has no optimizer/lr scheduler.")


def log_or_not_update(data, lognm, use_log):
    """Normalize ``data`` and update ``lognm`` when logarithmic storage is used."""
    if use_log:
        nm = np.linalg.norm(data)
        data = data / nm
        lognm = lognm + np.log(nm)
    return data, lognm


def truncate(S, chi_max=None, svd_min=None, trunc_cut=None):
    """S 需要是降序排列，且全都是正数！"""
    S = np.asarray(S)
    good = np.ones(len(S), dtype=bool)
    if chi_max is not None:
        keep = np.zeros(len(S), dtype=bool)
        keep[:chi_max] = True
        good &= keep
    if svd_min is not None:
        good &= S > svd_min
    if trunc_cut is not None:
        norm = np.linalg.norm(S)
        if norm != 0:
            normS = S / norm
            tail = np.flip(np.cumsum(np.flip(normS**2)))
            good &= tail > trunc_cut
    eps = float(np.sum(S[~good] ** 2))
    return good, TruncationError(eps=eps, ov=1.0 - 2.0 * eps)


def _split_tensor(tsr, lr_indx):
    shp = tsr.shape
    if lr_indx is None:
        left = tuple(range(len(shp) // 2))
        right = tuple(range(len(shp) // 2, len(shp)))
    else:
        left, right = tuple(lr_indx[0]), tuple(lr_indx[1])
    lshape = tuple(shp[i] for i in left)
    rshape = tuple(shp[i] for i in right)
    mat = np.transpose(tsr, left + right).reshape(np.prod(lshape, dtype=int), np.prod(rshape, dtype=int))
    return mat, lshape, rshape


def svd(tsr, *, lr_indx=None, trunc_para=(None, None, None), full_matrices=False):
    r"""张量 svd 分解，返回 A, S, B, error

    .. code-block:: text

        . ║          │     │
        --⬜--  ->  --▷--◇--⨞--
          ║          │     │

    Parameters
    ----------

    lr_indx 左右指标。

    ``None`` 从正中间的指标分开做 svd。

    ``(left_indx, right_indx)``: left_indx 为左指标，right_indx 为右指标。

    trunc_para = (chi_max, svd_min, trunc_cut)

    chi_max 保留的奇异值数目。

    svd_min 最小奇异值。

    trunc_cut 截断的比例。

    full_matrices 是否返回完整的矩阵。
    """
    mat, lshape, rshape = _split_tensor(np.asarray(tsr), lr_indx)
    U, S, Vh = scipy.linalg.svd(mat, full_matrices=full_matrices)
    good, err = truncate(S, *trunc_para)
    U, S, Vh = U[:, good], S[good], Vh[good, :]
    return U.reshape(*lshape, -1), S, Vh.reshape(-1, *rshape), err


def qr(tsr, *, lr_indx=None):
    r"""

    .. code-block:: text

               |                           |
              (b)                         (b)
               |            QR             |
        --(a)--⬜--(c)--    ---->    --(a)--▷--(d)--⬜--(c)--
               W                           A       S

    Parameters
    ----------

    lr_indx 左右指标。

    ``None`` 用左右侧的指标分解。

    ``(left_indx, right_indx)``: left_indx 为左指标，right_indx 为右指标。
    """
    tsr = np.asarray(tsr)
    if lr_indx is None:
        lshape = tsr.shape[:-1]
        rshape = (tsr.shape[-1],)
        mat = tsr.reshape(int(np.prod(lshape)), rshape[0])
    else:
        mat, lshape, rshape = _split_tensor(tsr, lr_indx)
    Q, R = np.linalg.qr(mat)
    return Q.reshape(*lshape, -1), R.reshape(-1, *rshape)


def rq(tsr, *, lr_indx=None):
    r"""
    .. code-block:: text

                       |                       |
                      (b)                     (b)
                       |           QR          |
        --(a)--⬜--(d)--⨞--(c)--   <---  --(a)--⬜--(c)--
               S       A                       W

    Parameters
    ----------

    lr_indx 左右指标。

    ``None`` 用左右侧的指标分解。

    ``(left_indx, right_indx)``: left_indx 为左指标，right_indx 为右指标。
    """
    tsr = np.asarray(tsr)
    if lr_indx is None:
        lshape = (tsr.shape[0],)
        rshape = tsr.shape[1:]
        mat = tsr.reshape(lshape[0], int(np.prod(rshape)))
    else:
        mat, lshape, rshape = _split_tensor(tsr, lr_indx)
    Q, R = np.linalg.qr(mat.T)
    L, U = R.T, Q.T
    return L.reshape(*lshape, -1), U.reshape(-1, *rshape)


def eigh(tsr, *, lr_indx=None, eigdirection=None, trunc_para=(None, None, None), pertube=None, pos=None, drt=None):
    r"""
    注意！！
    虽然返回的是 U, S, V，但 U, V 不一定是半幺的！！
    如果 eigdirection="right"，那么 U 是半幺的，V 不是；如果 eigdirection="left"，那么 V 是半幺的，U 不是。
    但总之 mat = U @ V。

    本征分解的原理如图所示（eigdirection="left"为例）：

    .. code-block:: text

        目标 - 利用本征分解实现（包含裁剪）：
                 ║                          |        |
               (bc)                        (b)      (c)
                 ║                          |        |
         --(a)---⬜---(d)--     -->   --(a)--▷--(e)---⬜--(d)--
                 W                          U        A

    ``pertube`` 用于和原实现保持一致：指定后会加到密度矩阵上，且必须显式指定 eigdirection。
    ``pos`` 和 ``drt`` 只是兼容参数，不参与 NumPy 计算。
    """
    mat, lshape, rshape = _split_tensor(np.asarray(tsr), lr_indx)
    chi_max, svd_min, trunc_cut = trunc_para
    if pertube is not None:
        assert eigdirection is not None, "direction must be specified when eigen_perturbation is not None"
    if eigdirection is None:
        eigdirection = "right" if mat.shape[0] < mat.shape[1] else "left"
    elif pertube is None:
        if chi_max is not None:
            chi_max = min(min(mat.shape), chi_max)
        else:
            chi_max = min(mat.shape)

    if eigdirection == "right":
        rho = mat @ mat.conj().T if pertube is None else mat @ mat.conj().T + pertube
        S2, U = np.linalg.eigh(rho)
        S2 = np.flip(S2, axis=0)
        U = np.flip(U, axis=1)
        S = np.sqrt(np.maximum(S2, 0))
        good, err = truncate(S, chi_max, svd_min, trunc_cut)
        U, S = U[:, good], S[good]
        Vh = U.conj().T @ mat
    elif eigdirection == "left":
        rho = mat.conj().T @ mat if pertube is None else mat.conj().T @ mat + pertube
        S2, V = np.linalg.eigh(rho)
        S2 = np.flip(S2, axis=0)
        V = np.flip(V, axis=1)
        S = np.sqrt(np.maximum(S2, 0))
        good, err = truncate(S, chi_max, svd_min, trunc_cut)
        V, S = V[:, good], S[good]
        Vh = V.conj().T
        U = mat @ V
    else:
        raise ValueError("direction must be 'left' or 'right'")

    return U.reshape(*lshape, -1), S, Vh.reshape(-1, *rshape), err, eigdirection


def tt_decompose(tsr, phys_dim=2, trunc_para=(None, None, None)):
    r"""执行 tt 分解，是 full_contract 的逆过程。

    输入 tsr 可以是一维或二维数组。

    返回 tt, Ss, lognm。
    """
    tsr = np.asarray(tsr)
    if isinstance(phys_dim, int):
        tmp = np.log(tsr.shape[0]) / np.log(phys_dim)
        assert float(tmp).is_integer(), "The physical dimension is not compatible with the tensor shape."
        phys_dim = [phys_dim] * int(tmp)
    else:
        phys_dim = list(phys_dim)
        assert int(np.prod(phys_dim)) == tsr.shape[0], "The physical dimension is not compatible with the tensor shape."

    # 然后进行 TT 分解。这里保持原实现的右到左分解顺序、奇异值归一化和 lognm 累积行为。
    nsite = len(phys_dim)
    Ws = [None] * nsite
    Ss = [None] * (len(phys_dim) + 1)
    Ss[-1] = np.ones(1, dtype=np.float64)
    lognm = 0.0

    if tsr.ndim == 1:
        lstdim = 1
        for i in range(1, nsite + 1):
            U, S, Vh = scipy.linalg.svd(tsr.reshape(-1, phys_dim[-i] * lstdim), full_matrices=False)
            nms = np.linalg.norm(S)
            S = S / nms
            lognm = lognm + np.log(nms)
            good, _ = truncate(S, *trunc_para)
            U, S, Vh = U[:, good], S[good], Vh[good, :]
            Ws[-i] = Vh.reshape(-1, phys_dim[-i], lstdim)
            Ss[-i - 1] = S
            tsr = U * S.reshape(1, -1)
            lstdim = len(S)
    elif tsr.ndim == 2:
        dims = list(phys_dim)
        for i in range(1, nsite + 1):
            permute_indx = (
                list(range(len(dims) - 1))
                + [j + len(dims) - 1 for j in range(1, len(dims))]
                + [len(dims) - 1, 2 * len(dims) - 1, 2 * len(dims)]
            )
            tsr = np.transpose(tsr.reshape(*dims, *dims, -1), permute_indx)

            lstdim = tsr.shape[-1]
            U, S, Vh = scipy.linalg.svd(tsr.reshape(-1, dims[-1] ** 2 * lstdim), full_matrices=False)
            nms = np.linalg.norm(S)
            S = S / nms
            lognm = lognm + np.log(nms)
            good, _ = truncate(S, *trunc_para)
            U, S, Vh = U[:, good], S[good], Vh[good, :]

            Ws[-i] = Vh.reshape(-1, dims[-1], dims[-1], lstdim)
            Ss[-i - 1] = S
            tsr = U * S.reshape(1, -1)
            dims = dims[:-1]
    else:
        raise ValueError("The tensor must be 1- or 2-dimensional.")
    sign = np.sign(np.real(U[0, 0]))
    if sign == 0:
        sign = 1.0
    Ws[0] = Ws[0] * sign
    lognm = lognm + np.log(np.abs(U[0, 0]))
    Ss[0] = np.ones(1, dtype=np.float64)
    return Ws, Ss, lognm


def argsort(a, sort=None, refer=None, **kwargs):
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
            b = abs(a)
        elif sort == 'm>' or sort == 'LM':
            b = -abs(a)
        elif sort == '<' or sort == 'SR' or sort == 'SA':
            b = np.real(a)
        elif sort == '>' or sort == 'LR' or sort == 'LA':
            b = -np.real(a)
        elif sort == 'SI':
            b = np.imag(a)
        elif sort == 'LI':
            b = -np.imag(a)
        else:
            raise ValueError("unknown sort option " + repr(sort))
    arg = np.argsort(b, **kwargs)
    if np.iscomplexobj(a) and refer is not None and (sort == 'LM' or sort == 'SM'):
        # todo 如何更好的判断简并情况??
        #!! 这里只是 pMPSv_app 项目的选择
        for i in range(len(a)):
            if abs(a[arg[i]].imag) < 1. and a[arg[i]].real > 0:
                arg[0], arg[i] = arg[i], arg[0]
                diff = abs(a - refer)
                argdiff = np.argsort(diff)
                if diff[argdiff[0]] < 1e-2*abs(a[arg[0]] - refer):
                    arg[0], argdiff[0] = argdiff[0], arg[0]
                break
    return arg

