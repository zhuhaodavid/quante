# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2024-12-11 11:26:19
# @Last Modified by:   dzwang
# @Last Modified time: 2025-01-11 20:09:43
import numpy as np
from ...linalg.svd_robust import TruncationError, svd


__all__ = ["truncate", "svd_tensor", "qr_tensor", "rq_tensor", ]


def truncate(S, chi_max=None, svd_tensor_min=None, trunc_cut=None):
    """ S 需要是降序排列，且全都是正数
    """
    good = np.ones(len(S), dtype=bool)
    if chi_max is not None:
        good2 = np.zeros(len(S), dtype=bool)
        good2[:chi_max] = True
        good = good & good2
    if svd_tensor_min is not None:
        good = good & (S > svd_tensor_min)
    if trunc_cut is not None:
        normS = S / np.linalg.norm(S)
        revert_cumsum = np.flip((np.cumsum(np.flip(normS**2, [0]), 0)), [0])
        good = good & (revert_cumsum > trunc_cut**2)
    eps = np.sum(S[~good]**2)
    ov = 1. - 2.*eps
    return good, TruncationError(eps, ov)


def svd_tensor(tensor:np.ndarray, lr_index=None, trunc_para=(None, None, None), full_matrices:bool=False):
    r"""张量 svd_tensor 分解，返回 A, S, B, error

    .. code-block:: text
    
        . ║          │     │
        --⬜--  ->  --▷--◇--⨞--
          ║          │     │

    Parameters
    ----------
    chi_max 保留的奇异值数目
    svd_tensor_min 最小奇异值
    trunc_cut 截断的比例

    normalize 是否归一化奇异值

    full_matrices 是否返回完整的矩阵

    Examples
    --------
    >>> tsr = tc.randn(3,7,5,2,dtype=tc.float64)
    >>> u,s,v,e = svd_tensor(tsr, lr_indx=[[1,2],[0,3]])
    >>> u.shape
    torch.Size([7, 5, 6])


    .. 警告:: 返回的张量 `U` 和 `V` 不是唯一的，也不是相对于 :attr:`A` 连续的。
                由于这种不唯一性，不同的硬件和软件可能会计算出不同的奇异向量。
                这种不唯一性是由于将任何一对奇异向量 :math:`u_k, v_k` 乘以 `-1`（在实数情况下）或
                乘以 :math:`e^{i \phi}, \phi \in \mathbb{R}`（在复数情况下）会产生另一对有效的奇异向量。
                因此，损失函数不应依赖于这种 :math:`e^{i \phi}` 量，因为它不是良定义的。
                在计算此函数的梯度时，会检查复数输入。因此，当输入是复数且在 CUDA 设备上时，
                计算此函数的梯度会将该设备与 CPU 同步。

    .. 警告:: 使用 `U` 或 `Vh` 计算的梯度仅在 :attr:`A` 没有重复的奇异值时才是有限的。
                如果 :attr:`A` 是矩形的，此外，零也不能是其奇异值之一。
                此外，如果任何两个奇异值之间的距离接近零，梯度将数值不稳定，因为它依赖于奇异值
                :math:`\sigma_i` 通过计算 
                :math:`\frac{1}{\min_{i \neq j} \sigma_i^2 - \sigma_j^2}`。
                在矩形情况下，当 :attr:`A` 有小奇异值时，梯度也将数值不稳定，因为它还依赖于计算
                :math:`\frac{1}{\sigma_i}`。

    """
    chi_max, svd_min, trunc_cut = trunc_para
    shape = tensor.shape

    if lr_index is None:
        left, right = shape[:len(shape)//2], shape[len(shape)//2:]
        tensor = tensor.reshape(np.prod(left), -1)
    else:
        left_index, right_index = lr_index
        left = [shape[i] for i in left_index]
        right = [shape[i] for i in right_index]
        tensor = tensor.transpose(*(left_index + right_index)).reshape(np.prod(left), np.prod(right))

    u, s, vt = svd(tensor, full_matrices=full_matrices)

    good, trunc_err = truncate(s, chi_max, svd_min, trunc_cut)
    if not all(good):
        u = u[:, good]
        s = s[good]
        vt = vt[good, :]

    return u.reshape(*left, -1), s, vt.reshape(-1, *right), trunc_err


def qr_tensor(tensor:np.ndarray, lr_index=None) -> tuple[np.ndarray, np.ndarray]:
    r"""

    .. code-block:: text
    
               |                           |
              (b)                         (b)
               |            qr_tensor             |
        --(a)--⬜--(c)--    ---->    --(a)--▷--(d)--⬜--(c)--
               W                           A       S

    Parameters
    ----------

    Examples
    --------
    >>> tsr = tc.randn(3,7,5,2,dtype=tc.float64)
    >>> q, r = qr_tensor(tsr, lr_indx=[[1,2],[0,3]])
    >>> print(q.shape, r.shape)


    .. 警告:: `R` 的对角线元素不一定为正。
            因此，返回的 qr_tensor 分解仅在 `R` 的对角线符号上是唯一的。
            因此，不同的平台（如 NumPy）或不同设备上的输入，
            可能会产生不同的有效分解。

    .. 警告:: qr_tensor 分解仅在每个矩阵的前 `k = min(m, n)` 列线性无关时才有定义。
            如果不满足此条件，不会抛出错误，但生成的 qr_tensor 可能不正确，
            其自动微分可能会失败或产生不正确的结果。
    """
    shape = tensor.shape

    if lr_index is None:
        left = shape[:-1]
        right = (shape[-1], )
        mat = tensor.reshape(-1, *right)
    else:
        left_indx, right_indx = lr_index
        left = [shape[i] for i in left_indx]
        right = [shape[i] for i in right_indx]
        mat = tensor.transpose(*(left_indx + right_indx)).reshape(np.prod(left), np.prod(right))

    Q, R = np.linalg.qr(mat)
    return Q.reshape(*left, -1), R.reshape(-1, *right)


def rq_tensor(tensor:np.ndarray, lr_index=None) -> tuple[np.ndarray, np.ndarray]:
    """
    .. code-block:: text
    
                       |                        |         
                      (b)                      (b)        
                       |           qr           |         
        --(a)--⬜--(d)--⨞--(c)--   <---   --(a)--⬜--(c)--    
               S       A                        W

    Parameters
    ----------

    lrdims 左右指标

    `None` 从正中间的指标分开做 svd_tensor
    
    (`left_indx`, `right_indx`): left_indx 为左指标，right_indx 为右指标
    """
    shape = tensor.shape

    if lr_index is None:
        # 如果没给，用左右侧的指标分解
        left = (shape[0],)
        right = shape[1:]
        mat = tensor.reshape(*left, -1)
    else:
        # 如果给了，需要排序
        left_index, right_index = lr_index
        left = [shape[i] for i in left_index]
        right = [shape[i] for i in right_index]
        mat = tensor.transpose(*(left_index + right_index)).reshape(np.prod(left), np.prod(right))

    q, r = np.linalg.qr(mat.T)
    L, U = r.T, q.T

    return L.reshape(*left, -1), U.reshape(-1, *right)
