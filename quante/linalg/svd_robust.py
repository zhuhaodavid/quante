# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-09-19 22:25:55
# @Last Modified by:   hzhu
# @Last Modified time: 2024-11-09 18:02:16
# This is from tenpy

import scipy.linalg as sla
import numpy as np
from typing import Optional

__all__ = ['svd', 'svd_truncate', 'truncate', 'TruncationError']


# 我们经常需要执行奇异值分解（SVD）。
# 通常，SVD 是一种矩阵分解，它总是定义良好并且也应该适用于病态矩阵。
# 但遗憾的是，`numpy.linalg.svd` 和 `scipy.linalg.svd` 有时会失败，
# 抛出 ``LinalgError("SVD did not converge")``。
# 原因是它们都调用了 LAPACK 函数 `#gesdd`
# （其中 `#` 取决于数据类型），该函数采用迭代方法，可能会失败。
# 然而，它通常比替代方案（且稳定的）`#gesvd` 快得多。

# 我们的解决方法如下：我们提供一个函数 `svd`，其调用签名与 scipy 的 svd 相同。
# 这个函数基本上只是 scipy 的 svd 的包装器，即我们继续调用更快的 `dgesdd`。
# 但如果失败，我们仍然可以使用 `dgesvd` 作为备份。

# 遗憾的是，`dgesvd` 和 `zgesvd` 直到 scipy 版本 '0.18.0' 才被包含进来（numpy 中也没有），
# 对于版本高于 '0.18.0' 的 scipy，我们使用 svd 的新关键字 'lapack_driver'
def svd(a,
        full_matrices=False,
        compute_uv=True,
        overwrite_a=False,
        check_finite=True,
        lapack_driver='gesdd',
        warn=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (更)稳定的奇异值分解（包装 :func:`scipy.linalg.svd`，如何是 hermitian 矩阵使用 `numpy.linalg.svd` 效率更高）
    
    将矩阵 `a` 分解为两个酉矩阵 ``U`` 和 ``Vh``，以及一个奇异值（实数，非负）的 1-D 数组 ``s``，
    使得 ``a == U @ S @ Vh``，其中 ``S`` 是一个适当形状的零矩阵，主对角线为 ``s``。

    带有 `gesvd` 备份计划。如果 `gesdd` 失败，尝试通过使用 lapack_driver `gesvd` 来避免引发 LinAlgError。

    Parameters
    ----------
    a : (M, N) 类数组
        要分解的矩阵。
    full_matrices : bool, 可选
        如果为 True（默认），`U` 和 `Vh` 的形状为 ``(M, M)``，``(N, N)``。
        如果为 False，形状为 ``(M, K)`` 和 ``(K, N)``，其中 ``K = min(M, N)``。
    compute_uv : bool, 可选
        是否计算 ``U`` 和 ``Vh``，除了 ``s`` 之外。默认是 True。
    overwrite_a : bool, 可选
        是否覆盖 `a`；可能会提高性能。默认是 False。
        如果 ``lapack_driver='gesdd'``，则忽略（即设置为 ``False``）。
    check_finite : bool, 可选
        是否检查输入矩阵仅包含有限数字。禁用可能会提高性能，但如果输入包含无穷大或 NaN，可能会导致问题（崩溃、非终止）。
    lapack_driver : {'gesdd', 'gesvd'}, 可选
        是否使用更高效的分治方法（``'gesdd'``）
        或通用矩形方法（``'gesvd'``）来计算 SVD。
        MATLAB 和 Octave 使用 ``'gesvd'`` 方法。
        默认是 ``'gesdd'``。
        如果 ``'gesdd'`` 失败，则使用 ``'gesvd'`` 作为备份。

        .. versionadded:: 0.18
    warn : bool
        当 SVD 失败时是否创建警告。

    Returns
    -------
    U : ndarray
        具有左奇异向量作为列的酉矩阵。形状为 ``(M, M)`` 或 ``(M, K)``，取决于 `full_matrices`。
    s : ndarray
        按非递增顺序排序的奇异值。形状为 (K,) ，其中 ``K = min(M, N)``。
    Vh : ndarray
        具有右奇异向量作为行的酉矩阵。形状为 ``(N, N)`` 或 ``(K, N)``，取决于 `full_matrices`。

    如果 ``compute_uv=False``，则仅返回 ``s``。

    Raises
    ------
    LinAlgError
        如果 SVD 计算不收敛。

    References
    ----------
    svdvals : 计算矩阵的奇异值。
    diagsvd : 给定向量 s 构造 Sigma 矩阵。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> m, n = 9, 6
    >>> a = rng.standard_normal((m, n)) + 1.j*rng.standard_normal((m, n))
    >>> U, s, Vh = linalg.svd(a)
    >>> U.shape,  s.shape, Vh.shape
    ((9, 9), (6,), (6, 6))

    从分解中重构原始矩阵：

    >>> sigma = np.zeros((m, n))
    >>> for i in range(min(m, n)):
    ...     sigma[i, i] = s[i]
    >>> a1 = np.dot(U, np.dot(sigma, Vh))
    >>> np.allclose(a, a1)
    True

    或者，使用 ``full_matrices=False``（注意此时 ``U`` 的形状为 ``(m, n)`` 而不是 ``(m, m)``）：

    >>> U, s, Vh = linalg.svd(a, full_matrices=False)
    >>> U.shape, s.shape, Vh.shape
    ((9, 6), (6,), (6, 6))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
    True

    >>> s2 = linalg.svd(a, compute_uv=False)
    >>> np.allclose(s, s2)
    True
    
    """
    if lapack_driver == 'gesdd':
        try:
            return sla.svd(a, full_matrices, compute_uv, False, check_finite)
        except np.linalg.LinAlgError:
            # 'gesdd' failed to converge, so we continue with the backup plan
            if warn:
                print("SVD with lapack_driver 'gesdd' failed. Use backup 'gesvd'",
                              flush=True)
            pass
    if lapack_driver not in ['gesdd', 'gesvd']:
        raise ValueError("invalid `lapack_driver`: " + str(lapack_driver))
    # 'gesvd' lapack driver
    return sla.svd(a, full_matrices, compute_uv, overwrite_a, check_finite, lapack_driver='gesvd')

def svd_truncate(mat:np.ndarray, Dc:Optional[int] = None, eps:float=1e-15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, v = svd(mat, full_matrices=False)  # 如果要用 numba 只需把 svd 换成 np.linalg.svd
    D_update = np.sum(s >= eps)
    D = min(Dc, D_update) if Dc is not None else D_update
    u, s, v = u[:, :D], s[:D], v[:D, :]
    return u, s, v


class TruncationError:
    r"""
    .. 警告 ::
            对于虚时间演化，这不是你感兴趣的误差！

    Parameters
    ----------
    eps : float
        所有被丢弃的施密特值平方和的总和。
        注意，如果你保留的奇异值达到 1.e-14（即比 64 位浮点数的机器精度稍高），
        `eps` 大约是 1.e-28（由于平方的原因）！
    ov : float
        重叠的下界 :math:`|\langle \psi_{trunc} | \psi_{correct} \rangle|^2`
        （假设两个状态都归一化）。
        这可能是你真正感兴趣的量。
        考虑了 `TEBD 维基百科文章 <https://en.wikipedia.org/wiki/Time-evolving_block_decimation>` 中误差部分解释的因子 2。
    """
    def __init__(self, eps:float = 0., ov:float = 1.) -> None:
        self.eps = eps
        self.ov = ov
    
    def __add__(self, other):
        res = TruncationError()
        res.eps = self.eps + other.eps
        res.ov = self.ov * other.ov
        return res
    
    def __iadd__(self, other):
        self.eps += other.eps
        self.ov *= other.ov
        return self

def truncate(S:np.ndarray, chi_max:int, svd_min:float, trunc_cut:float) -> tuple[np.ndarray, TruncationError]:
    """ 输入保证 S 需要是降序排列，且全都是正数！ """
    good = np.ones(len(S), dtype=np.bool_)
    if chi_max is not None:
        good2 = np.zeros(len(S), dtype=np.bool_)
        good2[:chi_max] = True
        good = good & good2
    if svd_min is not None:
        good = good & (S > svd_min)
    if trunc_cut is not None:
        revert_cumsum = np.cumsum(np.square(S)[::-1])[::-1]
        good = good & (revert_cumsum > trunc_cut**2)
    eps = S[~good].sum()
    ov = 1. - 2. * eps
    return good, TruncationError(eps, ov)

