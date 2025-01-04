# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-07-15 14:19:55
# @Last Modified by:   hzhu
# @Last Modified time: 2024-11-09 18:05:06

#!! linalg 中不要 import linalg 之外的文件

# todo: ikron 和 partial_trace 需要重写, partial_trace 存在问题

import math
import numpy as _np
import scipy.linalg as _sla
import scipy.sparse as _sparse
import itertools
from numbers import Integral
from typing import Union, Optional
import functools

__all__ = [
    "norm",
    "expm",
    "sqrtm",
    "logm",
]

__all__ += [
    "uptrig",
    "uptrig_inv",
    "uptrigindex",
    "uptrigindex_inv",
    'observe_states'
]

__all__ += [
    "kron",
    "ikron",
    "partial_trace",
]

__all__ += [
    "log_Gauss", "find_boundary", "interp", "fit", 
]

def norm(v: _np.ndarray) -> float:
    """无论是 torch.Tensor, numpy.ndarray 还是 scipy.sparse.sparray，都可以计算范数"""
    try:
        return v.norm()
    except AttributeError:
        try: 
            return _sparse.linalg.norm(v)
        except TypeError:
            return _np.linalg.norm(v)

def _eigh_hermitian_matrix(A:_np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
    from .eig_modified import eigh
    return eigh(A)

def expm(A:_np.ndarray, c: Union[float, complex] = 1.0) -> _np.ndarray:
    """Exponential Matrix, Hermitian matrix can be accelerated
    """
    is_herm = _np.allclose(A, A.conj().T)
    if is_herm:
        eigenvalues, eigenstates = _eigh_hermitian_matrix(A)
        if _np.isreal(A).all() and _np.isreal(c):
            new_eigenvalues = _np.exp(eigenvalues * c)
        else:
            new_eigenvalues = _np.exp(eigenvalues * c).astype(complex)
        return (eigenstates * new_eigenvalues) @ eigenstates.conj().transpose()
    else:
        return _sla.expm(c*A)

def sqrtm(A:_np.ndarray) -> _np.ndarray:
    """Square root Matrix, Hermitian matrix can be accelerated

    Args:
        A (np.ndarray): Matrix

    Returns:
        np.ndarray: square root Matrix
    """
    is_herm = _np.allclose(A, A.T.conj())
    if is_herm:
        eigenvalues, eigenstates = _eigh_hermitian_matrix(A)
        return eigenstates * _np.sqrt(eigenvalues.astype(complex)) @ eigenstates.T.conj()
    else:
        return _sla.sqrtm(A)


def logm(A:_np.ndarray) -> _np.ndarray:
    """Logarithm Matrix, Hermitain can be accelerated

    Args:
        A (np.ndarray): Matrix

    Returns:
        np.ndarray: Logarithm matrix
    """
    is_herm = _np.allclose(A, A.T.conj())
    if is_herm:
        eigenvalues, eigenstates = _eigh_hermitian_matrix(A)
        return eigenstates * _np.log(eigenvalues.astype(complex)) @ eigenstates.T.conj()
    else:
        return _sla.logm(A)


#####################################
# 矩阵元的操作
#####################################

def uptrig(mat):
    """返回矩阵的对角元和上三角矩阵元"""
    from .usenumba.operations_numba import uptri2list
    return uptri2list(mat)

def uptrig_inv(lis):
    """将 uptrig 上三角矩阵元素重新组装为上三角矩阵"""
    from .usenumba.operations_numba import list2uptri
    return list2uptri(lis)

def uptrigindex(row_indx:list, col_indx:list, dim:int) -> _np.ndarray:
    """uptrig 将 (i,j) 指标变为生成的 list 的指标"""
    from .usenumba.operations_numba import _uptrigindex
    return _uptrigindex(row_indx, col_indx, dim)

def uptrigindex_inv(indices:list, dim:int) -> _np.ndarray:
    """uptrig 生成的 list 的指标变为上三角矩阵的指标"""
    from .usenumba.operations_numba import _uptrigindex_inv
    return _uptrigindex_inv(indices, dim)

def observe_states(vecs:_np.ndarray, O:_np.ndarray) -> _np.ndarray:
    """
    计算 vecs 的观测值：
    
    Examples
    --------
    >>> mat = qla.rdmat(100, dtype=np.complex128)
    >>> eigs = qla.rdmat(100, dtype=np.complex128)
    >>> qla.observe_states(eigs, mat)
    """
    if  _np.issubdtype(vecs.dtype, _np.floating) and _np.issubdtype(O.dtype, _np.floating):
        from .usenumba.operations_numba import observe_states_float
        return observe_states_float(vecs, O)
    else:
        from .usenumba.operations_numba import observe_states_complex
        return observe_states_complex(vecs.astype(complex), O.astype(complex))
    

#####################################
# kron
#####################################

def kron(
    *ops: Union[_np.ndarray, _sparse.sparray],
    stype: Optional[str] = None,  # 后面的参数只有当输入有稀疏矩阵时才有用
    coo_build: bool = False, 
    chopped: bool = True,
    parallel: bool = False
    ) -> Union[_np.ndarray, _sparse.sparray]:
    """
    计算多个算符的克罗内克积(无论是密集矩阵/稀疏矩阵)。
    只要有一个是稀疏的，结果就是稀疏的。
    
    Examples
    ---------
    >>> op1 = np.array([[1, 2], [3, 4]])
    >>> op2 = np.array([[5, 6], [7, 8]])
    >>> kron(op1, op2)
    array([[ 5,  6, 10, 12],
           [15, 18, 20, 24],
           [ 7,  8, 14, 16],
           [21, 24, 28, 32]])

    Parameters
    ----------
    - ops (Union[_np.ndarray, _sparse.sparray]): 要计算克罗内克积的算符。
    - stype (str, 可选): 返回的稀疏矩阵的类型，可以是 "csr" (压缩稀疏行)、"bsr" (块稀疏行)、"csc" (压缩稀疏列) 或 "coo" (坐标格式)。默认为None。
    - chopped (bool, 可选): 是否在计算之前将算符中的零值设置为0。默认为True。
    - parallel (bool, 可选): 是否使用并行计算。默认为False。
    
    """
    tmp_stype = "coo" if coo_build or stype == "coo" else None
    
    if parallel:
        from .usenumba.numba_settings import parallel_reduce
        reducer = parallel_reduce
    else:
        reducer = functools.reduce
        
    kron_fuc = functools.partial(kron_dispatch, stype=tmp_stype, chopped=chopped)
    X = reducer(kron_fuc, ops)  # 通过 reduce 定义的程序将 kron 连续作用到 ops 上
    
    if stype is not None:
        return X.asformat(stype)
    
    if coo_build or (_sparse.issparse(X) and X.format == "coo"):
        return X.asformat("csr")
    return X

def kron_dispatch(a, b, stype=None, chopped=True):
    if _sparse.issparse(a) or _sparse.issparse(b):
        return _kron_sparse(a, b, stype=stype, chopped=chopped)

    return _kron_dense(a, b)

def _kron_dense(a, b):
    return _np.kron(a, b)

def _kron_sparse(a, b, stype=None, chopped=True):
    if stype is None:
        stype = (
            "bsr"
            if isinstance(b, _np.ndarray) or b.format == "bsr"
            else b.format
            if isinstance(a, _np.ndarray)
            else "csc"
            if a.format == "csc" and b.format == "csc"
            else "csr"
        )
    res = _sparse.kron(a, b).asformat(stype)
    if chopped:
        chop(res)
    return res

def chop(data, tol=1.0e-15, inplace=True):
    """
    将密集或稀疏数组中小于某个阈值的值设置为0。

    Parameters
    ----------
    - qob (密集或稀疏向量或算符): 要处理的量子对象。
    - tol (float, 可选): 低于该阈值的值将被设置为0，阈值是 ``max(abs(qob))`` 的分数。默认为1e-10。
    - inplace (bool, 可选): 是否在输入数组上操作或返回副本。默认为False。
    """
    minm = _np.abs(data).max() * tol  # minimum value tolerated
    if not inplace:
        data = data.copy()
    if _sparse.issparse(data):
        data.data.real[_np.abs(data.data.real) < minm] = 0.0
        if _np.issubdtype(data.dtype, _np.complexfloating):
            data.data.imag[_np.abs(data.data.imag) < minm] = 0.0
        data.eliminate_zeros()
    else:
        data.real[_np.abs(data.real) < minm] = 0.0
        data.imag[_np.abs(data.imag) < minm] = 0.0
    return data

#####################################
# ikron
#####################################

# todo 重写这段代码
def ikron(
    ops: Union[_np.ndarray, _sparse.sparray],
    dims: _np.ndarray,
    inds: Union[int, _np.ndarray],
    sparse: Optional[bool] = None,
    stype: Optional[str] = None,
    coo_build: bool = False,
    parallel: bool = False,
) -> Union[_np.ndarray, _sparse.sparray]:
    """
    多个单位一起生成
    
    Examples
    --------
    >>> xy = op.pauli_matrix('xy')
    >>> dims = (2, ) * 10
    >>> mat = qla.ikron(xy, dims, [3,4])

    得到的就是 iiixyiiiii
    """
    # 确保ops是一个列表
    if isinstance(ops, (_np.ndarray, _sparse.spmatrix, _sparse.sparray)):
        ops = (ops,)
    # 获取ops中所有操作数的数据类型
    dtype = common_type(*ops)
    # 确保维度和坐标已经被展平
    if _np.ndim(dims) > 1:
        raise NotImplementedError
    # 确保inds是一个列表
    elif _np.ndim(inds) == 0:
        inds = (inds,)
    # 从ops列表中推断稀疏性
    if sparse is None:
        sparse = any(_sparse.issparse(op) for op in ops)
    # 去除重复的inds
    if len(set(inds)) != len(inds):
        newops = []
        newinds = []
        for i, indsi in enumerate(inds):
            if indsi not in newinds:
                newops.append(ops[i])
                newinds.append(indsi)
            else:
                k = newinds.index(indsi)
                newops[k] = newops[k] @ ops[i]
        inds, ops = newinds, newops
    # 创建一个按索引排序的操作数列表
    inds, ops = zip(*sorted(zip(inds, itertools.cycle(ops))))
    inds, ops = set(inds), iter(ops)
    eye_kws = {
        "sparse": sparse,
        "stype": "coo",
        "dtype": dtype,
    }
    def gen_ops():
        cff_id = 1  # 记录压缩相邻的单位矩阵
        cff_ov = 1  # 记录叠加操作到多个维度
        for ind, dim in enumerate(dims):
            # 检查是否在此放置操作数
            if ind in inds:
                # 检查是否需要前置的单位矩阵
                if cff_id > 1:
                    yield eye(cff_id, **eye_kws)
                    cff_id = 1  # 重置单位矩阵大小
                # 检查是否是放置块中的第一个子系统
                if cff_ov == 1:
                    op = next(ops)
                    sz_op = op.shape[0]
                # 最终维度（块或总维度）-> 放置操作数
                if cff_ov * dim == sz_op or dim == -1:
                    yield op
                    cff_ov = 1
                # 累积子维度
                else:
                    cff_ov *= dim
            # 检查是否在多个子系统中叠加操作数
            elif cff_ov > 1:
                cff_ov *= dim
            # 否则累积相邻的单位矩阵
            else:
                cff_id *= dim
        # 检查是否需要后置的单位矩阵
        if cff_id > 1:
            yield eye(cff_id, **eye_kws)
    # 返回克罗内克积
    return kron(
        *gen_ops(),
        stype=stype,
        coo_build=coo_build,
        parallel=parallel
    )

_COMPLEX_DTYPES = {"complex64", "complex128"}
_DOUBLE_DTYPES = {"float64", "complex128"}
_DTYPE_MAP = {
    (False, False): "float32",
    (False, True): "float64",
    (True, False): "complex64",
    (True, True): "complex128",
}

def common_type(*arrays):
    """Quick compute the minimal dtype sufficient for ``arrays``."""
    dtypes = {array.dtype.name for array in arrays}
    has_complex = not _COMPLEX_DTYPES.isdisjoint(dtypes)
    has_double = not _DOUBLE_DTYPES.isdisjoint(dtypes)
    return _DTYPE_MAP[has_complex, has_double]

def eye(dim, sparse=False, stype="csr", dtype=float):
    if sparse:
        return _sparse.eye(dim, dtype=dtype, format=stype)
    else:
        return _np.eye(dim, dtype=dtype)
    

#####################################
# partial trace 这段代码来自 quimb
#####################################

 # todo quimb 写的太复杂，重写来实现这个功能
def partial_trace(p, dims, keep):
    """
    计算 p 的部分迹，dims 是各个格点的维数，keep 是要保留的格点位置
    
    Examples
    --------
    >>> L = 4
    >>> vec = np.random.rand(2**L)  # 对稀疏向量存在问题
    >>> rho_red = qla.partial_trace(vec, [2]*L, [1])
    >>> print(rho_red)  # np.einsum('abcdaecd->be', rho.reshape([2]*(2*L)))
    
    以及对矩阵：
    
    >>> L = 4
    >>> vec = ed.rdmat_rho(2**L, sparse=True)  # 或 False，必须是厄密的
    >>> rho_red = qla.partial_trace(vec, [2]*L, [1])
    >>> print(rho_red)
    """
    try:
        ndim = dims.ndim
    except AttributeError:
        ndim = len(_find_shape_of_nested_int_array(dims))

    if ndim >= 2:
        raise NotImplementedError
        # dims, keep = dim_map(dims, keep)

    if _sparse.issparse(p):
        return _partial_trace_simple(p, dims, keep)

    return _partial_trace_dense(p, dims, keep)

def _find_shape_of_nested_int_array(x):
    """Take a n-nested list/tuple of integers and find its array shape."""
    shape = [len(x)]
    sub_x = x[0]
    while not _np.issubdtype(type(sub_x), _np.integer):
        shape.append(len(sub_x))
        sub_x = sub_x[0]
    return tuple(shape)

def _partial_trace_simple(p, dims, keep):
    """Simple partial trace made up of consecutive single subsystem partial
    traces, augmented by 'compressing' the dimensions each time.
    """
    dims, keep = dim_compress(dims, keep)
    if len(keep) == 1:
        return _trace_keep(p, dims, *keep)
    lmax = max(enumerate(dims), key=lambda ix: (ix[0] not in keep) * ix[1])[0]
    p = _trace_lose(p, dims, lmax)
    dims = (*dims[:lmax], *dims[lmax + 1 :])
    keep = {(ind if ind < lmax else ind - 1) for ind in keep}
    return _partial_trace_simple(p, dims, keep)

def dim_compress(dims, inds):
    """Compress neighbouring subsytem dimensions.

    Take some dimensions and target indices and compress both, i.e.
    merge adjacent dimensions that are both either in ``dims`` or not. For
    example, if tensoring an operator onto a single site, with many sites
    the identity, treat these as single large identities.
    """
    if isinstance(inds, Integral):
        inds = (inds,)

    dims, inds = zip(*_dim_compressor(dims, inds))
    inds = tuple(i for i, b in enumerate(inds) if b)

    return dims, inds

def _dim_compressor(dims, inds):  # pragma: no cover
    """Helper function for ``dim_compress`` that does the heavy lifting.

    Parameters
    ----------
    dims : sequence of int
        The subsystem dimensions.
    inds : sequence of int
        The indices of the 'marked' subsystems.

    Returns
    -------
    generator of (int, int)
        Sequence of pairs of new dimension subsystem with marked flag {0, 1}.
    """
    blocksize_id = blocksize_op = 1
    autoplace_count = 0
    for i, dim in enumerate(dims):
        if dim < 0:
            if blocksize_op > 1:
                yield (blocksize_op, 1)
                blocksize_op = 1
            elif blocksize_id > 1:
                yield (blocksize_id, 0)
                blocksize_id = 1
            autoplace_count += dim
        elif i in inds:
            if blocksize_id > 1:
                yield (blocksize_id, 0)
                blocksize_id = 1
            elif autoplace_count < 0:
                yield (autoplace_count, 1)
                autoplace_count = 0
            blocksize_op *= dim
        else:
            if blocksize_op > 1:
                yield (blocksize_op, 1)
                blocksize_op = 1
            elif autoplace_count < 0:
                yield (autoplace_count, 1)
                autoplace_count = 0
            blocksize_id *= dim
    yield (
        (blocksize_op, 1)
        if blocksize_op > 1
        else (blocksize_id, 0)
        if blocksize_id > 1
        else (autoplace_count, 1)
    )

def _trace_keep(p, dims, keep):
    """Simple partial trace where the single subsytem
    at ``keep`` is kept.
    """
    dims = _np.asarray(dims)
    s = dims[keep]
    a = math.prod(dims[:keep])
    b = math.prod(dims[keep + 1 :])
    rhos = _np.zeros(shape=(s, s), dtype=_np.complex128)
    for i in range(s):
        for j in range(i, s):
            for k in range(a):
                i_i = b * i + s * b * k
                i_f = b * i + s * b * k + b
                j_i = b * j + s * b * k
                j_f = b * j + s * b * k + b
                rhos[i, j] += p[i_i:i_f, j_i:j_f].trace()
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos

def _trace_lose(p, dims, lose):
    """Simple partial trace where the single subsytem at ``lose``
    is traced out.
    """
    dims = _np.asarray(dims)
    e = dims[lose]
    a = math.prod(dims[:lose])
    b = math.prod(dims[lose + 1 :])
    rhos = _np.zeros(shape=(a * b, a * b), dtype=_np.complex128)
    for i in range(a * b):
        for j in range(i, a * b):
            i_i = e * b * (i // b) + (i % b)
            i_f = e * b * (i // b) + (i % b) + (e - 1) * b + 1
            j_i = e * b * (j // b) + (j % b)
            j_f = e * b * (j // b) + (j % b) + (e - 1) * b + 1
            rhos[i, j] = p[i_i:i_f:b, j_i:j_f:b].trace()
            if j != i:
                rhos[j, i] = rhos[i, j].conjugate()
    return rhos

def _partial_trace_dense(p, dims, keep):
    """Perform partial trace of a dense matrix."""
    if isinstance(keep, Integral):
        keep = (keep,)
    shp = p.shape
    if len(shp) == 1 or (len(shp) == 2 and (shp[0] == 1 or shp[1] == 1)):  # p = psi
        p = _np.asarray(p).reshape(dims)
        lose = ind_complement(keep, len(dims))
        p = _np.tensordot(p, p.conj(), (lose, lose))
        d = int(p.size**0.5)
        return p.reshape((d, d))
    else:
        p = _np.asarray(p).reshape((*dims, *dims))
        total_dims = len(dims)
        lose = ind_complement(keep, total_dims)
        lose2 = tuple(ind + total_dims for ind in lose)
        p = itrace(p, (lose, lose2))
    d = int(p.size**0.5)
    return p.reshape((d, d))

def ind_complement(inds, n):
    """Return the indices below ``n`` not contained in ``inds``."""
    return tuple(i for i in range(n) if i not in inds)

def itrace(a, axes=(0, 1)):
    """slower than np.enisum"""
    # Single index pair to trace out
    if isinstance(axes[0], Integral):
        return _np.trace(a, axis1=axes[0], axis2=axes[1])
    elif len(axes[0]) == 1:
        return _np.trace(a, axis1=axes[0][0], axis2=axes[1][0])

    # Multiple index pairs to trace out
    gone = set()
    for axis1, axis2 in zip(*axes):
        # Modify indices to adjust for traced out dimensions
        mod1 = sum(x < axis1 for x in gone)
        mod2 = sum(x < axis2 for x in gone)
        gone |= {axis1, axis2}
        a = _np.trace(a, axis1=axis1 - mod1, axis2=axis2 - mod2)
    return a





########################################
# 拟合
########################################


def fit(xs: list, ys: list, polynomial_degree: int) -> tuple:
    """
    将曲线光滑化
    
    参数:
    xs (list): x 坐标的数据列表
    ys (list): y 坐标的数据列表
    polynomial_degree (int): 多项式的阶数，表示拟合曲线的复杂度
    
    返回:
    tuple: 包含两个元素的元组，第一个元素是拟合后的 y 值数组，第二个元素是拟合多项式的系数
    
    Examples
    --------
    >>> xs = [1, 2, 3, 4, 5]
    >>> ys = [1, 4, 9, 16, 25]
    >>> polynomial_degree = 2
    >>> fit(xs, ys, polynomial_degree)
    (array([ 1.,  4.,  9., 16., 25.]), array([1., 0., 0.]))
    """
    Fit = _np.polynomial.Polynomial.fit(xs, ys, polynomial_degree).convert()
    return Fit(xs), Fit.coef


def interp(x, y, x0, kind="linear"):
    """插值
    Args:
        x (list): x
        y (list): y
        x0 (list): x0
        kind (string, optional): 插值类型。Defaults to 'linear'.
    Returns:
        list: f(x0)
    """
    from scipy.interpolate import interp1d

    return interp1d(x, y, kind=kind, bounds_error=False)(x0)


def find_boundary(x, y, zdata, a, clf=None, axes=None):
    """找到 (x, y, z) 图中的二分类边界，一边 z < a

    默认使用 Gassian 过程分类：clf = GaussianProcessClassifier(1.0 * RBF(1.0))

    Parameters
    ----------
    x : numpy.ndarray
        二维数组，x 方向数据
    y : numpy.ndarray
        二维数组，y 方向数据
    zdata : numpy.ndarray
        二维数组，z 方向数据
    a : real
        分界线
    clf : classifier, optional
        分类器。Defaults to None.
    axes : list, optional
        [x0, x1, y0, y1]. Defaults to None.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        边界的横纵坐标
    
    
    常用的分类器还有：
    
    - 支持向量机线性分类
        clf = sklearn.svm.SVC(kernel="linear", C=0.025)
    
    - 支持向量机分类
        clf = sklearn.svm.SVC(gamma=2, C=1)
    
    - 决策树分类
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=5)
    
    - MLPC 分类
        clf = sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000)
    
    - 高斯朴素贝叶斯分类
        clf = sklearn.naive_bayes.GaussianNB()
    
    - 随机森林分类
        clf = sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    
    - AdaBoost 分类
        clf = sklearn.ensemble.AdaBoostClassifier()
    
    - 二次判别分析算法
        clf = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()

    """
    import matplotlib.pyplot as _plt
    from sklearn.metrics import accuracy_score

    assert _np.all([isinstance(i, _np.ndarray) for i in [x, y, zdata]])
    assert x.ndim == 1 and y.ndim == 1 and zdata.ndim == 2
    assert zdata.shape == (y.size, x.size)
    pts = _np.array([[i, j] for i in x for j in y])
    vls = _np.array([zdata[j, i] for i in range(len(x)) for j in range(len(y))]) < a
    if clf is None:
        from sklearn.svm import SVC

        clf = SVC(gamma="scale", C=10)
    clf.fit(pts, vls)
    pre_y = clf.predict(pts)
    accuracy = accuracy_score(vls, pre_y)
    print(f"classify accuracy: {accuracy:0.2f}")
    if axes is None:
        axes = [x.min(), x.max(), y.min(), y.max()]
    x0s = _np.linspace(axes[0], axes[1], 2000)
    x1s = _np.linspace(axes[2], axes[3], 2000)
    x0, x1 = _np.meshgrid(x0s, x1s)
    X = _np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X)
    y_pred = y_pred.reshape(x0.shape)
    fig = _plt.figure("temp")
    cs = fig.add_subplot(111).contour(x0, x1, y_pred, levels=[0.5])
    _plt.close("temp")
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    return v[:, 0], v[:, 1]


def log_Gauss(mu, sigma2, xlist):
    """生成 xlist 对应的 log高斯分布数值"""
    ylist = []
    for xi in xlist:
        yi = (
            1
            / (2 * _np.pi * sigma2) ** 0.5
            * _np.exp(-((_np.log(abs(xi)) - mu) ** 2) / (2 * sigma2))
            / abs(xi)
            / 2
        )
        ylist.append(yi)
    return ylist


