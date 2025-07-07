# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-01 17:17:48
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-07 23:06:02

#!! linalg 中不要 import linalg 之外的文件

import os as _os
import platform as _platform
import warnings as _warnings

import numpy as _np
import scipy.linalg as _sla
import scipy.sparse as _sparse
import scipy.sparse.linalg as _spla

from typing import Optional, Union

from ...basicfun import (
    create_folder,
    get_free_space,
    load_hdf5,
    save_hdf5,
    logger,
)

__all__ = [
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "eighbetween"
]

# ------------------------------
# 格式转换
# ------------------------------

def _ishamiltonian(H):
    type_name = str(type(H))[:36]
    return True if type_name == "<class 'quspin.operators.hamiltonian" else False

def coo2list(sparmatrix):
    """将 coo 的 row col data 提取出来 储存
    返回 row, col, data, shape
    """
    if isinstance(sparmatrix, _sparse.coo_array) or isinstance(
        sparmatrix, _sparse.coo_matrix
    ):
        H_coo = sparmatrix
    else:
        try:
            H_coo = sparmatrix.static.tocoo(copy=False)
        except:
            try:
                H_coo = sparmatrix.tocoo(copy=False)
            except:
                raise TypeError(f"type {str(type(sparmatrix))} not understood")
    return H_coo.row, H_coo.col, H_coo.data, H_coo.shape[0]


def toarray(H) -> _np.ndarray:
    """hamiltonian 或者 coo 等 转 密矩阵"""
    if isinstance(H, _np.ndarray):
        assert H.ndim == 2
        return H
    else:
        try:
            return H.toarray()
        except:
            from .nbfuc.operations_numba import coo2array
            if _ishamiltonian(H):
                res = H.static.tocoo(copy=False)
            else:
                try:
                    res = H.tocoo(copy=False)
                except:
                    raise TypeError("type not understood")
            Hmat = coo2array(res.row, res.col, res.data, res.shape[0])
            return Hmat

# --------------------------------------------
#                 eigensolve
# --------------------------------------------

_EIG_BACKEND = {
    # backend, isherm, return_vecs
    ("numpy", False, True): _np.linalg.eig,
    ("numpy", True, True): _np.linalg.eigh,
    ("numpy", False, False): _np.linalg.eigvals,
    ("numpy", True, False): _np.linalg.eigvalsh,
    ("scipy", False, True): _sla.eig,
    ("scipy", True, True): _sla.eigh,
    ("scipy", False, False): _sla.eigvals,
    ("scipy", True, False): _sla.eigvalsh,
}


def eigensolve_core(
    H,
    backend,
    path_save_to="EigData/",
    isherm=False,
    file_name="tmp",
    reload=True,
    return_vecs=True,
):
    """return eigenvalue and eigenvector based on different backend"""
    if backend == "matlab":
        path_save_to = create_folder(path_save_to)
        E_file = path_save_to + "E_" + file_name + ".h5"
        psi_file = path_save_to + "psi_" + file_name + ".h5"
        if _os.path.exists(E_file):
            _os.remove(E_file)
        if _os.path.exists(psi_file):
            _os.remove(psi_file)
        if _ishamiltonian(H):
            Hmat = H.static.tocoo(copy=False)
        elif _sparse.issparse(H):
            Hmat = H.tocoo(copy=False)
        elif isinstance(H, _np.ndarray) and H.ndim == 2:
            Hmat = H
        else:
            raise Exception("not support type")
        return matlabeig(
            Hmat,
            path_save_to=path_save_to,
            file_name=file_name,
            reload=reload,
            return_vecs=return_vecs,
        )

    if H.shape[0] > 32766:
        logger.warning("may bad performance")

    Hmat = toarray(H)
    isherm = _np.allclose(Hmat, Hmat.conj().T) if isherm is None else isherm

    isherm_str = "H" if isherm else ""
    returnvec_str = "VALS" if return_vecs else ""

    if backend == "dsyevd":
        assert isherm
        logger.debug(
            f"EIG{returnvec_str}{isherm_str}ing ({backend}) ... (dim={Hmat.shape})"
        )
        if return_vecs:
            return _sla.lapack.dsyevd(Hmat)
        else:
            return _sla.lapack.dsyevd(Hmat, compute_v=0)[0]
        logger.debug("done")

    logger.debug(
        f"EIG{returnvec_str}{isherm_str}ing ({backend}) ... (dim={Hmat.shape})"
    )
    res = _EIG_BACKEND[backend, isherm, return_vecs](Hmat)
    logger.debug("done")

    return res


def _load_eigres(E_file):
    """load eigen result from E_file"""
    res = load_hdf5(E_file, group="", data="/")
    try:
        return res["real"] + 1j * res["imag"]
    except:
        return res["data"]


def _unwrap_save(save, defaultname="EigData"):
    """update path save to based on save option"""
    if save is True:
        # default save
        path_save_to = create_folder(defaultname)
        file_name = "tmp"
        E_file = path_save_to + "E_" + file_name + ".h5"
        psi_file = path_save_to + "psi_" + file_name + ".h5"
        if _os.path.exists(E_file):
            _os.remove(E_file)
        if _os.path.exists(psi_file):
            _os.remove(psi_file)

    elif save is not False:
        # unwrap the save parameter
        if isinstance(save, str):
            path_save_to = create_folder(defaultname)
            file_name = (save,)
        elif isinstance(save, list):
            path_save_to, file_name = save
            path_save_to = create_folder(path_save_to)
        else:
            raise TypeError("save should be bool, string or 2 element list")

        E_file = path_save_to + "E_" + file_name + ".h5"
        psi_file = path_save_to + "psi_" + file_name + ".h5"

    else:
        path_save_to, file_name = None, None
        E_file, psi_file = None, None
    return (
        path_save_to,
        file_name,
        E_file,
        psi_file,
    )


def eigensolve(
    H,
    isherm=None,
    backend=None,
    save=False,
    reload=True,
    return_vecs=True,
    autoblock=False,
    path_save="EigData/"
):
    """
    本征值分解，自动判断 backend，<32766 numpy else matlab>

    非厄密矩阵本征值分解，速度没有明显区别

    val, vec = eigensolve(mat)

    assert _np.isclose(mat, vec @ _np.diag(val) @ vec.T.conj()).all()
    """
    # determine which backend to use
    backend = "auto" if backend is None else backend.lower()
    if backend == "auto":
        if H.shape[0] > 32766:
            backend = "matlab"
        else:
            backend = "numpy"
    backend = backend.lower()
    Hmat = H

    # if use matlab, data has to be saved
    if backend == "matlab" and save is False:
        save = True
        logger.warning("matlab backend has to save result")

    path_save_to, file_name, E_file, psi_file = _unwrap_save(
        save, defaultname=path_save
    )

    # if data exists, just return the data
    if save is not False and save is not True:
        if return_vecs and (_os.path.exists(E_file) or _os.path.exists(psi_file)):
            logger.warning(
                f"data seems available at\n\t{E_file} \nor\n\t{psi_file}\ntrying to load from there. (if recalculate is needed, please remove data or change the file name)"
            )
            return _load_eigres(E_file), _load_eigres(psi_file)
        elif not return_vecs and (_os.path.exists(E_file) or _os.path.exists(psi_file)):
            logger.warning(
                f"data seems available at\n\t{E_file} \nor\n\t{psi_file}\ntrying to load from there. (if recalculate is needed, please remove data or change the file name)"
            )
            return _load_eigres(E_file)

    # check the space is enough to store result
    if path_save_to is not None:
        if backend != "matlab":
            Hmat = toarray(H)
            assert (
                Hmat.nbytes + _np.sqrt(Hmat.nbytes)
            ) / 1024 / 1024 / 1024 * 1.01 < get_free_space(path_save_to)

    if autoblock:
        # if autoblock, use numpy for each block
        Hmat = toarray(H)
        isherm = _np.allclose(Hmat, Hmat.conj().T) if isherm is None else isherm
        assert isherm
        if return_vecs:
            from .nbfuc.eig_modified_numba import _eigh_autoblocked
            dtpye = Hmat.dtype
            res = _eigh_autoblocked(Hmat, sort=True, dtpye=dtpye)
        else:
            from .nbfuc.eig_modified_numba import _eigvalsh_autoblocked
            dtpye = Hmat.dtype
            res = _eigvalsh_autoblocked(Hmat, sort=True, dtpye=dtpye)
        if res is None:
            raise Exception("too large")
    else:
        # if not block
        res = eigensolve_core(
            H=H,
            backend=backend,
            path_save_to=path_save_to,
            isherm=isherm,
            file_name=file_name,
            reload=reload,
            return_vecs=return_vecs,
        )

    # if backend is not matlab, save here
    if save and backend != "matlab":
        if return_vecs:
            save_hdf5(path_save_to + "E_" + file_name, res[0])
            save_hdf5(path_save_to + "psi_" + file_name, res[1])
        else:
            save_hdf5(path_save_to + "E_" + file_name, res)

    if reload:
        return res


# --------------------------------------------
#            partial eigensolve
# --------------------------------------------

from importlib.util import find_spec

SLEPC4PY_FOUND = find_spec("slepc4py")


def maybe_sort_and_project(lk, vk, P, sort=True):
    """sort or project back"""
    if sort:
        sortinds = _np.argsort(lk)
        lk, vk = lk[sortinds], vk[:, sortinds]

    # map eigenvectors out of subspace
    if P is not None:
        vk = P @ vk

    return lk, vk


_DENSE_EIG_METHODS = {
    (True, True): _np.linalg.eigh,
    (True, False): _np.linalg.eigvalsh,
    (False, True): _np.linalg.eig,
    (False, False): _np.linalg.eigvals,
}


def sort_inds(a, method, sigma=None):
    """sort based on different type"""
    _SORT_FUNCS = {
        "LM": lambda a: -abs(a),
        "SM": lambda a: -abs(1 / a),
        "SA": lambda a: a,
        "SR": lambda a: a.real,
        "SI": lambda a: a.imag,
        "LA": lambda a: -a,
        "LR": lambda a: -a.real,
        "LI": lambda a: -a.imag,
        "TM": lambda a: -1 / abs(abs(a) - sigma),
        "TR": lambda a: -1 / abs(a.real - sigma),
        "TI": lambda a: -1 / abs(a.imag - sigma),
    }
    return _np.argsort(_SORT_FUNCS[method.upper()](a))


def eigs_numpy(
    A, k, which=None, return_vecs=True, sigma=None, isherm=True, P=None, sort=True
):
    """Partial eigen-decomposition using numpy's dense linear algebra.

    Parameters
    ----------
    A : array_like
        Operator to partially eigen-decompose.
    k : int
        Number of eigenpairs to return.
    which : str, optional
        Which part of the spectrum to target.
    return_vecs : bool, optional
        Whether to return eigenvectors.
    sigma : None or float, optional
        Target eigenvalue.
    isherm : bool, optional
        Whether `a` is hermitian.
    P : array_like
        Perform the eigensolve in the subspace defined by this projector.
    sort : bool, optional
        Whether to sort reduced list of eigenpairs into ascending order.
    eig_opts
        Settings to pass to numpy.eig... functions.

    Returns
    -------
        lk, (vk): k eigenvalues (and eigenvectors) sorted according to which
    """
    # project into subspace
    if P is not None:
        A = P.conjugate().transpose() @ (A @ P)

    eig_fn = _DENSE_EIG_METHODS[(isherm, return_vecs)]
    isherm_str = "H" if isherm else ""

    if return_vecs:
        # get all eigenpairs
        logger.debug(f"EIG{isherm_str}ing ... (dim={A.shape})")
        lk, vk = eig_fn(A.toarray() if _sparse.issparse(A) else A)
        logger.debug("done")

        # sort and trim according to which k we want
        sk = sort_inds(lk, method=which, sigma=sigma)[:k]
        lk, vk = lk[sk], vk[:, sk]

        # also potentially sort into ascending order
        if sort:
            so = _np.argsort(lk)
            lk, vk = lk[so], vk[:, so]

        # map eigenvectors out of subspace
        if P is not None:
            vk = P @ vk

        return lk, vk

    else:
        # get all eigenvalues
        logger.debug(f"EIG{isherm_str}ing (numpy) ... (dim={A.shape})")
        lk = eig_fn(A.toarray() if _sparse.issparse(A) else A)
        logger.debug("done")

        # sort and trim according to which k we want
        sk = sort_inds(lk, method=which, sigma=sigma)[:k]
        lk = lk[sk]

        # also potentially sort into ascending order
        return _np.sort(lk) if sort else lk


def eigs_scipy(
    A,
    k,
    *,
    which=None,
    return_vecs=True,
    sigma=None,
    isherm=True,
    sort=True,
    P=None,
    tol=None,
    **eigs_opts,
):
    """Returns a few eigenpairs from a possibly sparse hermitian operator

    Parameters
    ----------
    A : array_like, sparse_matrix, LinearOperator or quimb.Lazy
        The operator to solve for.
    k : int
        Number of eigenpairs to return
    which : str, optional
        where in spectrum to take eigenvalues from (see
        :func:`scipy.sparse.linalg.eigsh`).
    return_vecs : bool, optional
        Whether to return the eigenvectors as well.
    sigma : float, optional
        Shift, if targeting interior eigenpairs.
    isherm : bool, optional
        Whether `A` is hermitian.
    P : array_like, sparse_matrix, LinearOperator or quimb.Lazy, optional
        Perform the eigensolve in the subspace defined by this projector.
    sort : bool, optional
        Whether to ensure the eigenvalues are sorted in ascending value.
    backend : None or 'primme', optional
        Which backend to use.
    eigs_opts
        Supplied to :func:`scipy.sparse.linalg.eigsh` or
        :func:`scipy.sparse.linalg.eigs`.

    Returns
    -------
    lk : (k,) array
        The eigenvalues.
    vk : (m, k) array
        Corresponding eigenvectors (if `return_vecs=True`).
    """
    # project into subspace
    if P is not None:
        A = P.conjugate().transpose() @ (A @ P)

    # Options that might get passed that scipy doesn't support
    eigs_opts.pop("EPSType", None)

    # convert certain options for scipy
    settings = {
        "k": k,
        "which": (
            "SA"
            if (which is None) and (sigma is None)
            else "LM"
            if (which is None) and (sigma is not None)
            else
            # For target using shift-invert scipy requires 'LM' ->
            "LM"
            if ("T" in which.upper()) and (sigma is not None)
            else which
        ),
        "sigma": sigma,
        "return_eigenvectors": return_vecs,
        "tol": 0 if tol is None else tol,
    }

    eigs = _spla.eigsh if isherm else _spla.eigs
    isherm_str = "H" if isherm else ""

    if return_vecs:
        logger.debug(f"EIGS{isherm_str}ing (scipy) ... (dim={A.shape})")
        lk, vk = eigs(A, **settings, **eigs_opts)
        logger.debug("done")
        return maybe_sort_and_project(lk, vk, P, sort)
    else:
        logger.debug(f"EIGVALS{isherm_str}ing (scipy)... (dim={A.shape})")
        lk = eigs(A, **settings, **eigs_opts)
        logger.debug("done")
        return _np.sort(lk) if sort else lk


def choose_backend(A, k, int_eps=False):
    """Pick a backend automatically for partial decompositions."""
    # LinOps -> not possible to simply convert to dense or use MPI processes
    A_is_linop = isinstance(A, _spla.LinearOperator)

    # small array or large part of subspace requested
    small_d_big_k = A.shape[0] ** 2 / k < (10000 if int_eps else 2000)

    if small_d_big_k and not A_is_linop:
        return "NUMPY"

    if not _sparse.issparse(A) and int_eps:
        _warnings.warn("shift-invert of dense matrix, back to scipy")
        return "SCIPY"

    # slepc seems faster for sparse, dense and LinearOperators
    if SLEPC4PY_FOUND:
        # only spool up an mpi pool for big sparse matrices though
        if _sparse.issparse(A) and A.nnz > 10000:
            return "SLEPC"

        return "SLEPC-NOMPI"

    return "SCIPY"


def eigensolve_partial(
    A,
    k,
    isherm=False,
    backend=None,
    save=False,
    reload=True,
    return_vecs=True,
    P=None,
    which=None,
    sigma=None,
    ncv=None,
    tol=None,
    v0=None,
    sort=True,
    fallback_to_scipy=False,
    **backend_opts,
):
    """
    部分本征值，本征向量
    """
    if _sparse.issparse(A) or type(A) in [_np.ndarray, _spla.LinearOperator]:
        mat = A
    else:
        try:
            mat = A.tocsr()
        except:
            raise TypeError("type not understood")

    # Choose backend to perform the decompostion
    bkd = "AUTO" if backend is None else backend.upper()
    if bkd == "AUTO":
        bkd = choose_backend(A, k, sigma is not None)

    path_save_to, file_name, E_file, psi_file = _unwrap_save(
        save, defaultname="PartialEigData/"
    )

    # if data exists, just return the data
    if save is not False and save is not True:
        if return_vecs and (_os.path.exists(E_file) or _os.path.exists(psi_file)):
            logger.warning(
                f"data seems available at\n\t{E_file} \nor\n\t{psi_file}\ntrying to load from there. (if recalculate is needed, please remove data or change the file name)"
            )
            return _load_eigres(E_file), _load_eigres(psi_file)
        elif not return_vecs and (_os.path.exists(E_file) or _os.path.exists(psi_file)):
            logger.warning(
                f"data seems available at\n\t{E_file} \nor\n\t{psi_file}\ntrying to load from there. (if recalculate is needed, please remove data or change the file name)"
            )
            return _load_eigres(E_file)

    which = (
        "SA"
        if (which is None) and (sigma is None)
        else "TR"
        if (which is None) and (sigma is not None)
        else which
    )

    if bkd == "NUMPY":
        res = eigs_numpy(
            A=mat,
            k=k,
            which=which,
            return_vecs=return_vecs,
            sigma=sigma,
            isherm=isherm,
            P=P,
            sort=sort,
        )
    else:
        settings = {
            "k": k,
            "which": which,
            "return_vecs": return_vecs,
            "sigma": sigma,
            "isherm": isherm,
            "ncv": ncv,
            "sort": sort,
            "tol": tol,
            "v0": v0,
        }

        if bkd == "SCIPY":
            res = eigs_scipy(mat, **settings, **backend_opts)
        else:
            # try to call quimb
            try:
                import quimb as qu

                if _sparse.issparse(A):
                    mat = _sparse.csr_matrix(mat)
                    assert (
                        _np.max(mat.indptr) < 2147483647
                        and _np.max(mat.indices) < 2147483647
                    )
                    mat.indptr, mat.indices = _np.int32(mat.indptr), _np.int32(
                        mat.indices
                    )

                logger.debug(f"QUIMB {bkd} ... (dim={A.shape})")
                res = qu.eigensystem_partial(
                    mat,
                    backend=backend,
                    fallback_to_scipy=fallback_to_scipy,
                    **settings,
                    **backend_opts,
                )
                logger.debug("done")
            except Exception as e:
                raise e

    # if backend is not matlab, save here
    if save and backend != "matlab":
        if return_vecs:
            save_hdf5(path_save_to + "E_" + file_name, res[0])
            save_hdf5(path_save_to + "psi_" + file_name, res[1])
        else:
            save_hdf5(path_save_to + "E_" + file_name, res)

    if reload:
        return res


# --------------------------------------------
#            wrappered interface
# --------------------------------------------


def eig(
    A: Union[_np.ndarray, _sparse.spmatrix],
    k: int = -1,
    unitary: bool = False,
    backend: Optional[str] = None,
    save: bool = False,
    reload: bool = True,
    autoblock: bool = False,
    *,
    which: Optional[str] = None,
    sigma: Optional[float] =None,
    ncv: Optional[int] = None,
    tol: Optional[int] = None,
    v0: Optional[int] = None,
    sort: bool = True,
    fallback_to_scipy: bool = False,
    P=None,
    **backend_opts,
) -> tuple[_np.ndarray, _np.ndarray]:
    """查找非厄米矩阵的算子的所有或部分特征对。

    Parameters
    ----------
    A : operator
        要分解的算子。
        
    k : int, 可选
        如果为负数，查找所有特征对，否则执行部分特征分解并查找 `k` 对。
        
    backend : {'AUTO', 'NUMPY', 'SCIPY', 'LOBPCG', 'SLEPC', 'SLEPC-NOMPI', 'MATLAB'}, 可选
        使用哪个求解器。('MATLAB' 用于 k=-1 的情况。)
        
    save: bool, str, list, 可选
        如果为 False，不保存。
        如果为 True，按默认路径保存，k=-1 时为 EigData，k>0 时为 PartialEigData。
        如果为 str，该字符串被解释为保存的文件名。
        如果为列表 [path_save_to, file_name]，则路径和文件名都被指定。
        
    reload: bool, 可选
        如果为 False，不返回任何内容。
        
    autoblock : bool, 可选
        如果为 True，自动识别并利用当前基中出现的对称性，通过行和列的置换形成块对角线。（仅适用于 k=-1）
        
    which : {'SA', 'LA', 'LM', 'SM', 'TR'}
        从谱的哪个部分获取特征值（参见 :func:`scipy.sparse.linalg.eigsh`）。
        
    sigma : float, 可选
        目标谱的哪一部分，如果 which 为 None，则意味着 which='TR'。
        
    ncv : int, 可选
        兰索斯向量的数量，可以用来优化速度。
        
    tol : None 或 float
        查找特征值的容差。
        
    v0 : None 或 类 1D-array
        用于迭代的初始向量猜测。
        
    sort : bool, 可选
        是否按升序显式排序特征值。
        
    fallback_to_scipy : bool, 可选
        如果发生错误且未使用 scipy，尝试使用 scipy。
        
    sort : bool, 可选
        是否按升序排序特征对。
        
    P : array_like, sparse_matrix, LinearOperator, 可选
        在此投影算子定义的子空间中执行特征求解。（仅适用于 k=-1）
        
    backend_opts
        提供给后端求解器的选项。
        根据指定的求解器 backend 提供具体的参数
    """
    if k < 0:
        if unitary:
            _, U = eigh(A+A.conj().T)
            return _np.sum(U.conj() * (A @ U), axis=0), U
        return eigensolve(
            H=A,
            isherm=None,
            backend=backend,
            save=save,
            reload=reload,
            return_vecs=True,
            autoblock=autoblock,
        )

    return eigensolve_partial(
        A,
        k=k,
        isherm=False,
        backend=backend,
        save=save,
        reload=reload,
        return_vecs=True,
        which=which,
        sigma=sigma,
        ncv=ncv,
        tol=tol,
        v0=v0,
        sort=sort,
        fallback_to_scipy=fallback_to_scipy,
        P=P,
        **backend_opts,
    )


def eigh(A: Union[_np.ndarray, _sparse.spmatrix],
         k: int = -1,
         backend: Optional[str] = None,
         save: bool = False, 
         reload: bool = True,
         autoblock: bool = False,
         *,
         which: str='SA',
         sigma: Optional[float] = None,
         ncv: Optional[int] = None,
         tol: Optional[float] = None,
         v0: Optional[_np.ndarray] = None,
         sort: bool = True,
         fallback_to_scipy: bool = False,
         P: Optional[_np.ndarray] = None,
         **backend_opts
         ) -> tuple[_np.ndarray, _np.ndarray]:
    """计算本征值厄米矩阵的算子的所有或部分特征对，与 `eig` 参数相同"""
    if k < 0:
        return eigensolve(
            H=A,
            isherm=True,
            backend=backend,
            save=save,
            reload=reload,
            return_vecs=True,
            autoblock=autoblock,
        )

    return eigensolve_partial(A, k=k, isherm=True, backend=backend, save=save, reload=reload, return_vecs=True, which=which, sigma=sigma, ncv=ncv, tol=tol, v0=v0, sort=sort, fallback_to_scipy=fallback_to_scipy, P=P, **backend_opts,)



def eigvals(
    A: Union[_np.ndarray, _sparse.spmatrix],
    k: int = -1,
    backend: Optional[str] = None,
    save: bool = False,
    reload: bool = True,
    autoblock: bool = False,
    unitary: bool = False,
    *,
    which: Optional[str] = None,
    sigma: Optional[float] = None,
    ncv: Optional[int] = None,
    tol: Optional[float] = None,
    v0: Optional[_np.ndarray] = None,
    sort: bool = True,
    fallback_to_scipy: bool = False,
    P: Optional[_np.ndarray] = None,
    **backend_opts,
) -> _np.ndarray:
    """查找非厄米矩阵的算子的所有或部分特征值。与 `eig` 的参数基本相同。"""
    if k < 0:
        if unitary:
            # 幺正矩阵会快拆成两个厄密矩阵会快很多
            res1 = eigvalsh((A+A.conj().T))/2
            res1 = res1[_np.argsort(1/_np.abs(res1))]
            res2 = eigvalsh((A-A.conj().T)/1j)/2
            res2 = res2[_np.argsort(1/_np.abs(res2))[::-1]]
            return _np.array(res1) + 1j * _np.array(res2)
        return eigensolve(
            H=A,
            isherm=None,
            backend=backend,
            save=save,
            reload=reload,
            return_vecs=False,
            autoblock=autoblock,
        )


    return eigensolve_partial(
        A,
        k=k,
        isherm=False,
        backend=backend,
        save=save,
        reload=reload,
        return_vecs=False,
        which=which,
        sigma=sigma,
        ncv=ncv,
        tol=tol,
        v0=v0,
        sort=sort,
        fallback_to_scipy=fallback_to_scipy,
        P=P,
        **backend_opts,
    )


def eigvalsh(
    A: Union[_np.ndarray, _sparse.spmatrix],
    k: int = -1,
    backend: Optional['str'] = None,
    save: bool = False,
    reload: bool = True,
    autoblock: bool = False,
    *,
    which: Optional[str] = None,
    sigma: Optional[float] = None,
    ncv: Optional[int] = None,
    tol: Optional[float] = None,
    v0: Optional[_np.ndarray] = None,
    sort: bool = True,
    fallback_to_scipy: bool = False,
    P: Optional[_np.ndarray] = None,
    **backend_opts,
) -> _np.ndarray:
    """查找厄米矩阵的算子的所有或部分特征对。与 `eigvals` 参数基本相同。"""
    if k < 0:
        return eigensolve(
            H=A,
            isherm=True,
            backend=backend,
            save=save,
            reload=reload,
            return_vecs=False,
            autoblock=autoblock,
            **backend_opts
        )

    return eigensolve_partial(
        A,
        k=k,
        isherm=True,
        backend=backend,
        save=save,
        reload=reload,
        return_vecs=False,
        which=which,
        sigma=sigma,
        ncv=ncv,
        tol=tol,
        v0=v0,
        sort=sort,
        fallback_to_scipy=fallback_to_scipy,
        P=P,
        **backend_opts,
    )


######################
# API to matlab
######################


def matlabeig(
    H, path_save_to="EigData/", file_name=None, reload=False, return_vecs=True
):
    """调用 matlab eig
    
    Parameters:
    -----------
    H : sparse.array, numpy.ndarray
        矩阵，最好是稀疏的
    path_save_to : str, optional
        保存路径. Defaults to "EigData/".
    file_name : str, optional
        保存文件名. Defaults to None.
    reload : bool, optional
        是否重新加载回 python. Defaults to False.
    return_vecs : bool, optional
        是否计算本征向量. Defaults to True.
    """
    spaceRequired = matlabeig_pre(H=H, path_save_to=path_save_to, file_name=file_name)
    assert spaceRequired < get_free_space(path_save_to)
    matlabeig_list(
        path_save_to=path_save_to, file_names=[file_name], return_vecs=return_vecs
    )
    if reload:
        if return_vecs:
            return _load_eigres(path_save_to + "E_" + file_name + ".h5"), _load_eigres(
                path_save_to + "psi_" + file_name + ".h5"
            )
        else:
            return _load_eigres(path_save_to + "E_" + file_name + ".h5")


def matlabeigvals(H, path_save_to="EigData/", file_name=None, reload=False):
    return matlabeig(
        H=H,
        path_save_to=path_save_to,
        file_name=file_name,
        reload=reload,
        return_vecs=False,
    )


def matlabeig_pre(H, file_name=None, path_save_to="EigData/"):
    """如果手动设置，这里的tempPath应当选择path_save_to下的子文件夹mat4py"""
    import scipy.io as _sio

    path_save_to = create_folder(path_save_to)
    tempPath = path_save_to + "mat4py/"
    tempPath = create_folder(tempPath)
    if file_name is None:
        file_name = "tmp"
    if isinstance(H, _np.ndarray):
        spaceRequired = (H.nbytes + _np.sqrt(H.nbytes)) / 1024 / 1024 / 1024 * 1.01
        assert spaceRequired < get_free_space(tempPath)
        # save_hdf5(tempPath + file_name + ".h5", "", {"H": H, "dim": H.shape[0]})
        try:
            _sio.savemat(tempPath + file_name + ".mat", {"H": H, "dim": H.shape[0]})
        except Exception as e:
            # 删除已创建的 .mat 文件
            mat_file = tempPath + file_name + ".mat"
            if _os.path.exists(mat_file):
                _os.remove(mat_file)
            # 改为保存为 .h5 文件
            save_hdf5(tempPath + file_name + ".h5", "", {"H": H, "dim": H.shape[0]})
    else:
        row, col, data, dim = coo2list(H)
        assert (
            data.nbytes + 2 * row.nbytes
        ) / 1024 / 1024 / 1024 * 1.01 < get_free_space(tempPath)
        try:
            _sio.savemat(
                tempPath + file_name + ".mat",
                {"row": row, "col": col, "data": data, "dim": dim},
            )
        except Exception as e:
            # 删除已创建的 .mat 文件
            mat_file = tempPath + file_name + ".mat"
            if _os.path.exists(mat_file):
                _os.remove(mat_file)
            # 改为保存为 .h5 文件
            save_hdf5(
                tempPath + file_name + ".h5",
                "",
                {"row": row, "col": col, "data": data, "dim": dim},
            )
        spaceRequired = data.itemsize * dim * (dim + 1) / 1024 / 1024 / 1024
    return spaceRequired


def __get_last_line(filename):
    """
    get last line of a file
    :param filename: file name
    :return: last line or None for empty file
    """
    try:
        filesize = _os.path.getsize(filename)
        if filesize == 0:
            return None
        else:
            with open(filename, "rb") as fp:  # to use seek from end, must use mode 'rb'
                offset = -8  # initialize offset
                while -offset < filesize:  # offset cannot exceed file size
                    fp.seek(
                        offset, 2
                    )  # read # offset chars from eof(represent by number '2')
                    lines = fp.readlines()  # read from fp to eof
                    if len(lines) >= 2:  # if contains at least 2 lines
                        return lines[-1]  # then last line is totally included
                    else:
                        offset *= 2  # enlarge offset
                fp.seek(0)
                lines = fp.readlines()
                return lines[-1]
    except FileNotFoundError:
        print(filename + " not found!")
        return None


def matlabeig_list(path_save_to="EigData/", file_names=None, return_vecs=True, usegpu=0):
    path_save_to = create_folder(path_save_to)
    tempPath = path_save_to + "mat4py/"
    if file_names is None:
        file_names = ["tmp"]
    para = "'" + "','".join(file_names) + "'"
    curfilename = _os.path.dirname(__file__)

    # create a temp script file to avoid long command line
    script_file = _os.path.join(tempPath, "matlab_script.m")
    with open(script_file, "w") as f:
        f.write(f"cd '{curfilename}/matlab4py';\n")
        f.write("try\n")
        if return_vecs:
            f.write(f"    matlabeig('{tempPath}', '{path_save_to}', '{usegpu}', {para});\n")
        else:
            f.write(f"    matlabeigvals('{tempPath}', '{path_save_to}', '{usegpu}', {para});\n")
        f.write("catch ME\n")
        f.write("    fprintf('%s\\n', string(getReport(ME, 'extended', 'hyperlinks', 'off')));\n")
        f.write("end\n")
        f.write("quit;\n")
    
    # run command in terminal
    command = "matlab "
    if _platform.system() == "Windows":
        command += f"-nosplash -wait -logfile {tempPath}outfile.log -r \"run('{script_file}')\""
    else:
        command += f"-nosplash -nodisplay -r \"run('{script_file}')\" >> {tempPath}outfile.log 2>&1"

    logger.warning(f"change to matlab by command:\n {command}")
    _os.system(command)
    lastline = __get_last_line(rf"{tempPath}outfile.log")
    if lastline != b"finished":
        raise Exception(f"error, please check {tempPath}outfile.log")


def matlabeigvals_list(path_save_to="EigData/", file_names=None, path_load_from=None):
    return matlabeig_list(
        path_save_to=path_save_to,
        file_names=file_names,
        return_vecs=False,
    )


####################
# sparse eig
####################


def eighbetween(
    H,
    subset_by_value=None,
    subset_by_index=None,
    value_number=None,
    driver="evr",
    return_vecs=True,
):
    """
    计算 H 中的部分谱
    可以指定 index 范围，如 subset_by_index = (10, 500) 指第 10 个到第 499 个本征态
    可以指定 value 范围，如 subset_by_value = (-1, 1) 指本征值在 -1 到 1 之间的所有本征态
        
    可以指定从 value 开始某个范围的本征态
        - 如 value_number = ("up", 0, 3) 指大于 0 的三个最小本征态
        - 如 value_number = ("down", 0, 3) 指小于 0 的三个最小本征态
        - 如 value_number = ("between", 0, 6) 指 0 附近的 6 个本征态
    
    前两者是用 eigh 计算，需要稠密矩阵，
    最后是用 eigsh 计算，但要计算矩阵的逆算符
    最后一个需要
    
    """
    if (
        subset_by_index != None and subset_by_value == None and value_number == None
    ) or (subset_by_index == None and subset_by_value != None and value_number == None):
        Hmat = toarray(H)
        return _sla.eigh(
            Hmat,
            subset_by_value=subset_by_value,
            subset_by_index=subset_by_index,
            driver=driver,
            eigvals_only=not return_vecs,
        )
    elif subset_by_index == None and subset_by_index == None and value_number != None:
        direct, value, k = value_number
        if _sparse.issparse(H) or type(H) in [_np.ndarray, _spla.LinearOperator]:
            Hmat = H
        else:
            try:
                Hmat = H.tocsr()
            except:
                raise TypeError("type not understood")
        if direct == "up":
            return _spla.eigsh(
                Hmat,
                k,
                sigma=value,
                which="LA",
                return_eigenvectors=return_vecs,
            )
        elif direct == "down":
            return _spla.eigsh(
                Hmat,
                k,
                sigma=value,
                which="SA",
                return_eigenvectors=return_vecs,
            )
        elif direct == "between":
            assert k > 1
            return _spla.eigsh(
                Hmat,
                k,
                sigma=value,
                which="BE",
                return_eigenvectors=return_vecs,
            )
        else:
            raise ValueError("direction not understood")
