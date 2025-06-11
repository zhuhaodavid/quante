# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 16:50:19
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-11 20:19:19

import numpy as _np
import scipy.sparse as _sparse

from typing import Union, Literal
number = Union[int, float, complex]

__all__ = [
    "random_matrix",
    "random_orthorgonal_matrix_close_I",
    "random_unitary_matrix_close_I",
    "random_sparse_matrix",
]

# @njit
def _random_simple_matrix(dim, seed=None):
    if seed is not None:
        _np.random.seed(seed)
    real_part = _np.random.randn(dim, dim)
    imag_part = _np.random.randn(dim, dim)
    return real_part + 1.0j * imag_part

# @njit
def _random_simple_real_matrix(dim, seed):
    if seed is not None:
        _np.random.seed(seed)
    return _np.random.randn(dim, dim)

# @njit
def _random_symmetric_matrix_goe(dim, seed):
    real_matrix = _random_simple_real_matrix(dim, seed)
    return 0.5 * (real_matrix + real_matrix.T)

# @njit
def _random_hermition_matrix_gue(dim, seed):
    complex_matrix = _random_simple_matrix(dim, seed)
    return (complex_matrix + complex_matrix.T.conj()) * 0.5

# @njit
def _random_hermition_matrix_gse(dim, seed):
    """Class AII: Gaussian Symplectic Ensemble
    (N x N complex Hermitian with symplectic symmetry)"""
    assert dim % 2 == 0, "Dimension must be even for GSE."
    N = dim // 2
    A = _random_hermition_matrix_gue(N, seed)  # Complex part
    B = _random_simple_matrix(N, seed)  # Symplectic part
    B = (B - B.T) / 2  # Skew-symmetric
    upper = _np.hstack((A, B))
    lower = _np.hstack((-B.conj(), A.conj()))
    return _np.vstack((upper, lower))

# @njit
def _random_unitary_matrix_cue(dim, seed):
    # rng = _np.random.default_rng(seed=seed)
    # A = rng.standard_normal((dim, dim)) + 1.j * rng.standard_normal((dim, dim))
    A = _random_simple_matrix(dim, seed)
    Q, R = _np.linalg.qr(A)
    # Q-R is not unique; to make it unique ensure that the diagonal of R is positive
    # Q' = Q*L; R' = L^{-1} *R, where L = diag(phase(diagonal(R)))
    L = _np.array([R[i, i] for i in range(R.shape[0])])
    L[_np.abs(L) < 1.e-15] = 1.
    Q *= L / _np.abs(L)
    return Q

# @njit
def _randomize_orthogonal_coe(dim, seed):
    U = _random_unitary_matrix_cue(dim, seed)
    U_contiguous = _np.ascontiguousarray(U)
    return _np.dot(U_contiguous.T, U_contiguous)

# @njit
def _random_orthogonal_matrix_cre(dim, seed):
    A = _random_simple_real_matrix(dim, seed)
    Q, R = _np.linalg.qr(A)
    # Q-R is not unique; to make it unique ensure that the diagonal of R is positive
    # Q' = Q*L; R' = L^{-1} *R, where L = diag(phase(diagonal(R)))
    L = _np.array([R[i, i] for i in range(R.shape[0])])
    Q *= _np.sign(L)
    return Q

# @njit
def _random_singular_matrix(dim, seed=None):
    if seed is not None:
        _np.random.seed(seed)
    n1 = _np.random.randint(low=0, high=2, size=dim - 1)
    while _np.all(n1 == 0):
        n1 = _np.random.randint(low=0, high=2, size=dim - 1)
    n = _np.random.standard_normal(size=dim)
    for i in range(dim - 1):
        if n1[i] == 1:
            n[i + 1] = n[i]
    a = _np.diag(n) + _np.diag(n1, 1)
    u = _np.random.standard_normal((dim, dim))
    return _np.linalg.inv(u) @ a @ u

# @njit
def _random_normal_matrix(dim, seed=None):
    v = _random_simple_matrix(dim, seed)
    u = _random_unitary_matrix_cue(dim, seed=seed)
    u_contiguous = _np.ascontiguousarray(u)
    v_contiguous = _np.ascontiguousarray(_np.diag(v))
    return u_contiguous @ v_contiguous @ u_contiguous.conj().T

# @njit
def _random_noninv_matrix(dim, seed=None):
    if seed is not None:
        _np.random.seed(seed)
    n = _np.random.randint(1, dim)
    v = _np.random.standard_normal(dim)
    poslis = list(range(n))
    for _ in range(n):
        v[poslis.pop(_np.random.randint(len(poslis)))] = 0
    u = _np.random.standard_normal((dim, dim))
    u_contiguous = _np.ascontiguousarray(u)
    v_contiguous = _np.ascontiguousarray(_np.diag(v))
    return _np.linalg.inv(u_contiguous) @ v_contiguous @ u_contiguous

# @njit
def _random_real_eigen_matrix(dim, seed):
    v = _random_simple_real_matrix(dim, seed)
    u = _random_simple_matrix(dim, seed)
    return (_np.linalg.inv(u) * v) @ u

# @njit
def _random_positive_matrix(dim, seed):
    mat = _random_simple_matrix(dim, seed)
    return mat @ mat.conjugate().transpose()

# @njit
def _random_density_matrix(dim, seed):
    res = _random_positive_matrix(dim=dim, seed=seed)
    trace = _np.sum(_np.diag(res))
    res[:] /= trace
    return res

type_to_function = {
    "GinUE": _random_simple_matrix,
    "GinOE": _random_simple_real_matrix,
    "GOE": _random_symmetric_matrix_goe,
    "GUE": _random_hermition_matrix_gue,
    "GSE": _random_hermition_matrix_gse,
    "CUE": _random_unitary_matrix_cue,
    "COE": _randomize_orthogonal_coe,
    "CRE": _random_orthogonal_matrix_cre,
    "singular": _random_singular_matrix,
    "normal": _random_normal_matrix,
    "noninv": _random_noninv_matrix,
    "realeig": _random_real_eigen_matrix,
    "positive": _random_positive_matrix,
    "rho": _random_density_matrix,
}


def random_matrix(
    dim, 
    mtype: Literal['GinUE', 'GinOE', 'GOE', 'GUE', 'GSE', 'CUE', 'COE', 'CRE',
                'singular', 'normal', 'noninv', 'realeig', 'positive', 'rho'] = "GinUE",
    seed = None
):
    r"""generate a random matrix of given type and dimension.

    Parameters
    ----------
    dim : int
        The dimension of the square matrix to be generated.
    type : str, optional
        The type of the random matrix to be generated. Default is "GinUE".
        - "GinUE/simple" -> Each matrix element is drawn from a complex Gaussian distribution.
        - "GinOE/simple_real" -> Each matrix element is drawn from a real Gaussian distribution.
        - "GOE"/"symm" -> Real symmetric matrix.
        - "GUE"/"herm" -> Hermitian matrix.
        - "CUE"/"unit" -> Unitary matrix.
        - "COE"/"orth" -> Orthogonal matrix (COE).
        - "CRE" -> Orthogonal matrix (CRE).
        - "positive" -> Positive definite matrix.
        - "normal" -> Normal matrix.
        - "noninv" -> Non-invertible matrix.
        - "realeig" -> Matrix with real eigenvalues.
        - "singular" -> Singular matrix.
        - "rho" -> Density matrix.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    
        
    Random matrix ensembles
    -----------------------
    =========== ======================== ======================= ================== ===========
    ensemble    drawn from               measure                 invariant under    beta
    =========== ======================== ======================= ================== ===========
    GOE         real, symmetric          ``~ exp(-n/4 tr(H^2))`` orthogonal O       1
    ----------- ------------------------ ----------------------- ------------------ -----------
    GUE         hermitian                ``~ exp(-n/2 tr(H^2))`` unitary U          2
    ----------- ------------------------ ----------------------- ------------------ -----------
    CRE         O(n)                     Haar                    orthogonal O       /
    ----------- ------------------------ ----------------------- ------------------ -----------
    COE         U in U(n) with U = U^T   Haar                    orthogonal O       1
    ----------- ------------------------ ----------------------- ------------------ -----------
    CUE         U(n)                     Haar                    unitary U          2
    ----------- ------------------------ ----------------------- ------------------ -----------
    O_close_1   O(n)                     ?                       /                  /
    ----------- ------------------------ ----------------------- ------------------ -----------
    U_close_1   U(n)                     ?                       /                  /
    =========== ======================== ======================= ================== ===========

    tenfold symmetry classification
    ------------------------------
    - TRS - Time Reversal Symmetry,
        .. math::
            T H^* T^{-1} = H,  T T^* = \pm 1
    
    - PHS - Particle-Hole Symmetry,
        .. math::
            C H^T C^{-1} = - H,  C C^* = \pm 1
    
    - CS - Chiral symmetry,
        .. math::
            S H S^{-1} = - H,  S^2 = + 1

    ========= ======= ======= ======= ================
    AZ class  TRS     PHS     CS      Example Ensemble
    ========= ======= ======= ======= ================
    A         0       0       0       GUE
    --------- ------- ------- ------- ----------------
    AI        +1      0       0       GOE
    --------- ------- ------- ------- ----------------
    AII       -1      0       0       GSE
    --------- ------- ------- ------- ----------------
    AIII      0       0       1       Chiral unitary
    --------- ------- ------- ------- ----------------
    BDI       +1      +1      1       Chiral orthogonal
    --------- ------- ------- ------- ----------------
    CII       -1      -1      1       Chiral symplectic
    --------- ------- ------- ------- ----------------
    D         0       +1      0       BdG
    --------- ------- ------- ------- ----------------
    DIII      -1      +1      1       BdG
    --------- ------- ------- ------- ----------------
    C         0       -1      0       BdG
    --------- ------- ------- ------- ----------------
    CI        +1      -1      1       BdG
    ========= ======= ======= ======= ================
 
    Notes
    -----
    
    - classification of random matrices:
        - simple matrix: can be diagonalized by similarity transformation.
        - normal matrix: can be diagonalized by unitary transformation.
        (H . H^† = H^† . H)
        - non-normal matrix: can be diagonalized by similarity transformation, but not by unitary transformation.
        - non-simple matrix: cannot be diagonalized by similarity transformation, only can be transformed to Jordan form.
    
    .. code-block:: text
        +-------------------+--------------------+
        |                   |  Normal matrix     |
        |   simple matrix   +--------------------+
        |                   |  Non-normal matrix |
        +-------------------+--------------------+
        | Non-simple matrix |                    |
        +-------------------+--------------------+

    - eigenvalue,
        - there are always eigenvalues for matrices, even if they are not simple matrices.
        - if the eigenvalues of a normal matrix are real, then the normal matrix must be hermitian.
   
    - random ensembles of random matrices:  
    
          
    - basic properties of random matrices:
        - simple matrices (complex or real): with probability 1, they can be diagonalized by similarity 
        transformation; with probability 1, they cannot be diagonalized by unitary transformation; with 
        probability 1, they are invertible.
        - normal matrices (hermitian, symmetric, unitary, real orthogonal): they can always be diagonalized
        by similarity transformation; they can always be diagonalized by unitary transformation; with
        probability 1, they are invertible.
        - non-normal matrices: they can always be diagonalized by similarity transformation; with probability 1,
        they are invertible.
        - non-simple matrices: they cannot be diagonalized by similarity transformation; they can only be
        transformed to Jordan form.
        - singular matrices: they cannot be diagonalized by similarity transformation (the inverse matrix of the
        matrix used to diagonalize the matrix diverges); they cannot be diagonalized by unitary transformation;
        with probability 1, they are invertible.
        - non-invertible matrices: they are not invertible or the inverse matrix diverges; with probability 1,
        they can be diagonalized by similarity transformation; with probability 1, they cannot be diagonalized
        by unitary transformation.
        - real eigenvalue single matrix: with probability 1, they can be diagonalized by similarity transformation;
        with probability 1, they cannot be diagonalized by unitary transformation; with probability 1, they are
        invertible.
    
    Methods for generating random numbers can be found at:
    https://numpy.org/doc/stable/reference/random/generator.html#random-matrix-generation

    """
    mtype = mtype.replace("simple", "GinUE")
    mtype = mtype.replace("simple_real", "GinOE")
    mtype = mtype.replace("herm", "GUE")
    mtype = mtype.replace("unit", "CUE")
    mtype = mtype.replace("symm", "GOE")
    mtype = mtype.replace("orth", "COE")
    if mtype in type_to_function:
        return type_to_function[mtype](dim, seed=seed)
    else:
        raise ValueError(f"Unknown type '{mtype}'.")


def random_orthorgonal_matrix_close_I(dim, a=0.01, seed=None):
    r"""返回一个接近单位矩阵的随机正交矩阵。

    参数 a 决定结果与单位矩阵的接近程度；
    
    当 :math:`a \rightarrow 0` 时，:math:`<|O-E|>_a = 0``（其中 `E` 是单位矩阵）。
    """
    A = _random_symmetric_matrix_goe(dim, seed=seed) / (2. * dim)**0.5  # scale such that eigenvalues are in [-1, 1]
    E = _np.eye(dim)
    Q, R = _np.linalg.qr(E + a * A)
    L = _np.diagonal(R)  # make QR decomposition unique & ensure Q is close to one for small `a`
    Q *= _np.sign(L)
    return Q


def random_unitary_matrix_close_I(dim, a=0.01, seed=None):
    r"""返回一个接近单位矩阵的随机正交矩阵。

    接近单位矩阵的正交矩阵（对于小的 `a`）。
    特征值是独立同分布的，形式为 ``exp(1.j*a*x)``，其中 `x` 在 [-1, 1] 区间内均匀分布。
    """
    U = _random_unitary_matrix_cue(dim, seed=seed)
    E = _np.exp(1.j * a * (_np.random.rand(dim) * 2. - 1.))
    return _np.dot(U * E, U.T.conj())


def random_sparse_matrix(dim, density=0.1, seed=None):
    """生成一个稀疏矩阵
    """
    rng = _np.random.default_rng(seed=seed)
    nnz = round(density * dim * dim)
    ijs = rng.choice(range(0, dim**2), size=nnz, replace=False)
    i, j = _np.divmod(ijs, dim)
    data = rng.standard_normal(nnz) + 1.j * rng.standard_normal(nnz)
    return _sparse.coo_matrix((data, (i, j)), shape=(dim, dim)).asformat("csr")

