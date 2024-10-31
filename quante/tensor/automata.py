# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-12-04 14:38:49
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 15:17:33

import numpy as _np
from ..generate.matrix import pauli_matrix
from .mps import MPS


__all__ = [
    "automata_mpo",
    "automata_mps",
    "contract",
    "get_sparse_matrix",
]


def _remove_full_zero(
    coefficients: list[float], basiss: list[str]
) -> tuple[bool, list[float], list[str], float]:
    """
    Example:
        Input: coefficients = [0.4, 0.5, 0.6, 0.7], basiss = ["00", "01", "10", "11"]
        Output: (True, [0.5, 0.6, 0.7], ["01", "10", "11"], 0.4)
    Note:
        The full-zero basis, if present, is expected to be at the first position.
    """
    N = len(basiss[0])
    full_zero = "0" * N
    full_zero_coe = 0.0
    if basiss[0] == full_zero:
        full_zero_coe = coefficients[0]
        coefficients = coefficients[1:]
        basiss = basiss[1:]
        has_full_zero = True
    else:
        has_full_zero = False
    return has_full_zero, coefficients, basiss, full_zero_coe


def _get_Qmat(i: int, left: int, right: int, Qmat: _np.ndarray, N: int) -> _np.ndarray:
    """
    Initial Qmat:
                       ---> N site
                 [[1,| 1, 1, 1, ...],
                   --|---------------
               |  [1,| 0, 0, 0, ...],
               v     |    .
               n     |    .
             basis   |    .
                  [1,| 0, 0, 0, ...],
                  [1,| 0, 0, 0, ...]]

    # Each element of row is dependent of
    # its position, previous element and the max number of this column.
    """
    for j in range(N):  # j-th site of basis
        k = j + 1  # k-th column of Qmat
        max_val_in_col = _np.max(
            Qmat[:, k]
        )  # Optimize by calculating once per iteration
        if left == right:
            Qmat[i, k] = 1 if j < left else -1
        elif j < left:
            Qmat[i, k] = 1
        elif j == left:
            Qmat[i, k] = max_val_in_col + 1
        elif left < j < right:
            Qmat[i, k] = (
                Qmat[i, k - 1]
                if max_val_in_col < Qmat[i, k - 1]
                else max_val_in_col + 1
            )
        elif j >= right:
            Qmat[i, k] = -1
    return Qmat


def _finalize_Qmat(Qmat: _np.ndarray) -> _np.ndarray:
    """
    Example:
        Before: Qmat =  [[1,  1,  1],    After: Qmat = [[1, 2, 3],
                         [1,  2, -1],                   [1, 3, 3]]
                         [1, -1, -1]]
    Where D = 2 + 1 = 5
    """
    D = _np.max(Qmat) + 1  # the max bond dimension
    Qmat[Qmat == -1] = D
    Qmat[:, -1] = 1
    Qmat = Qmat[1:, :]
    return Qmat


def _convert_to_real_if_possible(Ws) -> list[_np.ndarray]:
    if all(
        _np.all(_np.imag(W) == 0) for W in Ws
    ):  # todo 还能再修改 np.real_if_close()??
        Ws = [_np.real(W).astype(_np.float64) for W in Ws]
    return Ws


def automata_mps(coefficients: list[float], basiss: list[str], d: int = 2) -> MPS:
    """
    Args: coefficients, basiss, d (Defaults to 2)
    Returns: Class: MPS

    Basis could only to be string such as "01101..."
    Input:
        coefficients, basiss = [np.sqrt(1/N)]*N, state_w_str(N)   or
        coefficients, basiss = vector_to_spin_bsiss(vector)
    """ 
    up, down = pauli_matrix("u")[:, -1], pauli_matrix("d")[:, -1]  # (2,1) -> (2,)

    # * initialize each tensor of MPS
    has_full_zero, coefficients, basiss, full_zero_coe = _remove_full_zero(
        coefficients, basiss
    )
    N, n = len(basiss[0]), len(basiss)  # numbers of site N, numbers of basis n
    Qmat = _np.zeros((n + 1, N + 1), dtype=_np.int64)
    Qmat[0, :], Qmat[:, 0] = 1, 1
    for i in range(n):
        basis = basiss[i]
        left, right = basis.find("1"), basis.rfind("1")
        assert left != -1, "Need get rid of the full zero state first"
        Qmat = _get_Qmat(
            i + 1, left, right, Qmat, N
        )  # i-th basis corresponds to (i+1)-th row of Qmat
    Qmat = _finalize_Qmat(Qmat)
    # Qmat2wf(Qmat, coefficients, basiss)
    Q = [_np.max(Qmat[:, i]) for i in range(N + 1)]
    Ws = [_np.zeros((Q[i], d, Q[i + 1]), dtype=_np.complex128) for i in range(N)]

    # * recode position of the first not "1" element
    coe_positions = [
        _np.argmax(row != 1) if _np.any(row != 1) else len(row) - 1 for row in Qmat
    ]

    # * write element of 3-order tensor of mps
    for i, basis in enumerate(basiss):
        Qrow = Qmat[i, :]
        for j, spin in enumerate(basis):
            coefficient = coefficients[i] if j == coe_positions[i] - 1 else 1.0
            Ws[j][Qrow[j] - 1, :, Qrow[j + 1] - 1] = (
                coefficient * up if spin == "0" else coefficient * down
            )
            # "-1" because 1x1 matrix only has [0,0] element
    Ws[-1][0, :, 0] = (
        Ws[-1][0, :, 0] + full_zero_coe * up
        if has_full_zero is True
        else Ws[-1][0, :, 0]
    )
    Ws = _convert_to_real_if_possible(Ws)
    return MPS([_np.array([1.0])] * (N + 1), Ws, is_canonical=False)


def _fill_Id_in_local_hamiltonian(
    N: int, hlocals: list[str], positions: list[tuple]
) -> list[str]:
    """
    Example:
        Input:
            N = 4
            hlocals = ['xx', 'yy', 'zz']
            positions = [(1, 2), (0, 1), (2, 3)]
        Output:
            ['IxxI', 'yyII', 'IIzz']
    """
    hlocals_Id = [None] * len(hlocals)
    for i, (hlocal, position) in enumerate(zip(hlocals, positions)):
        temp = ["I" for _ in range(N)]
        for j, index in enumerate(position):
            temp[index] = hlocal[j]
        hlocals_Id[i] = "".join(temp)
    return hlocals_Id

def automata_mpo(
    N: int,
    hlocals: str,
    positions: tuple,
    coefficients: float,
    d: int = 2,
    pauli: int = True,
    local_matrix_function=None,
    dtype=None
) -> list[_np.ndarray]:
    """
    Basis could only to be string such as "01101..."

    Example:
    >>> L = 10
    >>> ham = op.heisenberg_operator(L)
    >>> automata_mpo(L, *ham.split_data())
    """
    coefficients = _np.real_if_close(coefficients)
    if dtype is None:
        dtype = coefficients.dtype
        
    if local_matrix_function is None:
        local_matrix_function = lambda x: pauli_matrix(x.upper() if x in ['x', 'y', 'z'] else x) if pauli else pauli_matrix(x.upper() if x in ['X', 'Y', 'Z'] else x)
    
    # * rewrite hlocals str to be full str
    hlocals = _fill_Id_in_local_hamiltonian(N, hlocals, positions)

    # * initialize each tensor in mpo
    n = len(hlocals)  # n: local hamitonian h_{i,j}, N: site of each local hamitonian
    Qmat = _np.zeros((n + 1, N + 1), dtype=_np.int64)
    Qmat[0, :], Qmat[:, 0] = 1, 1
    for i, position in enumerate(positions):
        left, right = position[0], position[-1]
        Qmat = _get_Qmat(i + 1, left, right, Qmat, N)
    Qmat = _finalize_Qmat(Qmat)
    # Qmat2wf(Qmat, coefficients, hlocals)  # 图示
    Q = [_np.max(Qmat[:, i]) for i in range(N + 1)]
    mpo = [_np.zeros((Q[i], d, d, Q[i + 1]), dtype=dtype) for i in range(N)]

    # mpo_sign = [_np.zeros((Q[i], Q[i + 1]), dtype=object) for i in range(N)]

    # * write element of 4-order of mpo
    for i, (hlocal, position) in enumerate(zip(hlocals, positions)):
        Qrow = Qmat[i, :]
        for j, operator in enumerate(hlocal):
            coefficient = coefficients[i] if j == position[0] else 1.0
            operator_mat = local_matrix_function(operator)

            self_add = True
            if Qrow[j] == Qrow[j + 1] == 1 and j != N - 1:
                self_add = False
            if Qrow[j] == _np.max(Q):
                self_add = False

            if self_add:
                mpo[j][Qrow[j] - 1, :, :, Qrow[j + 1] - 1] += coefficient * operator_mat
                # mpo_sign[j][Qrow[j] - 1, Qrow[j + 1] - 1] = (
                #     str(mpo_sign[j][Qrow[j] - 1, Qrow[j + 1] - 1])
                #     + "+"
                #     + str(coefficient)
                #     + operator
                # )
            else:
                mpo[j][Qrow[j] - 1, :, :, Qrow[j + 1] - 1] = coefficient * operator_mat
                # mpo_sign[j][Qrow[j] - 1, Qrow[j + 1] - 1] = str(coefficient) + operator
    # return mpo, mpo_sign
    
    return mpo


# ================================
# 利用 automata 生成稀疏矩阵
# ================================

import scipy.sparse as _sparse


def local_sparse_contract_right2left(Ws0, Ws1):
    """从右到左收缩"""
    e1, e2, e3, e4 = Ws0.shape
    d1 = len(Ws1)
    d4 = len(Ws1[0])

    res = []
    for i in range(e1): # 左侧张量行指标
        row = []
        for j in range(d4): # 右侧张量行指标

            tmp1 = _sparse.csr_array(Ws0[i, :, :, 0])
            tmp2 = Ws1[0][j]
            spmtx = _sparse.kron(tmp1, tmp2, format="csr")
            for l in range(1, d1):
                tmp1 = _sparse.csr_array(Ws0[i, :, :, l])
                if tmp1.nnz == 0:
                    continue
                tmp2 = Ws1[l][j]
                spmtx += _sparse.kron(tmp1, tmp2, format="csr")
            row.append(spmtx)
        res.append(row)

    return res


def local_sparse_contract_left2right(Ws0, Ws1):
    """从左到右收缩"""
    d1 = len(Ws0)
    d4 = len(Ws0[0])
    e1, e2, e3, e4 = Ws1.shape

    res = []
    for i in range(d1):
        row = []
        for j in range(e4):

            tmp1 = Ws0[i][0]
            tmp2 = _sparse.csr_array(Ws1[0, :, :, j])
            spmtx = _sparse.kron(tmp1, tmp2, format="csr")
            for l in range(1, d4):
                tmp2 = _sparse.csr_array(Ws1[l, :, :, j])
                if tmp2.nnz == 0:
                    continue
                tmp1 = Ws0[i][l]
                spmtx += _sparse.kron(tmp1, tmp2, format="csr")

            row.append(spmtx)
        res.append(row)

    return res


def get_sparse_matrix(
    L: int,
    hlocals: list[str],
    positions: list[tuple[int, ...]],
    coefficients: list[float],
    pauli: int = True,
    usecuda: bool = False,
) -> _sparse.csr_matrix:
    """
    利用 automata 生成稀疏矩阵
    
    Example:
    >>> L = 10
    >>> ham = op.heisenberg_operator(L)
    >>> hlocals, positions, coefficients = [], [], []
    >>> for opt, pos, coef in ham.each_term():
    >>>     hlocals.append(opt)
    >>>     positions.append(pos)
    >>>     coefficients.append(coef)
    >>> get_sparse_matrix(L, hlocals, positions, coefficients)
    
    用 GPU 直积，要 17 秒，automata 只要 3 秒    
    >>> L = 24
    >>> ham = qt.generate.operas.heisenberg_operator(L)
    >>> ham = ham.expandxy()
    >>> import time
    >>> t = time.time()
    >>> res = cpx.scipy.sparse.coo_matrix((2**L, 2**L), dtype=cp.float64)
    >>> for oper, pos, coef in ham.each_term():
    >>>     leftI = cpx.scipy.sparse.eye(2**pos[0])
    >>>     rightI = cpx.scipy.sparse.eye(2**(L-pos[-1]-1))
    >>>     tmp = cpx.scipy.sparse.coo_matrix(cp.asarray(qt.generate.matrix.pauli_matrix(oper)))
    >>>     res += cpx.scipy.sparse.kron(cpx.scipy.sparse.kron(leftI, tmp), rightI)
    >>>     cp.get_default_memory_pool().free_all_blocks()
    >>> print(time.time()-t)

    """
    Ws = automata_mpo(L, hlocals, positions, coefficients, pauli=pauli)
    assert L % 2 == 0, "L must be even"
    mid = len(Ws) // 2

    # 从左到右收缩
    d1, _, _, d4 = Ws[0].shape
    resL = [[_sparse.csr_array(Ws[0][i, :, :, j]) for j in range(d4)] for i in range(d1)]
    for i in range(1, mid):
        resL = local_sparse_contract_left2right(resL, Ws[i])

    # 从右到左收缩
    d1, _, _, d4 = Ws[-1].shape
    resR = [
        [_sparse.csr_array(Ws[-1][i, :, :, j]) for j in range(d4)] for i in range(d1)
    ]
    for i in range(1, mid):
        resR = local_sparse_contract_right2left(Ws[-i - 1], resR)

    if usecuda:
        if L > 22:  # 小于 22 时，CPU, GPU 之间的数据传输时间不值得
            try:
                # 最后直积求和
                import cupyx as cpx
                import cupy as cp
                if L > 25:  # 小于 25 时，GPU 的内存不够，cpx.scipy.sparse.kron需要非常多的中间内存
                    kron = lambda x,y : _sparse.kron(x, y, format='coo')
                    _tocsr = lambda mat: cpx.scipy.sparse.csr_matrix((cp.asarray(mat.data), cp.asarray(mat.indices), cp.asarray(mat.indptr)), shape=mat.shape)
                else:
                    kron = lambda x,y : cpx.scipy.sparse.kron(x, y, format='coo')
                    _tocsr = lambda mat: mat
                usecuda = True
            except ImportError:
                usecuda = False
    
    if not usecuda:
        kron = lambda x,y : _sparse.kron(x, y, format='csr')
        _tocsr = lambda mat: mat
    
    # 最后直积求和
    res = kron(resL[0][0], resR[0][0])
    # cp.get_default_memory_pool().free_all_blocks()
    
    res = _tocsr(res)
    for i in range(1, len(resR)):
        tmp = kron(resL[0][i], resR[i][0])
        res += _tocsr(tmp)
        # cp.get_default_memory_pool().free_all_blocks()

    if usecuda:
        res = res.get()
        # cp.get_default_memory_pool().free_all_blocks()
        
    return res


def contract(Ws: list[_np.ndarray], type:str) -> _np.ndarray:
    """
    Args: Ws, type=("mps" or "mpo")
    Returns: vector or matrix
    """
    if type == "mps":
        result = _np.array([[[1.0]]])
        for W in Ws:
            result = _np.einsum("abc,cde->abde", result, W, optimize=True)
            result = result.reshape(1, -1, W.shape[-1])
        return result[0, :, 0]
    elif type == "mpo":
        result = _np.array([[[[1.0]]]])
        d = Ws[0].shape[1]
        for i, W in enumerate(Ws):
            result = _np.einsum("abcd,defg->abecfg", result, W, optimize=True)
            result = result.reshape(1, d ** (i + 1), d ** (i + 1), W.shape[-1])
        return result[0, :, :, 0]
    raise TypeError("Need a type 'mps' or 'mpo'")
