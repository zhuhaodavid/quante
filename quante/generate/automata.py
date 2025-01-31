# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 14:11:36
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-31 13:16:57

import numpy as _np
from quante.generate.matrix import pauli_matrix
from typing import Callable
from functools import lru_cache
import scipy.sparse as _sparse

# ================================================================
#  automata 的一个简单的实现

#    这个实现的优势是：逻辑清晰，代码简洁易读，可以用于理解 automata 的原理
   
#    存在的问题是：效率低，尤其是 L 大时，速度会迅速下降，
#        主要是 fill_Id 会导致多此重复单位阵
#        时间复杂度为 O(n * L), n 为 local hamiltonian 的个数
#        例如 L = 1000, n = 3000, 会导致 9e6 的计算量
# ================================================================

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
    Examples
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


def _fill_Id_in_local_hamiltonian(
    N: int, hlocals: list[str], positions: list[tuple]
) -> list[str]:
    """
    Examples
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


def simple_automata_mpo(
    N: int,
    hlocals: list[str],
    positions: list[tuple],
    coefficients: _np.ndarray,
    d: int = 2,
    pauli: int = True,
    local_matrix_function=None,
    dtype=None
) -> list[_np.ndarray]:
    """
    Basis could only to be string such as "01101..."

    called from `generate.opera.Oper.automata`
    
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
        left, right = position[0], position[-1]  #!! left < right 一定吗？
        Qmat = _get_Qmat(i + 1, left, right, Qmat, N)
    Qmat = _finalize_Qmat(Qmat)
    Q = [_np.max(Qmat[:, i]) for i in range(N + 1)]
    mpo = [_np.zeros((Q[i], d, d, Q[i + 1]), dtype=dtype) for i in range(N)]

    # * write element of 4-order of mpo
    for i, (hlocal, position) in enumerate(zip(hlocals, positions)):
        # print(hlocal)  #!! 看看这个就知道为什么慢了
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
            else:
                mpo[j][Qrow[j] - 1, :, :, Qrow[j + 1] - 1] = coefficient * operator_mat
    
    return mpo


def automata_mpo_str(
    N: int,
    hlocals: list[str],
    positions: list[tuple],
    coefficients: _np.ndarray,
    dtype=None
) -> list[_np.ndarray]:
    """
    字符格式，方便调试
    """
    coefficients = _np.real_if_close(coefficients)
    if dtype is None:
        dtype = coefficients.dtype
        
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

    mpo_sign = [_np.zeros((Q[i], Q[i + 1]), dtype=object) for i in range(N)]

    # * write element of 4-order of mpo
    for i, (hlocal, position) in enumerate(zip(hlocals, positions)):
        Qrow = Qmat[i, :]
        for j, operator in enumerate(hlocal):
            coefficient = coefficients[i] if j == position[0] else 1.0

            self_add = True
            if Qrow[j] == Qrow[j + 1] == 1 and j != N - 1:
                self_add = False
            if Qrow[j] == _np.max(Q):
                self_add = False

            if self_add:
                mpo_sign[j][Qrow[j] - 1, Qrow[j + 1] - 1] = (
                    str(mpo_sign[j][Qrow[j] - 1, Qrow[j + 1] - 1])
                    + "+"
                    + str(coefficient)
                    + operator
                )
            else:
                mpo_sign[j][Qrow[j] - 1, Qrow[j + 1] - 1] = str(coefficient) + operator
    for i in mpo_sign:
        print(i)


def automata_mps(array):
    """根据 ndarray/sparse_array 生成相应的 mps，spin 1/2

    Parameters
    ----------
    array : ndarray/sparse_array
        系数矩阵表示

    Returns
    -------
    list[ndarray]
        mps 表示
    """
    d = 2  # 2 for spin
    up, down = pauli_matrix("u")[:, -1], pauli_matrix("d")[:, -1]  # (2,1) -> (2,)

    # * initialize each tensor of MPS
    spare_array = _sparse.coo_array(array.reshape(-1, 1))
    basiss, coefficients = spare_array.row, spare_array.data
    has_full_zero = basiss[0] == 0
    if has_full_zero:
        full_zero_coe = coefficients[0]
        coefficients = coefficients[1:]
        basiss = basiss[1:]

    N, n = int(_np.log2(spare_array.shape[0])), len(basiss)  # numbers of site N, numbers of basis n
    assert 2 ** N == spare_array.shape[0], "The length of array must be 2^N"
    Qmat = _np.zeros((n + 1, N + 1), dtype=basiss.dtype)
    Qmat[0, :], Qmat[:, 0] = 1, 1
    for i in range(n):
        basis = basiss[i]
        left, right = N-int(_np.log2(basis))-1, N-int(_np.log2(basis & -basis))-1
        Qmat = _get_Qmat(
            i + 1, left, right, Qmat, N
        )  # i-th basis corresponds to (i+1)-th row of Qmat
    Qmat = _finalize_Qmat(Qmat)
    # Qmat2wf(Qmat, coefficients, basiss)
    Q = [_np.max(Qmat[:, i]) for i in range(N + 1)]
    Ws = [_np.zeros((Q[i], d, Q[i + 1]), dtype=coefficients.dtype) for i in range(N)]

    # * recode position of the first not "1" element
    coe_positions = [
        _np.argmax(row != 1) if _np.any(row != 1) else len(row) - 1 for row in Qmat
    ]

    # * write element of 3-order tensor of mps
    for i, basis in enumerate(basiss):
        Qrow = Qmat[i, :]
        for j in range(N):
            spin = "1" if (basis >> (N-1-j)) & 1 else "0"
            coefficient = coefficients[i] if j == coe_positions[i] - 1 else 1.0
            Ws[j][Qrow[j] - 1, :, Qrow[j + 1] - 1] = (
                coefficient * up if spin == "0" else coefficient * down
            )
            # "-1" because 1x1 matrix only has [0,0] element
    if has_full_zero:
        Ws[-1][0, :, 0] += full_zero_coe * up
    return Ws

# ================================================================
#  更高效的 automata 实现
# ================================================================

def get_weight_fuc(path, L):
    weight_fuc = [[] for _ in range(L)]
    dims_info = [[1, 1]] + [[1, 1] for _ in range(L - 2)] + [[1, 1]]

    # 遍历每一条路径
    for opstr, indx, J in path:
        # 相同位置不能出现两个算符，如 "xx" @ [1,1]
        assert len(set(indx)) == len(indx), f"opstr {opstr}, {indx} is illegal"

        start, end = _np.min(indx), _np.max(indx)  # 拿到算符从何处开始，到何处结束
        weight_fuc, dims_info = _add_weight_func(
            start, end, indx, opstr, weight_fuc, J, dims_info
        )

    _, start_of_last_row = _get_id_of_first_last(weight_fuc)
    for i in range(start_of_last_row, L - 1):
        dims_info[i][1] += 1
        dims_info[i + 1][0] += 1

    return weight_fuc, dims_info


def _add_weight_func(start, end, indx, opstr, weight_fuc, J, dims_info):
    weight_fuc_i = []  # 这一项产生的权重函数临时存在这里

    state_cur = 0  # 记录当前的行数
    Jpos = -1  # 记录系数存放的位置

    for i in range(start, end + 1):
        # 如果 i 在 indx 当中
        if i in indx:
            # 先找到在 opstr 中对应的算符
            opstr_indx = indx.index(i)
            opstr_cur = opstr[opstr_indx]

        else:
            opstr_indx = -1
            opstr_cur = "i"

        # 如果是最后一个算符，那么直接放到回收站
        if i == end:
            sigma = (state_cur, -1, i, opstr_cur)
            weight_fuc_i.append(sigma)
            if Jpos == -1:
                Jpos = i
            continue

        # 首先要扩大维数了
        dims_info[i][1] += 1
        dims_info[i + 1][0] += 1

        new_state = dims_info[i][1] - 1  # 这个是新增加的行
        sigma = (state_cur, new_state, i, opstr_cur)
        weight_fuc_i.append(sigma)

        # 如果 Jpos 还没有指定，就指定在这个 i
        if Jpos == -1:
            Jpos = i

        # 更新当前行
        state_cur = new_state

    if Jpos == -1 and len(opstr) >= 1:
        raise Exception

    # 将 weight_fuc_i 中的元素按照位置填入 weight_fuc 中
    for num_in, num_out, i, opstr_cur in weight_fuc_i:
        J_res = J if i == Jpos else 1.0
        sigma = (num_in, num_out, opstr_cur, J_res)
        weight_fuc[i].append(sigma)

    return weight_fuc, dims_info


def _get_id_of_first_last(weight_fuc):
    end_of_first_row, start_of_last_row = 0, -1
    for i, weight_fuc_i in enumerate(weight_fuc):
        for j in weight_fuc_i:
            if j[0] == 0:
                end_of_first_row = i
            if j[1] == -1 and start_of_last_row == -1:
                start_of_last_row = i
    return end_of_first_row, start_of_last_row


def _get_mpo(dims_info, weight_fuc, local_matrix, local_dim, dtype):
    end_of_first_row, start_of_last_row = _get_id_of_first_last(weight_fuc)
    mpo = [_np.zeros((i, local_dim, local_dim, j), dtype=dtype) for i, j in dims_info]

    for i, mpo_i in enumerate(mpo):
        for row, col, opstr, J in weight_fuc[i]:
            mpo_i[row, :, :, col] += J * local_matrix(opstr)
        if i < end_of_first_row:
            mpo_i[0, :, :, 0] += local_matrix("i")
        if i > start_of_last_row:
            mpo_i[-1, :, :, -1] += local_matrix("i")
    return mpo


def automata_mpo(
    path:list[str, tuple, float], L, local_matrix:Callable[[str], _np.ndarray] | None = None, dtype=_np.complex128
):
    # 包装 local_matrix
    lru_local_matrix = lru_cache(maxsize=None)(local_matrix)
    local_dim = local_matrix("i").shape[0]

    # 获取 weight_fuc 和 dims_info
    weight_fuc, dims_info = get_weight_fuc(path, L)

    # 生成 mpo
    return _get_mpo(dims_info, weight_fuc, lru_local_matrix, local_dim, dtype)

