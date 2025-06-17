# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:34:42
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:35:37

import numpy as _np
from typing import Callable
from functools import lru_cache

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


