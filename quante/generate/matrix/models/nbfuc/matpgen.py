# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 20:29:20
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-11 20:45:52

# this file saves some fast matrix generation functions
# used for testing and debugging

from typing import Optional
import scipy.sparse as _sparse
from .....linalg.operations import ikron
from ...pauli import pauli_matrix

def parallel_build_matrix(hlocals, positions, coefficients, L, S, pauli=False, sparse=False, parallel=False, nthreads:Optional[int] =None):
    """
    通过直积生成矩阵，效率并不是否高，可以用于验证
    
    Examples
    --------
    >>> from quante.generate.matrix import parallel_build_matrix
    >>> mat = parallel_build_matrix(*ham.split_data(), L, S, pauli=False, sparse=False, parallel=True, nthreads=4)
    """
    
    # 生成操作符字典
    if S == 0.5:
        scale_factor = 2. if pauli else 1.
        OPER = {op: pauli_matrix(op, S=S) * scale_factor if op in 'xyZ' else pauli_matrix(op, S=S) for op in "xyZzpmiI"}
    else:
        OPER = {op: pauli_matrix(op, S=S) for op in "xyzpmiI"}
        OPER["Z"] = pauli_matrix("Z", S=S) * 2
        
    
    ikron_kws = {'sparse': True, 'stype': 'coo', 'coo_build': True}
    dims = (int(2 * S + 1),) * L
    
    def gen_term(args:tuple[str, int, float]):
        opstr, indx, j = args
        tmp = [_sparse.csc_array(OPER[oi]) for oi in opstr]
        return j * ikron(tmp, dims, indx, **ikron_kws) # type: ignore  #todo ikron 需要重写
    
    if parallel:
        from .....basicfun.utils_numba import get_thread_pool, parallel_reduce
        pool = get_thread_pool(nthreads)
        ham:_sparse.csr_array = parallel_reduce(lambda a,b : a+b, pool.map(gen_term, zip(hlocals, positions, coefficients)))
    else:
        ham:_sparse.csr_array = sum(map(gen_term, zip(hlocals, positions, coefficients))) # type: ignore
    
    if sparse:
        return ham.tocsr()
    
    return ham.toarray()


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
    oper,
    pauli: bool = False,
    usecuda: bool = False,
) -> _sparse.csr_matrix:
    """
    利用 automata 生成稀疏矩阵
    
    Examples
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
    Ws = oper.automata(L, pauli=pauli)
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

