# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-22 15:12:21
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-22 22:23:45

import numpy as np
import scipy.sparse as sp
import numba
from ...generate.basis.symmetry.basis_class_nb import coodiaglists2csr
from tqdm import tqdm

# 辅助函数：判断是否为对角元素
@numba.njit
def _is_diagonal(row, col):
    for i in range(row.size):
        if row[i] != col[i]:
            return False

    return True

def _get_index_type(Ns):
    if Ns < np.iinfo(np.int32).max:
        return np.int32
    else:
        return np.int64

@numba.njit
def _update_diag(diag, ind, ME):
    for i in range(ind.size):
        diag[ind[i]] += ME[i]


def _make_matrix(self, op_list, dtype):
    """takes list of operator strings and couplings to create matrix."""
    diag = None
    index_type = _get_index_type(self.Ns)

    diag_list = []
    offdiag_list = []
    
    # 顺序执行 (Windows 或单线程)
    op_list = tqdm(op_list, ascii=True)
    for opstr, indx, J in op_list:
        ME, row, col = self.Op(opstr, indx, J, dtype)
        if len(ME) > 0:
            row = row.astype(index_type)
            col = col.astype(index_type)
            if _is_diagonal(row, col):
                diag_list.append((ME, row))
            else:
                offdiag_list.append((ME, row, col))

    if len(diag_list) > 0:
        diag = np.zeros(self.Ns, dtype=dtype)
        for ME, row in diag_list:
            _update_diag(diag, row, ME)
    if len(offdiag_list) > 0:
        row_result = [i[1] for i in offdiag_list]
        col_result = [i[2] for i in offdiag_list]
        ele_result = [i[0] for i in offdiag_list]
        return coodiaglists2csr(row_result, col_result, ele_result, diag, self.Ns, index_type, dtype)
    
    if diag is not None:
        return sp.dia_array((np.atleast_2d(diag),[0]),shape=(self.Ns,self.Ns),dtype=dtype)
    else:
        return sp.dia_array((self.Ns,self.Ns),dtype=dtype)


def optimize_basis(basis):
    from types import MethodType
    basis._make_matrix = MethodType(_make_matrix, basis)
    return basis


# 使用示例
if __name__ == "__main__":
    import quante as qt
    from quspin.operators import hamiltonian
    from quspin.basis import spin_basis_1d

    ham = qt.generate.operas.heisenberg_operator(L=10)
    static = ham.to_quspin()

    basis = spin_basis_1d(L=10, Nup=5)
    H1 = hamiltonian(static, [], basis=basis, dtype=np.complex128, check_herm=False, check_pcon=False, check_symm=False).static
    basis = optimize_basis(basis)
    H2 = hamiltonian(static, [], basis=basis, dtype=np.complex128, check_herm=False, check_pcon=False, check_symm=False).static
    print(H1-H2)