# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-08 14:20:14
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-16 17:16:21

from quspin.operators import hamiltonian as qshamiltonian
from quspin.basis.base import _get_index_type, _is_diagonal, _update_diag
from quspin.basis import spin_basis_1d, spin_basis_general

from ....generate.operas import Oper, SpinOper
from .super_basis import spin_super_basis

import numpy as _np
import scipy.sparse as _sp
from warnings import warn

def hamiltonian(
    oper, basis, dtype, 
    *, 
    sparse=True, check_symm=False, check_herm=False, check_pcon=False,
    **kwargs
):
    _kwargs = {
        "dynamic_list": []
    }
    _kwargs.update(kwargs)

    if isinstance(oper, SpinOper):
        if basis._pauli == -1:
            pauli = True
        elif basis._pauli == 0:
            pauli = False
        else:
            raise ValueError("basis._pauli must be -1 or 0")
        if check_symm:
            dic = {}
            if isinstance(basis, spin_basis_1d):
                L = basis.L
                blocks = basis.blocks
                if blocks['kblock'] is not None:
                    a = blocks['a']
                    dic['kblock'] = (_np.arange(L) + a) % L
                elif blocks['pblock'] is not None:
                    dic['pblock'] = _np.arange(L-1, -1, -1)
                elif blocks['zblock'] is not None:
                    dic['zblock'] = -(_np.arange(L) + 1)
                elif blocks['pzblock'] is not None:
                    dic['pzblock'] = -(_np.arange(L-1, -1, -1) + 1)
                    warn("check pzblock is not fully supported yet")
                elif blocks['zAblock'] is not None:
                    warn("check zAblock is not fully supported yet")
                elif blocks['zBblock'] is not None:
                    warn("check zBblock is not fully supported yet")
                # print(basis._Np)
                dic['Nup'] = basis._Np
            elif isinstance(basis, spin_basis_general):
                L = basis._pcon_args['N']
                dic = basis._maps_dict
                dic['Nup'] = basis._pcon_args['Nup']
            elif isinstance(basis, spin_super_basis):
                L = 2*basis._user_N
                for k, (f,p1,p2,p3) in basis._user_maps.items():
                    res = []
                    for i in range(L):
                        s = 1 << (L-i-1)
                        sp = f(s,2*basis._user_N,0,p3)
                        
                        ii = int(_np.log2(sp))
                        if ii == _np.log2(sp):
                            ii = L - ii -1
                        else:
                            sp = ((1 << L) - 1) ^ sp
                            ii = int(np.log2(sp)) - L
                        res.append(ii) 
                    dic[k] = _np.array(res)
                
                if basis._user_pcon_dict is not None:
                    nsf = basis._user_pcon_dict.get('next_state', None)
                    if nsf is not None:
                        if nsf.__name__ == 'next_state_Np':
                            dic['Nup'] = 1
                        elif nsf.__name__ == 'next_state_Nd':
                            dic['Ndiff'] = 1
                        elif nsf.__name__ == 'next_state_Np_Nd':
                            dic['Nup2'] = 1
                        else:
                            raise ValueError(f"Unknown next_state: {nsf.__name__}")
            oper.check_symm(L, pauli=pauli, maps=dic)
            check_symm = False
            check_pcon = False
        oper = oper.to_quspin(pauli=pauli)
    elif isinstance(oper, Oper):
        oper = oper.to_quspin()
        if check_symm:
            raise ValueError("only SpinOper can check_symm now")
    else:
        raise ValueError(f"oper must be Oper, not {type(oper)}")
    
    ham = qshamiltonian(
        static_list=oper,
        basis=basis,
        dtype=dtype,
        check_symm=check_pcon,
        check_herm=check_herm,
        check_pcon=check_pcon,
        **_kwargs
    )
    if sparse:
        return ham.tocsr()
    else:
        return ham.toarray()


def shift_sector_oper(asym_basis, sym_basis, liou_asym_qs, dtype=_np.complex128):
    op_list = []
    for opnm, coef in liou_asym_qs:
        for i, *j in coef:
            op_list.append([opnm, j, _np.real_if_close(i)])
    v_in = _np.eye(sym_basis.Ns, dtype=dtype)
    mat10 = _np.zeros((asym_basis.Ns, sym_basis.Ns), dtype=dtype)
    asym_basis.Op_shift_sector(sym_basis, op_list, v_in=v_in, v_out=mat10, dtype=dtype)
    return _sp.csr_array(mat10)



def shift_sector_oper_sparse(basis_left, basis_right, liou_asym_qs, dtype=_np.complex128):
    off_diag = None
    diag = None
    minNs = min(basis_left.Ns, basis_right.Ns)
    maxNs = max(basis_left.Ns, basis_right.Ns)
    index_type = _get_index_type(maxNs)

    op_list = []
    for opnm, coef in liou_asym_qs:
        for i, *j in coef:
            op_list.append([opnm, j, _np.real_if_close(i)])

    for opstr, indx, J in op_list:
        ME, row, col = _Op(basis_left, basis_right, opstr, indx, J, dtype)
        if len(ME) > 0:
            imax = max(row.max(), col.max())
            row = row.astype(index_type)
            col = col.astype(index_type)
            if _is_diagonal(row, col):
                if diag is None:
                    diag = _np.zeros(minNs, dtype=dtype)
                _update_diag(diag, row, ME)
            else:
                if off_diag is None:
                    off_diag = _sp.csr_matrix(
                        (ME, (row, col)), shape=(basis_left.Ns, basis_right.Ns), dtype=dtype
                    )
                else:
                    off_diag = off_diag + _sp.csr_matrix(
                        (ME, (row, col)), shape=(basis_left.Ns, basis_right.Ns), dtype=dtype
                    )

    if diag is not None and off_diag is not None:
        indptr = _np.arange(minNs + 1)
        return off_diag + _sp.csr_matrix(
            (diag, indptr[: minNs], indptr), shape=(basis_left.Ns, basis_right.Ns), dtype=dtype
        )

    elif off_diag is not None:
        return off_diag
    elif diag is not None:
        return _sp.dia_matrix(
            (_np.atleast_2d(diag), [0]), shape=(basis_left.Ns, basis_right.Ns), dtype=dtype
        )
    else:
        return _sp.dia_matrix((basis_left.Ns, basis_right.Ns), dtype=dtype)


def _Op(basis_left, basis_right, opstr, indx, J, dtype):
    if basis_right._S == "1/2":
        ME, row, col = hcb_basis_general._Op(basis_right, opstr, indx, J, dtype)
        if basis_right._pauli == 1:
            n = len(opstr.replace("I", ""))
            ME *= 1 << n
        elif basis_right._pauli == -1:
            n = len(opstr.replace("I", "").replace("+", "").replace("-", ""))
            ME *= 1 << n

        return ME, row, col

    else:
        raise NotImplementedError("only support spin-1/2 or hard-core boson now")


