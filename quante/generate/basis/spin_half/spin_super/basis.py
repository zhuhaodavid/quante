# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-01 15:20:57
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-01 18:53:34

from ...basis_class import SpinHalfBasis
import numpy as np
import scipy.sparse as _sp

class SpinHalfSuperBasis(SpinHalfBasis):
    def __init__(self, L: int, Ndiff, Nup, indx_order, **blocks) -> None:
        super().__init__(L)
        self.Nup = Nup
        self.Ndiff = Ndiff

        ns = []
        ps = []
        bs = []
        for key, (_perm, _block) in blocks.items():
            ns.append(key)
            if _perm.ndim == 1:
                _perm = np.array([
                    [2*L+i, 2*L-a-1, 1] if i < 0 else [2*L-i-1, 2*L-a-1, 0]
                    for a,i in enumerate(_perm)
                ])
            ps.append(_perm)
            bs.append(_block)
        self.block_name = ns
        self.perm = np.array(ps)
        self.block = np.array(bs)

        if indx_order == 'stacked':
            self.indx_order = indx_order
            _ancillary_perm = [i+j*L for j in [1,0] for i in range(0,L)]
        elif indx_order == 'snake':
            self.indx_order = indx_order
            _ancillary_perm = [i+j for i in range(0,2*L,2) for j in [1,0]]
        else:
            raise ValueError(f"indx_order should be 'stacked' or 'snake', not {indx_order}")
        
        self.ancillary_perm = np.array([
            [2*L+i, 2*L-a-1, 1] if i < 0 else [2*L-i-1, 2*L-a-1, 0]
            for a,i in enumerate(_ancillary_perm)
        ])
    
    def _real_sparse_matrix(self, op_list_sym, op_list_asym, hascomplex, savememory=False):
        from ...basis_class_nb import _is_diagonal, _update_diag, add_, _get_index_type, coodiaglists2csr
        diag = None
        off_diag = None
       
        dtype = np.complex128 if hascomplex or self.default_complex else np.float64

        # 预设内存，避免反复分配内存
        real_Ns = self.Ns * self._double_Ns
        index_type = _get_index_type(real_Ns)
        row_init = np.empty(real_Ns, dtype=index_type)
        col_init = np.empty(real_Ns, dtype=index_type)
        ele_int = np.empty(real_Ns, dtype=dtype)
        
        if not savememory:
            row_result = []
            col_result = []
            ele_result = []
        
        for oplist, sym in [(op_list_sym, True), (op_list_asym, False)]:
            for opnm, posn, coef in oplist:
                row, col, ele = self._Op(opnm, posn, coef, row_init, col_init, ele_int, op_sym=sym) 
                # 主要的时间花费，一半时间花在这里
                if(len(ele)>0):
                    if row is None:
                        if diag is None:
                            diag = ele
                        else:
                            add_(diag, ele)
                            # diag += ele
                    elif _is_diagonal(row,col):
                        if diag is None:
                            diag = np.zeros(self.Ns,dtype=dtype)
                        _update_diag(diag,row,ele)
                    else:
                        if not savememory:
                            ele_result.append(ele)
                            row_result.append(row)
                            col_result.append(col)
                        else:
                            # todo, csr 如何并行相加？
                            tmp = _sp.csr_array((ele,(row,col)),shape=(self.Ns,self.Ns),dtype=dtype) 
                            off_diag = tmp if off_diag is None else off_diag + tmp
        
        if not savememory:
            if len(ele_result) > 0:
                return coodiaglists2csr(row_result, col_result, ele_result, diag, self.Ns, index_type, dtype)
                # return coolists2csr2(row_result, col_result, ele_result, self.Ns, dtype)
        else:
            if off_diag is not None and diag is None:
                return off_diag
            elif off_diag is not None and diag is not None:
                indptr = np.arange(self.Ns+1)
                return off_diag + _sp.csr_array((diag,indptr[:self.Ns],indptr),shape=(self.Ns,self.Ns),dtype=dtype)

        if diag is not None:
            return _sp.dia_array((np.atleast_2d(diag),[0]),shape=(self.Ns,self.Ns),dtype=dtype)
        else:
            return _sp.dia_array((self.Ns,self.Ns),dtype=dtype)
    
        pass

############################################
# Full
##############################################       
class BasisFull(SpinHalfSuperBasis):
    def __init__(self, L: int, Ndiff, Nup, indx_order, **blocks) -> None:
        assert len(blocks) == 0, "Ndiff should not have Z2 symmetry"
        super().__init__(L, Ndiff, Nup, indx_order, **blocks)
        from .basis_core import construct_full_basis
        self.s_list, self.Ns_sym, self.Ns_asym = construct_full_basis(2*self.L, self.ancillary_perm)
        self.Ns = self.Ns_sym + self.Ns_asym
        self.default_complex = False

    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init, op_sym):
        if op_sym:
            from .matrix_core import single_sparse_matrix_element_full_sym
            return single_sparse_matrix_element_full_sym(
                opnm, posn, coef, self.L, self.ancillary_perm, self.Ns_sym, self.Ns, self.s_list, row_init, col_init, ME_init
            )
        else:
            from .matrix_core import single_sparse_matrix_element_full_asym
            return single_sparse_matrix_element_full_asym(
                opnm, posn, coef, self.L, self.ancillary_perm, self.Ns_sym, self.Ns, self.s_list, row_init, col_init, ME_init
            )

