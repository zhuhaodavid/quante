# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-01 15:20:57
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-02 21:44:26

from ...basis_class import SpinHalfBasis
import numpy as np
import scipy.sparse as _sp

class SpinHalfSuperBasis(SpinHalfBasis):
    def __init__(self, L: int, Ndiff, Nup, indx_order, **blocks) -> None:
        super().__init__(L)

        if isinstance(Nup, int):
            Nup = [Nup]
        if isinstance(Ndiff, int):
            Ndiff = [Ndiff]
        
        if Ndiff is not None:
            _Ndiff = []
            for n in Ndiff:
                assert n >= 0, "Ndiff should be non-negative"
                if n not in _Ndiff:
                    _Ndiff.append(n)
                if -n not in _Ndiff:
                    _Ndiff.append(-n)
            if Nup is not None:
                _Nup = []
                for ndiff in _Ndiff:
                    for nup in Nup:
                        assert (nup + ndiff) % 2 == 0, f"Nup={nup} and Ndiff={ndiff} are incompatible"
                        _Nup.append([(nup + ndiff)//2, (nup - ndiff)//2])
                Nup = np.array(_Nup)
        else:
            _Ndiff = None
        
        self.Nup = Nup
        self.Ndiff = _Ndiff

        ns = []
        ps = []
        bs = []
        for key, (_perm, _block) in blocks.items():
            ns.append(key)
            _perm = np.array(_perm)
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
        
        self._ancillary_perm = np.array([
            [2*L+i, 2*L-a-1, 1] if i < 0 else [2*L-i-1, 2*L-a-1, 0]
            for a,i in enumerate(_ancillary_perm)
        ])
    
    def _real_sparse_matrix(self, op_list_sym, op_list_asym, hascomplex, savememory=False):
        from ...basis_class_nb import _is_diagonal, _update_diag, add_, _get_index_type, coodiaglists2csr, coolists2csr2
        off_diag = None
        diag = None
        
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
        
        op_sym = True
        for op_list, op_sym in [(op_list_sym, True), (op_list_asym, False)]:
            for opnm, posn, coef in op_list:
                row, col, ele = self._Op_real(opnm, posn, coef, row_init, col_init, ele_int, op_sym)  # 主要的时间花费，一半时间花在这里
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

############################################
# Full
############################################     
class BasisFull(SpinHalfSuperBasis):
    def __init__(self, L: int, Ndiff, Nup, indx_order, **blocks) -> None:
        assert len(blocks) == 0, "Ndiff should not have Z2 symmetry"
        super().__init__(L, Ndiff, Nup, indx_order, **blocks)

        if Nup is not None and Ndiff is not None:
            from .basis_core import construct_Nup2_basis
            self.s_list, self.Ns_sym, self.Ns_asym = construct_Nup2_basis(
                self.L, self.Nup, self._ancillary_perm
            )
        elif Nup is not None:
            from .basis_core import construct_Ndiff_basis
            flipmask = 0
            self.s_list, self.Ns_sym, self.Ns_asym = construct_Ndiff_basis(
                self.L, self.Nup, self._ancillary_perm, flipmask
            )
        elif Ndiff is not None:
            from .basis_core import construct_Ndiff_basis
            flipmask = (1 << L) - 1
            self.s_list, self.Ns_sym, self.Ns_asym = construct_Ndiff_basis(
                self.L, self.Ndiff, self._ancillary_perm, flipmask
            )
        else:
            from .basis_core import construct_full_basis
            self.s_list, self.Ns_sym, self.Ns_asym = construct_full_basis(
                self.L, self._ancillary_perm
            )
        self.Ns = self.Ns_sym + self.Ns_asym
        self.default_complex = False

    def _Op_real(self, opnm, posn, coef, row_init, col_init, ME_init, op_sym):
        if op_sym:
            from .matrix_core import single_sparse_matrix_element_full_sym
            return single_sparse_matrix_element_full_sym(
                opnm, posn, coef, 2*self.L, self._ancillary_perm, self.Ns_sym, self.Ns, self.s_list, row_init, col_init, ME_init
            )
        else:
            from .matrix_core import single_sparse_matrix_element_full_asym
            return single_sparse_matrix_element_full_asym(
                opnm, posn, coef, 2*self.L, self._ancillary_perm, self.Ns_sym, self.Ns, self.s_list, row_init, col_init, ME_init
            )
    
    def project(self, vec):
        """Project a vector to the symmetry sector.
        """
        from .basis_core import project_full
        if vec.ndim == 1:
            vec = vec.reshape(-1,1)
        return project_full(vec, 2*self.L, self.s_list, self.Ns_sym, self.Ns, self._ancillary_perm, vec.dtype)

    def recover(self, vec):
        """Recover a vector from the symmetry sector.
        """
        from .basis_core import recover_full
        if vec.ndim == 1:
            vec = vec.reshape(-1,1)
        res = recover_full(vec, 2*self.L, self.s_list, self.Ns_sym, self.Ns, self._ancillary_perm)
        return np.real_if_close(res)

    def projection_matrix(self):
        """Return the projection matrix to the symmetry sector.
        """
        from .basis_core import projmat_full
        row, col, ele = projmat_full(self.L, self.s_list, self.Ns_sym, self.Ns, self._ancillary_perm)
        return _sp.csr_array((ele,(row,col)),shape=(1<<2*self.L, self.Ns),dtype=np.complex128)

############################################
# Z21
############################################     
class BasisZ21(SpinHalfSuperBasis):
    def __init__(self, L: int, Ndiff, Nup, indx_order, **blocks) -> None:
        assert len(blocks) == 1, "Ndiff should not have Z2 symmetry"
        super().__init__(L, Ndiff, Nup, indx_order, **blocks)

        if Nup is not None and Ndiff is not None:
            from .basis_core import construct_Nup2_basis_Z21
            self.s_list, self.R_list, self.Ns_sym, self.Ns_asym = construct_Nup2_basis_Z21(
                self.L, self.Nup, self.perm[0], self.block[0],
                self._ancillary_perm
            )
        elif Nup is not None:
            from .basis_core import construct_Ndiff_basis_Z21
            flipmask = 0
            self.s_list, self.R_list, self.Ns_sym, self.Ns_asym = construct_Ndiff_basis_Z21(
                self.L, self.Nup, self.perm[0], self.block[0],
                self._ancillary_perm, flipmask
            )
        elif Ndiff is not None:
            from .basis_core import construct_Ndiff_basis_Z21
            flipmask = (1 << L) - 1
            self.s_list, self.R_list, self.Ns_sym, self.Ns_asym = construct_Ndiff_basis_Z21(
                self.L, self.Ndiff, self.perm[0], self.block[0],
                self._ancillary_perm, flipmask
            )
        else:
            from .basis_core import construct_basis_Z21
            self.s_list, self.R_list, self.Ns_sym, self.Ns_asym = construct_basis_Z21(
                2*self.L, self.perm[0], self.block[0], self._ancillary_perm
            )
        self.Ns = self.Ns_sym + self.Ns_asym
        self.default_complex = False

    def _Op_real(self, opnm, posn, coef, row_init, col_init, ME_init, op_sym):
        if op_sym:
            from .matrix_core import single_sparse_matrix_element_Z21_sym
            return single_sparse_matrix_element_Z21_sym(
                opnm, posn, coef, 2*self.L, self.perm[0], self.block[0], 
                self._ancillary_perm, self.R_list, self.Ns_sym, self.Ns, 
                self.s_list, row_init, col_init, ME_init
            )
        else:
            from .matrix_core import single_sparse_matrix_element_Z21_asym
            return single_sparse_matrix_element_Z21_asym(
                opnm, posn, coef, 2*self.L, self.perm[0], self.block[0], 
                self._ancillary_perm, self.R_list, self.Ns_sym, self.Ns, 
                self.s_list, row_init, col_init, ME_init
            )
    
    def project(self, vec):
        """Project a vector to the symmetry sector.
        """
        from .basis_core import project_Z21
        if vec.ndim == 1:
            vec = vec.reshape(-1,1)
        res = project_Z21(vec, 2*self.L, self.s_list, self.Ns_sym, self.Ns, self._ancillary_perm, self.perm[0], self.block[0])
        return np.real_if_close(res)

    def recover(self, vec):
        """Recover a vector from the symmetry sector.
        """
        from .basis_core import recover_Z21
        if vec.ndim == 1:
            vec = vec.reshape(-1,1)
        return recover_Z21(vec, 2*self.L, self.s_list, self.Ns_sym, self.Ns, self._ancillary_perm, self.perm[0], self.block[0])

    def projection_matrix(self):
        """Return the projection matrix to the symmetry sector.
        """
        from .basis_core import projmat_Z21
        row, col, ele = projmat_Z21(self.L, self.s_list, self.Ns_sym, self.Ns, self._ancillary_perm,
                                    self.perm[0], self.block[0])
        return _sp.csr_array((ele,(row,col)),shape=(1<<2*self.L, self.Ns),dtype=np.complex128)



