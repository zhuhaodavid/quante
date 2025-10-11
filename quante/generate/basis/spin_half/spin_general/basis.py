# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-28 14:27:09
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-11 22:40:47

from ...basis_class import SpinHalfBasis
import numpy as np

def get_permute_number(L, perm):
    indx0 = np.arange(L)
    sign0 = np.ones(L, dtype=int)
    indx = np.copy(indx0)
    sign = np.copy(sign0)

    for i in range(L+1):
        indx, sign = _permute(L, indx, sign, perm)
        if all(indx == indx0) and all(sign == sign0):
            return i + 1
    
    raise ValueError("perm should be self-inverse")

def _permute(L, indx, sign, perm):
    res = np.zeros(L, dtype=int)
    for ri, r in enumerate(indx):
        for pi, p in enumerate(perm[:, 0]):
            if r == p:
                res[ri] = perm[pi, 1]
                sign[ri] *= 1-2*perm[pi, 2]
                break
    return res, sign

def _valdate_z2(L, perm, block):
    indx0 = np.arange(L)
    sign0 = np.ones(L, dtype=int)
    indx, sign = _permute(L, indx0, sign0, perm)
    indx, sign = _permute(L, indx, sign, perm)
    assert all(indx == np.arange(L)), f"perm should be self-inverse, perm: \n{perm}"
    assert all(sign == 1), f"perm should be self-inverse, perm: \n{perm}"
    assert block in [0, 1], "pblock should be 0 or 1"

def _valdate_z2_commute(L, perm0, perm1):
    indx0 = np.arange(L)
    sign0 = np.ones(L, dtype=int)

    indx1, sign1 = _permute(L, indx0, sign0, perm0)
    indx1, sign1 = _permute(L, indx1, sign1, perm1)

    indx2, sign2 = _permute(L, indx0, sign0, perm1)
    indx2, sign2 = _permute(L, indx2, sign2, perm0)

    assert all(indx1 == indx2) and all(sign1 == sign2), f"perm0 and perm1 should commute, perm0: \n{perm0},\n perm1: \n{perm1}"


  
############################################
# Z2N
##############################################
class SpinHalfGeneralBasis(SpinHalfBasis):
    def __init__(self, L: int, flipset, Ndiff, Nup, **blocks) -> None:
        super().__init__(L)

        self.Ndiff = None
        self.Nup2 = None
        self.flipset = None

        if isinstance(Ndiff, int):
            Ndiff = [Ndiff]
        if isinstance(Nup, int):
            Nup = [Nup]

        if Ndiff is not None and Nup is not None:
            _Nup2 = []
            for i in Nup:
                for j in Ndiff:
                    assert (i+j)%2 == 0, f"Nup + Ndiff must be even, but got Nup={i}, Ndiff={j}."
                    _Nup2.append(((i+j)//2, (i-j)//2))
            self.Nup2 = np.array(_Nup2, dtype=np.int64)
            self.flipmask = sum(1 << (L-1-flip) for flip in flipset)
        elif Nup is not None:
            self.flipmask = 0
            self.Ndiff = Nup
            self.flipnumber = 0
        elif Ndiff is not None:
            self.flipmask = sum(1 << (L-1-flip) for flip in flipset)
            self.Ndiff = np.sort(list(set(Ndiff)))
            self.flipnumber = len(flipset)
        else:
            pass

        ns = []
        ps = []
        bs = []
        ms = []
        for key, (_perm, _block, _m) in blocks.items():
            ns.append(key)
            if _perm.ndim == 1:
                _perm = np.array([
                    [L+i, L-a-1, 1] if i < 0 else [L-i-1, L-a-1, 0]
                    for a,i in enumerate(_perm)
                ])
            ps.append(_perm)
            bs.append(_block)
            ms.append(_m)
        self.block_name = ns
        self.perm = np.array(ps, dtype=np.int64).reshape(-1, self.L, 3)
        self.block = np.array(bs, dtype=np.int64).reshape(-1)
        self.ns = np.array(ms, dtype=np.int64).reshape(-1)

    def _validate_Ndiff(self) -> None:
        min_ndiff = -self.flipnumber
        max_ndiff = self.L - self.flipnumber
        for ndiff in self.Ndiff:
            assert ndiff in list(range(min_ndiff, max_ndiff+1)), "Ndiff should be in range(L//2)"

    def permute(self, s, which_perm):
        from .basis_core import perm_operation
        return perm_operation(s, self.perm[which_perm])


class BasisZ2N(SpinHalfGeneralBasis):
    def __init__(self, L: int, flipset, Ndiff, Nup, **blocks) -> None:
        super().__init__(L, flipset, Ndiff, Nup, **blocks)
        self._validate_block()
         
        if self.Nup2 is not None:
            from .basis_core import construct_Nup2_Z2N_basis
            self.Ns, self.s_list, self.R_list = construct_Nup2_Z2N_basis(
                self.L, self.flipmask, self.Nup2, self.perm, self.block,
            )
        elif self.Ndiff is not None:
            self._validate_Ndiff()
            from .basis_core import construct_Ndiff_Z2N_basis
            self.Ns, self.s_list, self.R_list = construct_Ndiff_Z2N_basis(
                self.L, self.flipmask, self.Ndiff, self.perm, self.block 
            ) 
        else:
            from .basis_core import construct_Z2N_basis
            self.Ns, self.s_list, self.R_list = construct_Z2N_basis(
                self.L, self.perm, self.block 
            )
        self.default_complex = False

    def _validate_block(self) -> None:
        for i, (perm1, block1) in enumerate(zip(self.perm, self.block)):
            _valdate_z2(self.L, perm1, block1)
            for j, (perm2, block2) in enumerate(zip(self.perm, self.block)):
                if i < j:
                    _valdate_z2_commute(self.L, perm1, perm2)
    
    def _validate_Ndiff(self) -> None:
        for ndiff in self.Ndiff:
            assert ndiff in list(range(-self.L, self.L+1)), "Ndiff should be in range(L//2)"
   
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_Z2N
        return single_sparse_matrix_element_Z2N(
            opnm, posn, coef, self.L, self.perm, self.block, self.Ns, self.s_list, self.R_list, 
            row_init, col_init, ME_init
        )

# class BasisZNN(SpinHalfGeneralBasis):
#     def __init__(self, L: int, flipset, Ndiff, Nup, **blocks) -> None:
#         super().__init__(L, flipset, Ndiff, Nup, **blocks)
#         # self._validate_block()
         
#         if self.Nup2 is not None:
#             from .basis_core import construct_Nup2_ZNN_basis
#             self.Ns, self.s_list, self.R_list = construct_Nup2_ZNN_basis(
#                 self.L, self.flipmask, self.Nup2, self.perm, self.block,
#             )
#         elif self.Ndiff is not None:
#             self._validate_Ndiff()
#             from .basis_core import construct_Ndiff_ZNN_basis
#             self.Ns, self.s_list, self.R_list = construct_Ndiff_ZNN_basis(
#                 self.L, self.flipmask, self.Ndiff, self.perm, self.block 
#             ) 
#         else:
#             from .basis_core import construct_ZNN_basis
#             self.Ns, self.s_list, self.R_list = construct_ZNN_basis(
#                 self.L, self.perm, self.block, self.ns
#             )
#         self.default_complex = False

    # def _validate_block(self) -> None:
    #     for i, (perm1, block1) in enumerate(zip(self.perm, self.block)):
    #         _valdate_z2(self.L, perm1, block1)
    #         for j, (perm2, block2) in enumerate(zip(self.perm, self.block)):
    #             if i < j:
    #                 _valdate_z2_commute(self.L, perm1, perm2)
    
    # def _validate_Ndiff(self) -> None:
    #     for ndiff in self.Ndiff:
    #         assert ndiff in list(range(-self.L, self.L+1)), "Ndiff should be in range(L//2)"
   
    # def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
    #     from .matrix_core import single_sparse_matrix_element_Z2N
    #     return single_sparse_matrix_element_Z2N(
    #         opnm, posn, coef, self.L, self.perm, self.block, self.Ns, self.s_list, self.R_list, 
    #         row_init, col_init, ME_init
    #     )

# class SpinHalfGeneralBasis(SpinHalfBasis):
#     def __init__(self, L: int, flipset, Ndiff, **blocks) -> None:
#         super().__init__(L)

#         if flipset is not None and Ndiff is not None:
#             self.flipmask = sum(1 << (L-1-flip) for flip in flipset)
#             if isinstance(Ndiff, int):
#                 Ndiff = [Ndiff]
#             self.Ndiff = np.sort(list(set(Ndiff)))
#             self.flipnumber = len(flipset)
#         else:
#             self.flipmask = None
#             self.Ndiff = None
#             self.flipnumber = 0

#         ns = []
#         ps = []
#         bs = []
#         for key, (_perm, _block) in blocks.items():
#             ns.append(key)
#             if _perm.ndim == 1:
#                 _perm = np.array([
#                     [L+i, L-a-1, 1] if i < 0 else [L-i-1, L-a-1, 0]
#                     for a,i in enumerate(_perm)
#                 ])
#             ps.append(_perm)
#             bs.append(_block)
#         self.block_name = ns
#         self.perm = np.array(ps, dtype=np.int64).reshape(-1, self.L, 3)
#         self.block = np.array(bs, dtype=np.int64).reshape(-1)

#     def _validate_Ndiff(self) -> None:
#         min_ndiff = -self.flipnumber
#         max_ndiff = self.L - self.flipnumber
#         for ndiff in self.Ndiff:
#             assert ndiff in list(range(min_ndiff, max_ndiff+1)), "Ndiff should be in range(L//2)"

#     def permute(self, s, which_perm):
#         from .basis_core import perm_operation
#         return perm_operation(s, self.perm[which_perm])

    

# ############################################
# # Ndiff
# ##############################################       
# class BasisNdiff(SpinHalfGeneralBasis):
#     def __init__(self, L: int, flipset, Ndiff, Nup2, **blocks) -> None:
#         """
#         参数：
#         - L (int): 系统的大小。
#         - Ndiff (Optional[int]): 
#         """
#         assert len(blocks) == 0, "Ndiff should not have Z2 symmetry"
#         super().__init__(L, flipset, Ndiff, **blocks)
#         self._validate_Ndiff()
#         if Nup2 is not None:
#             from .basis_core import construct_Nup2_basis
#             self.Ns, self.s_list = construct_Nup2_basis(self.L, self.flipmask, Nup2)
#         else:
#             from .basis_core import construct_Ndiff_basis
#             self.Ns, self.s_list = construct_Ndiff_basis(self.L, self.flipmask, self.Ndiff)
#         self.default_complex = False

#     def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
#         from .matrix_core import single_sparse_matrix_element_Nup
#         return single_sparse_matrix_element_Nup(opnm, posn, coef, self.L, self.Ns, self.s_list, row_init, col_init, ME_init)



# ############################################
# # Z21
# ##############################################
# class BasisZ21(SpinHalfGeneralBasis):
#     def __init__(self, L: int, flipset, Ndiff, Nup2, **blocks) -> None:
#         """
#         参数：
#         - L (int): 系统的大小。
#         - pblock (Optional[int]): 反演对称性块。-1 或 1。
#         """
#         assert len(blocks) == 1, "Z21 should have one Z2 symmetry"
#         super().__init__(L, flipset, Ndiff, **blocks)
#         self._validate_block()

#         if Nup2 is not None:
#             from .basis_core import construct_Nup2_Z21_basis
#             self.Ns, self.s_list = construct_Nup2_Z21_basis(self.L, self.flipmask, Nup2, self.perm[0], self.block[0])
#         elif Ndiff is not None:
#             self._validate_Ndiff()
#             from .basis_core import construct_Ndiff_Z21_basis
#             self.Ns, self.s_list = construct_Ndiff_Z21_basis(
#                 self.L, self.flipmask, self.Ndiff, self.perm[0], self.block[0]
#             )
#         else:
#             from .basis_core import construct_Z21_basis
#             self.Ns, self.s_list = construct_Z21_basis(self.L, self.perm[0], self.block[0])
        
#         self.default_complex = False

#     def _validate_block(self) -> None:
#         _valdate_z2(self.L, self.perm[0], self.block[0])
 
#     def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
#         from .matrix_core import single_sparse_matrix_element_Z21
#         # print(self.perm[0])
#         return single_sparse_matrix_element_Z21(
#             opnm, posn, coef, self.L, self.perm[0], self.block[0], 
#             self.Ns, self.s_list, row_init, col_init, ME_init
#         )
    
# ############################################
# # Z22
# ##############################################
# class BasisZ22(SpinHalfGeneralBasis):
#     def __init__(self, L: int, flipset, Ndiff, Nup2, **blocks) -> None:
#         """
#         参数：
#         - L (int): 系统的大小。
#         - pblock (Optional[int]): 反演对称性块。-1 或 1。
#         """
#         assert len(blocks) == 2, "Z22 should have two Z2 symmetries"
#         super().__init__(L, flipset, Ndiff, **blocks)
#         self._validate_block()

#         if Nup2 is not None:
#             from .basis_core import construct_Nup2_Z22_basis
#             self.Ns, self.s_list, self.R_list = construct_Nup2_Z22_basis(
#                 self.L, self.flipmask, Nup2, self.perm[0], self.block[0],
#                 self.perm[1], self.block[1]
#             )
#         elif Ndiff is not None:
#             self._validate_Ndiff()
#             from .basis_core import construct_Ndiff_Z22_basis
#             self.Ns, self.s_list, self.R_list = construct_Ndiff_Z22_basis(
#                 self.L, self.flipmask, self.Ndiff, self.perm[0], self.block[0], self.perm[1], self.block[1]
#             ) 
#         else:
#             from .basis_core import construct_Z22_basis
#             self.Ns, self.s_list, self.R_list = construct_Z22_basis(
#                 self.L, self.perm[0], self.block[0], self.perm[1], self.block[1]
#             )
#         self.default_complex = False

#     def _validate_block(self) -> None:
#         _valdate_z2(self.L, self.perm[0], self.block[0])
#         _valdate_z2(self.L, self.perm[1], self.block[1])
#         _valdate_z2_commute(self.L, self.perm[0], self.perm[1])
   
#     def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
#         from .matrix_core import single_sparse_matrix_element_Z22
#         return single_sparse_matrix_element_Z22(
#             opnm, posn, coef, self.L, self.perm[0], self.perm[1], self.block[0], self.block[1], 
#             self.Ns, self.s_list, self.R_list, row_init, col_init, ME_init
#         )

# ############################################
# # Z23
# ##############################################
# class BasisZ23(SpinHalfGeneralBasis):
#     def __init__(self, L: int, flipset, Ndiff, Nup2, **blocks) -> None:
#         """
#         参数：
#         - L (int): 系统的大小。
#         - pblock (Optional[int]): 反演对称性块。-1 或 1。
#         """
#         assert len(blocks) == 3, "Z23 should have three Z2 symmetries"
#         super().__init__(L, flipset, Ndiff, **blocks)
#         self._validate_block()
        
#         if Nup2 is not None:
#             from .basis_core import construct_Nup2_Z23_basis
#             self.Ns, self.s_list, self.R_list = construct_Nup2_Z23_basis(
#                 self.L, self.flipmask, Nup2, self.perm[0], self.block[0],
#                 self.perm[1], self.block[1], self.perm[2], self.block[2]
#             )
#         elif Ndiff is None:
#             from .basis_core import construct_Z23_basis
#             self.Ns, self.s_list, self.R_list = construct_Z23_basis(
#                 self.L, self.perm[0], self.block[0], self.perm[1], 
#                 self.block[1], self.perm[2], self.block[2]
#             )
#         else:
#             self._validate_Ndiff()
#             from .basis_core import construct_Ndiff_Z23_basis
#             self.Ns, self.s_list, self.R_list = construct_Ndiff_Z23_basis(
#                 self.L, self.flipmask, self.Ndiff, self.perm[0], 
#                 self.block[0], self.perm[1], self.block[1], 
#                 self.perm[2], self.block[2]
#             )
        
#         self.default_complex = False

#     def _validate_block(self) -> None:
#         _valdate_z2(self.L, self.perm[0], self.block[0])
#         _valdate_z2(self.L, self.perm[1], self.block[1])
#         _valdate_z2(self.L, self.perm[2], self.block[2])
#         _valdate_z2_commute(self.L, self.perm[0], self.perm[1])
#         _valdate_z2_commute(self.L, self.perm[0], self.perm[2])
#         _valdate_z2_commute(self.L, self.perm[1], self.perm[2])
 
#     def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
#         from .matrix_core import single_sparse_matrix_element_Z23
#         return single_sparse_matrix_element_Z23(
#             opnm, posn, coef, self.L, self.perm[0], self.perm[1], self.perm[2], self.block[0], 
#             self.block[1], self.block[2], self.Ns, self.s_list, self.R_list, row_init, col_init, ME_init
#         )
