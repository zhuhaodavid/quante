# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-22 13:10:02
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-02 00:47:27

import numpy as np
from typing import Literal
from .spin import SpinOper

class LiouvilleOper(SpinOper):
    def __init__(
        self,
        L:int,
        ham:SpinOper,
        lindblad_ops:list[SpinOper],
        indx_order:Literal['stacked', 'snake']='stacked', 
        flip:bool=False
    ):
        r"""
        The Liouvillian is given by the following equation:
        
        .. math::
            \mathcal{L}(\rho) = -i [H, \rho] + \sum_{l} L_l \rho L_l^{\dagger} - \frac{1}{2} \sum_{l} (L_l^{\dagger} L_l \rho + \rho L_l^{\dagger} L_l)
        
        where :math:`H` is the Hamiltonian, :math:`L_l` are the Lindblad operators, and :math:`\rho` is the density matrix.

        In vectorized space, the Liouvillian can be represented as:
        .. math::
            \mathcal{L} = -i (H_{eff} \otimes I - I \otimes H_{eff}^*) + \sum_{l} L_l \otimes L_l^* 
        
        where :math:`\otimes` denotes the Kronecker product, :math:`I` is the identity operator and 
        :math:`H_{eff} = H - \frac{i}{2} \sum_{l} L_l^{\dagger} L_l` is the effective non-Hermitian Hamiltonian.

        The space the operator acts on is :math:`\mathcal{H} \otimes \mathcal{H}`, where :math:`\mathcal{H}` is the Hilbert space of the system. 
        We denote the left and right parts of the space as the first and second :math:`\mathcal{H}` respectively.

        The space is ordered as follows:
        - 'stacked' format, the default order by kronecker product,
            .. code-block:: text
                |
                |    0     1      2               L-1
                |    o --- o  --- o  ---  ...  --- o      right sites
                |    o --- o  --- o  ---  ...  --- o      left sites
                |    L    L+1    L+2              2L-1


        - 'snake' format, the order is like a snake,
            .. code-block:: text
                |
                |    0     2      4               2L-2   left sites
                |    o --- o  --- o  ---  ...  --- o
                |    o --- o  --- o  ---  ...  --- o     right sites
                |    1     3      5               2L-1
            

        Parameters
        ----------
        L : int
            The total number of sites.
        ham : Operator
            The Hamiltonian operator.
        lindblad_ops : list of Operators
            The Lindblad operators.
        indx_order : str, optional
            The format of the Liouvillian operator, by default 'chain'.
            Options are 'stacked' and 'snake'.
        flip : bool, optional
            Whether to flip the operators in the right part, by default False.
            If True, the operators are flipped as follows:
            - 'x' -> 'x'
            - 'y' -> '-y'
            - 'z' -> '-z'
            - 'p' -> 'm'
            - 'm' -> 'p'
            - 'n' -> 'n'
        
        Returns
        -------
        Operator
            The Liouvillian operator in the specified format.
        """
        self._L = L
        self.ham = ham
        self.lindblad_ops = lindblad_ops
        self.flip = flip
        self.indx_order = indx_order
        self._check_liou_length()
        data = self._build_liouvillian()
        super().__init__(data)
    
    @property
    def L(self):
        return self._L
    
    def _check_liou_length(self):
        if self.L < self.ham.L:
            raise ValueError(f"ham length {self.ham.L} exceeds L {self.L}")
        for lo in self.lindblad_ops:
            if self.L < lo.L:
                raise ValueError(f"lindblad_ops length {lo.L} exceeds L {self.L}")
    
    def _build_liouvillian(self):
        if self.indx_order == 'stacked':
            return self._build_liouvillian_stacked()
        elif self.indx_order == 'snake':
            return self._build_liouvillian_snake()
        else:
            raise ValueError(f"indx_order should be 'stacked' or 'snake', but not {self.indx_order}")

    def _build_liouvillian_stacked(self):
        L = self.L
        ham = self.ham
        flip = self.flip
        lindblad_ops = self.lindblad_ops

        res = self.builder()
        # transpose ham
        for opstr, posn, coef in ham.each_term():
            # H oxx I
            res += opstr, posn, coef * (-1j)
            if flip:
                opstr, coef = _flip_opstr(opstr, coef)
            # I oxx H^*
            num_y = opstr.count('y')
            res += opstr, [p+L for p in posn], (-1)**num_y * np.conj(coef) * (1j)
        
        for lo in lindblad_ops:
            for opstr, posn, coef in (lo.hc() @ lo).each_term():
                # Ldag L oxx I
                res += opstr, posn, coef * (-0.5)
                if flip:
                    opstr, coef = _flip_opstr(opstr, coef)
                # I oxx (Ldag L)^*
                num_y = opstr.count('y')
                res += opstr, [p+L for p in posn], (-1)**num_y * np.conj(coef) * (-0.5)
            
            for opstr1, posn1, coef1 in lo.each_term():
                for opstr2, posn2, coef2 in lo.each_term():
                    if flip:
                        opstr2, coef2 = _flip_opstr(opstr2, coef2)
                    # L oxx L^*
                    num_y = opstr2.count('y')
                    res += (
                        opstr1 + opstr2, 
                        list(posn1) + [p+L for p in posn2], 
                        (-1)**num_y * coef1 * np.conj(coef2)
                    )
    
        return res._build_dict()
    
    def _build_liouvillian_snake(self):
        ham = self.ham
        lindblad_ops = self.lindblad_ops
        flip = self.flip

        res = self.builder()
        for opstr, posn, coef in ham.each_term():
            # H oxx I
            res += opstr, [2*p for p in posn], coef * (-1j)
            if flip:
                opstr, coef = _flip_opstr(opstr, coef)
            # I oxx H^*
            num_y = opstr.count('y')
            res += opstr, [2*p+1 for p in posn], (-1)**num_y * np.conj(coef) * (1j)
        
        for lo in lindblad_ops:
            for opstr, posn, coef in (lo.hc() @ lo).each_term():
                # Ldag L oxx I
                res += opstr, [2*p for p in posn], coef * (-0.5)
                if flip:
                    opstr, coef = _flip_opstr(opstr, coef)
                # I oxx (Ldag L)^*
                num_y = opstr.count('y')
                res += opstr, [2*p+1 for p in posn], (-1)**num_y * np.conj(coef) * (-0.5)
            
            for opstr1, posn1, coef1 in lo.each_term():
                for opstr2, posn2, coef2 in lo.each_term():
                    if flip:
                        opstr2, coef2 = _flip_opstr(opstr2, coef2)
                    # L oxx L^*
                    num_y = opstr2.count('y')
                    newoper = ''.join(
                        opstr1[i] if j == 0 else opstr2[i] 
                        for i in range(len(opstr1))
                        for j in range(2)
                    )
                    newposn = [
                        2*posn1[i] if j == 0 else 2*posn2[i] + 1
                        for i in range(len(opstr1))
                        for j in range(2)
                    ]
                    res += (
                        newoper,
                        newposn,
                        (-1)**num_y * coef1 * np.conj(coef2)
                    )
        return res._build_dict()

    def sym_asym_split(
        self, 
        pauli:bool,
    ) -> tuple[SpinOper, SpinOper]:
        """Split the Liouvillian into its symmetric and antisymmetric parts.

        Parameters
        ----------
        pauli : bool
            Whether to clean the operator using Pauli basis.
        indx_order : Literal['stacked', 'snake'], optional
            The type of splitting to be performed, by default 'stacked'
        flip : bool, optional
            Whether to flip the operators in the right part, by default False.
            If True, the operators are flipped as follows:
            - 'x' -> 'x'
            - 'y' -> '-y'
            - 'z' -> '-z'
            - 'p' -> 'm'
            - 'm' -> 'p'
            - 'n' -> 'n'
        
        Returns
        -------
        tuple[SpinOper, SpinOper]
            The symmetric and antisymmetric parts of the Liouvillian.
        """
        self._check_pauli(pauli)
        L = self.L
        # first do vertical reflection
        if self.indx_order == 'stacked':
            perm = np.flipud(np.arange(2*L).reshape(2,-1)).flatten()
        elif self.indx_order == 'snake':
            perm = np.flipud(np.arange(2*L).reshape(-1,2)).T.flatten()
        if self.flip:
            perm = - (perm + 1)
        oper_reflacted = self.transform(perm)
        oper_sym = self + oper_reflacted
        oper_sym *= 0.5
        oper_antisym = self - oper_reflacted
        oper_antisym *= 0.5
        return oper_sym.clean(pauli=pauli), oper_antisym.clean(pauli=pauli)
    
    def to_matrix(self, basis, pauli, sparse=False):
        from ..basis.spin_half.spin_super.basis import SpinHalfSuperBasis
        if isinstance(basis, SpinHalfSuperBasis):
            liou_sym, liou_asym = self.sym_asym_split(pauli=True)
            liou_sym_list, sym_complex = liou_sym._convert_to_quick_form(2*self.L)
            liou_asym_list, asym_complex = (1j*liou_asym)._convert_to_quick_form(2*self.L)
            assert not sym_complex, "sym part should be real"
            assert not asym_complex, "asym part should be real"
            mat0 = basis._real_sparse_matrix(liou_sym_list, liou_asym_list, False)
            return mat0 if sparse else mat0.toarray()
        return super().to_matrix(basis, pauli, sparse)


def _flip_opstr(opstr, coef):
    assert 'n' not in opstr, "flip must be False when 'n' operator is present"
    opstr = opstr.replace('p', '#').replace('m', 'p').replace('#', 'm')
    num_yzZ = opstr.count('y') + opstr.count('z') + opstr.count('Z')
    coef = (-1)**num_yzZ * coef
    return opstr, coef

