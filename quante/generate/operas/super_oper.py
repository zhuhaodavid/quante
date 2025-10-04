# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-22 13:10:02
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-04 17:40:47

import numpy as np
from typing import Literal
import scipy.sparse as sps
from scipy.sparse.linalg import LinearOperator, spsolve, eigsh, svds

from .spin import SpinOper

__all__ = ['LiouvilleOper', 'LiouvillianLinearOperator']

class LiouvilleOper(SpinOper):
    def __init__(
        self,
        L:int,
        ham:SpinOper,
        lind_ops:list[SpinOper],
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
        self.lindblad_ops = lind_ops
        self.flip = flip
        self.indx_order = indx_order
        self._check_liou_length()
        data = self._build_liouvillian()
        super().__init__(data)
    
    @property
    def L(self):
        return self._L
    
    def show_split(self):
        print(f"Liouvillian Operator, L={self.L}, indx_order={self.indx_order}, flip={self.flip}")
        print(f"Hamiltonian:")
        self.ham.show()
        print(f"Lindblad operators:")
        for i, lo in enumerate(self.lindblad_ops):
            print(f"  L_{i}:")
            lo.show()
    
    def show(self):
        print(f"Liouvillian Operator in vectorized space, L={self.L}, indx_order={self.indx_order}, flip={self.flip}")
        super().show()

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
        if basis.L == self.L:
            res = self.to_linearoperator(basis, pauli).to_matrix()
            return res if sparse else res.toarray()
        elif basis.L == 2 * self.L:
            return super().to_matrix(basis, pauli, sparse)
        else:
            raise ValueError(f"basis.L {basis.L} does not match Liouvillian L {self.L} or {2*self.L}")
    
    def nonhermitian_part(self):
        r"""Get the non-Hamiltonian part of the Liouvillian.

        .. math::
            \mathcal{L}_{non-H} = H - \frac{i}{2} \sum_{l} L_l^{\dagger} L_l

        Parameters
        ----------
        pauli : bool
            Whether to clean the operator using Pauli basis.

        Returns
        -------
        SpinOper
            The non-Hamiltonian part of the Liouvillian.
        """
        Heff = self.ham.copy()
        for lo in self.lindblad_ops:
            Heff += (-1j/2) * (lo.hc() @ lo)
        return Heff
    
    def to_linearoperator(self, basis, pauli:bool, sparse=True):
        from ..basis.spin_half.spin_1d.basis import SpinHalf1DBasis
        if not isinstance(basis, SpinHalf1DBasis):
            raise ValueError("need basis in Hilbert space not the vectorized space")
        
        self._check_pauli(pauli)
        ham = self.ham.to_matrix(basis=basis, pauli=pauli, sparse=sparse)
        lindblad_ops = [lo.to_matrix(basis=basis, pauli=pauli, sparse=sparse) for lo in self.lindblad_ops]
        return LiouvillianLinearOperator(ham=ham, lindblad_ops=lindblad_ops)
    
    def steady_state(self, basis, pauli, method:Literal['direct', 'eig', 'svd'] = 'direct'):
        assert method in ['direct', 'eig', 'svd'], "method should be 'direct' or 'eig' or 'svd'"
        L_mat = self.to_matrix(basis, pauli, sparse=True)
        return steady_state(L_mat, method=method)

def _flip_opstr(opstr, coef):
    assert 'n' not in opstr, "flip must be False when 'n' operator is present"
    opstr = opstr.replace('p', '#').replace('m', 'p').replace('#', 'm')
    num_yzZ = opstr.count('y') + opstr.count('z') + opstr.count('Z')
    coef = (-1)**num_yzZ * coef
    return opstr, coef

class LiouvillianLinearOperator(LinearOperator):
    def __init__(
        self,
        ham:sps.csr_array|None,
        lindblad_ops:list[sps.csr_array]|None,
    ):
        r"""
        The Liouvillian is given by the following equation:
        
        .. math::
            \mathcal{L}(\rho) = -i [H, \rho] + \sum_{l} L_l \rho L_l^{\dagger} - \frac{1}{2} \sum_{l} (L_l^{\dagger} L_l \rho + \rho L_l^{\dagger} L_l)
        
        where :math:`H` is the Hamiltonian, :math:`L_l` are the Lindblad operators, and :math:`\rho` is the density matrix.
        """
        if ham is None and lindblad_ops is None:
            raise ValueError("ham and lindblad_ops cannot be both None")
        
        self.ham = ham
        self.lindblad_ops = lindblad_ops
        self.Ns = ham.shape[0] if ham is not None else lindblad_ops[0].shape[0]
        self.dtype = np.dtype(np.complex128)
        self.shape = (self.Ns**2, self.Ns**2)
        self._ham_eff = None
    
    @property
    def ham_eff(self):
        if self._ham_eff is None:
            if self.ham is None:
                tmp = 0
                for lo in self.lindblad_ops:
                    if sps.issparse(lo) and lo.nnz == 0:
                        continue
                    tmp = tmp + lo.conj().T @ lo
                self._ham_eff =  - 1j * tmp/2
            elif self.lindblad_ops is None:
                self._ham_eff = self.ham
            else:
                tmp = 0
                for lo in self.lindblad_ops:
                    if sps.issparse(lo) and lo.nnz == 0:
                        continue
                    tmp = tmp + lo.conj().T @ lo
                self._ham_eff = self.ham - 1j * tmp/2
        return self._ham_eff

    @property
    def trace(self):
        a = 2 * self.Ns * self.ham_eff.trace().imag
        if self.lindblad_ops is None:
            return a
        b = sum(abs(lo.trace())**2 for lo in self.lindblad_ops)
        res = a + b
        if isinstance(res, np.ndarray):
            return res.item()
        else:
            return res
   
    def lo_mul(self, flatten_rho):
        rho = flatten_rho.reshape(self.Ns, self.Ns)
        drho_dt = -1j * (self.ham_eff @ rho - rho @ self.ham_eff.conj().T) 
        if self.lindblad_ops is None:
            return drho_dt
        for lo in self.lindblad_ops:
            drho_dt += lo @ rho @ lo.conj().T 
        return drho_dt.flatten()

    def __call__(self, rho):
        return  self.lo_mul(rho.flatten()).reshape(self.Ns, self.Ns)

    def _matvec(self, flatten_rho):
        return self.lo_mul(flatten_rho)

    def _rmatvec(self, flatten_rho):
        rho = flatten_rho.reshape(self.Ns, self.Ns)
        drho_dt = -1j * (self.ham_eff.T @ rho - rho @ self.ham_eff.conj()) 
        if self.lindblad_ops is None:
            return drho_dt.flatten()
        for lo in self.lindblad_ops:
            drho_dt += lo.T @ rho @ lo.conj()
        return drho_dt.flatten()
 
    def _sum_jump(self):
        """将所有的 jump operator 进行求和，得到一个稀疏矩阵"""
        if self.lindblad_ops is None:
            return None
        # self._sum_jump = sum(sps.kron(lo, lo.conj()) for lo in self.lindblad_ops)
        # 如果 lo 比较多且简单，那么下面的方法会更高效（占用内存会更多）
        from ..basis.basis_class_nb import coodiaglists2csr
        row_result = []
        col_result = []
        ele_result = [] 
        for lo in self.lindblad_ops:
            tmp = sps.kron(lo, lo.conj())
            row_result.append(tmp.row)
            col_result.append(tmp.col)
            ele_result.append(tmp.data)
        return coodiaglists2csr(row_result=row_result, col_result=col_result, ele_result=ele_result, diag=None, n_row=self.Ns**2, index_type=np.int32, dtype=np.complex128)
   
    def to_matrix(self):
        eye = sps.eye(self.Ns)
        nonherm = -1j * (sps.kron(self.ham_eff, eye) - sps.kron(eye, self.ham_eff.conj()))
        if self.lindblad_ops is None:
            return nonherm
        return nonherm + self._sum_jump()
    
    def strady_sate(self, method:Literal['direct', 'eig', 'svd'] = 'direct'):
        assert method in ['direct', 'eig', 'svd'], "method should be 'direct' or 'eig' or 'svd'"
        mat = self.to_matrix()
        return steady_state(mat, method=method)
   
def steady_state(L_mat, method:Literal['direct', 'eig', 'svd'] = 'direct'):
    N = L_mat.shape[0]
    n = int(np.sqrt(N))
    assert N == n * n, "L_mat shape is not a square of an integer"
    if method == 'direct':
        # Find the weight, to stable the iteration
        weight = np.mean(abs(L_mat.data))

        # add normalization constraint by adding a row of vec(weight*I)
        # Create an n x n sparse matrix with the first row as (weight * I).reshape(1, -1), others are zeros
        eye_row = sps.lil_array((N, N))
        eye_row[0, :] = (sps.eye(n, format='lil') * weight).reshape(1, -1)
        L_mat_aug = L_mat + eye_row.tocsr()

        # initial guess
        x0 = np.zeros((N, 1), dtype=np.complex128)
        x0[0, 0] = weight

        out = spsolve(L_mat_aug, x0)
        return out.reshape(n, n)
    elif method == 'eig':
        # from .nbfuc.operations_numba import dot_parallel
        def LdagL_matvec(x):
            # return dot_parallel(L.conj().T, dot_parallel(L, x))
            return L_mat.conj().T @ (L_mat @ x)
        linop = LinearOperator((N, N), matvec=LdagL_matvec, dtype=np.complex128)
        val, vec = eigsh(linop, k=1, which='SM')
        rho = vec.reshape(L_mat.Ns, L_mat.Ns)
        return rho / np.trace(rho)
    elif method == 'svd':
        u, s, v = svds(L_mat, k=1, which='SM')
        rho = v.reshape(n, n)
        return rho / rho.trace()
    else:
        raise ValueError("method should be 'direct' or 'eig' or 'svd'")


# def make_Liouvillian(ham, lindblad_ops, basis, pauli):
#     hammat = ham.to_matrix(basis, sparse=True, pauli=pauli)
#     lindmat = [lo.to_matrix(basis, sparse=True, pauli=pauli) for lo in lindblad_ops]
#     return Liouvillian(hammat, lindmat)

