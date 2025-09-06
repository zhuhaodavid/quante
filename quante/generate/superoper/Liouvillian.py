# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-14 22:03:39
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-07 02:40:50

import numpy as np
import scipy.sparse as sps

from scipy.sparse.linalg import LinearOperator, spsolve, eigsh, svds
from typing import Literal

__all__ = ['Liouvillian', 'make_Liouvillian', 'make_LiouvillianOper']

class Liouvillian(LinearOperator):
    def __init__(
        self,
        ham:sps.csr_array|None,
        lindblad_ops:list[sps.csr_array]|None,
        default_mul = 'sp'
    ):
        r"""
        The Liouvillian is given by the following equation:
        
        .. math::
            \mathcal{L}(\rho) = -i [H, \rho] + \sum_{l} L_l \rho L_l^{\dagger} - \frac{1}{2} \sum_{l} (L_l^{\dagger} L_l \rho + \rho L_l^{\dagger} L_l)
        
        where :math:`H` is the Hamiltonian, :math:`L_l` are the Lindblad operators, and :math:`\rho` is the density matrix.

        Notes
        -----
        - The Liouvillian is a linear operator that acts on the density matrix.
        - The Hamiltonian and Lindblad operators should be given in the sparse matrix format.
        - The Lindbladian can be sparse or dense, but it is recommended to use sparse matrices for large systems and dense matrices for small systems.
        
        Parameters
        ----------
        ham : sps.csr_array | None
            The Hamiltonian of the system.
        lindblad_ops : list[sps.csr_array] | None
            The Lindblad operators of the system.
        default_mul : Literal['sp', 'lo'], optional
            control how to multiply the Liouvillian with a vector, by default 'sp'
            - 'sp': use sparse matrix multiplication
            - 'lo': use linear operator multiplication
        """
        # if ham is not None:
        #     assert sps.issparse(ham), "ham must be sparse matrix"
        if ham is None and lindblad_ops is None:
            raise ValueError("ham and lindblad_ops cannot be both None")
        
        self.ham = ham
        self.lindblad_ops = lindblad_ops
        self.Ns = ham.shape[0] if ham is not None else lindblad_ops[0].shape[0]
        self.dtype = np.dtype(np.complex128)
        self.shape = (self.Ns**2, self.Ns**2)
        self.default_mul = default_mul
        self._matrix = self._ham_eff = self._sum_jump = None
    
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
    def sum_jump(self):
        """将所有的 jump operator 进行求和，得到一个稀疏矩阵"""
        if self._sum_jump is None:
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
            self._sum_jump = coodiaglists2csr(row_result=row_result, col_result=col_result, ele_result=ele_result, diag=None, n_row=self.Ns**2, index_type=np.int32, dtype=np.complex128)
        return self._sum_jump

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
    
    @property
    def matrix(self):
        if self._matrix is None:
            eye = sps.eye(self.Ns)
            nonherm = -1j * (sps.kron(self.ham_eff, eye) - sps.kron(eye, self.ham_eff.conj()))
            if self.lindblad_ops is None:
                return nonherm
            self._matrix = nonherm + self.sum_jump
        return self._matrix
    
    def to_matrix(self, sparse=True):
        if sparse:
            return self.matrix
        else:
            return self.matrix.toarray()
    
    def clear_sp(self):
        """clear the sparse matrix"""
        self._matrix = None
        self._ham_eff = None
        self._sum_jump = None
    
    def sp_mul(self, flatten_rho):
        return self.matrix @ flatten_rho
    
    def lo_mul(self, flatten_rho):
        rho = flatten_rho.reshape(self.Ns, self.Ns)
        drho_dt = -1j * (self.ham_eff @ rho - rho @ self.ham_eff.conj().T) 
        if self.lindblad_ops is None:
            return drho_dt
        for lo in self.lindblad_ops:
            drho_dt += lo @ rho @ lo.conj().T 
        return drho_dt.flatten()

    def __call__(self, rho):
        if self.default_mul == 'sp':
            return self.sp_mul(rho.flatten()).reshape(self.Ns, self.Ns)
        elif self.default_mul == 'lo':
            return  self.lo_mul(rho.flatten()).reshape(self.Ns, self.Ns)
        else:
            raise ValueError("default_mul must be 'sp' or 'lo'")

    def _matvec(self, flatten_rho):
        if self.default_mul == 'sp':
            return self.sp_mul(flatten_rho)
        elif self.default_mul == 'lo':
            return self.lo_mul(flatten_rho)
        else:
            raise ValueError("default_mul must be 'sp' or 'lo'")

    def _rmatvec(self, flatten_rho):
        if self.default_mul == 'sp':
            return flatten_rho @ self.matrix
        elif self.default_mul == 'lo':
            rho = flatten_rho.reshape(self.Ns, self.Ns)
            drho_dt = -1j * (self.ham_eff.T @ rho - rho @ self.ham_eff.conj()) 
            if self.lindblad_ops is None:
                return drho_dt.flatten()
            for lo in self.lindblad_ops:
                drho_dt += lo.T @ rho @ lo.conj()
            return drho_dt.flatten()

    def _tolo(self):
        if self.default_mul == 'sp':
            return self.matrix, None
        elif self.default_mul == 'lo':
            return self, self.trace
    
    def steady_state(self, method:Literal['direct', 'eig', 'svd'] = 'direct'):
        if method == 'direct':
            # Find the weight, to stable the iteration
            L_mat = self.matrix
            weight = np.mean(abs(L_mat.data))

            # add normalization constraint by adding a row of vec(weight*I)
            n = self.Ns
            N = n * n
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
            L = self.matrix
            n = self.Ns
            N = n * n
            # from .nbfuc.operations_numba import dot_parallel
            def LdagL_matvec(x):
                # return dot_parallel(L.conj().T, dot_parallel(L, x))
                return L.conj().T @ (L @ x)
            linop = LinearOperator((N, N), matvec=LdagL_matvec, dtype=np.complex128)
            val, vec = eigsh(linop, k=1, which='SM')
            rho = vec.reshape(self.Ns, self.Ns)
            return rho / np.trace(rho)
        elif method == 'svd':
            n = self.Ns
            N = n * n
            L_mat = self.matrix
            u, s, v = svds(L_mat, k=1, which='SM')
            rho = v.reshape(n, n)
            return rho / rho.trace()
        else:
            raise ValueError("method should be 'direct' or 'eig' or 'svd'")


def make_Liouvillian(ham, lindblad_ops, basis):
    hammat = ham.to_matrix(basis, sparse=True)
    lindmat = [lo.to_matrix(basis, sparse=True) for lo in lindblad_ops]
    return Liouvillian(hammat, lindmat)


def make_LiouvillianOper(L, ham, lindblad_ops, format:Literal['chain', 'ladder']='chain'):
    r"""Create a Liouvillian operator in a operator format on vectorized space.

    Notes
    -----
    The Liouvillian is given by the following equation:
    .. math::
        \mathcal{L}(\rho) = -i [H, \rho] + \sum_{l} L_l \rho L_l^{\dagger} - \frac{1}{2} \sum_{l} (L_l^{\dagger} L_l \rho + \rho L_l^{\dagger} L_l)
    
    where :math:`H` is the Hamiltonian, :math:`L_l` are the Lindblad operators, and :math:`\rho` is the density matrix.

    In vectorized space, the Liouvillian can be represented as:
    .. math::
        \mathcal{L} = -i (H_{eff} \otimes I - I \otimes H_{eff}^*) + \sum_{l} L_l \otimes L_l^* 
    
    where :math:`\otimes` denotes the Kronecker product, :math:`I` is the identity operator and 
    :math:`H_{eff} = H - \frac{i}{2} \sum_{l} L_l^{\dagger} L_l` is the effective non-Hermitian Hamiltonian.

    The space the operator acts on is :math:`\mathcal{H} \otimes \mathcal{H}`, where :math:`\mathcal{H}` is the Hilbert space of the system. We denote the left and right parts of the space as the first and second :math:`\mathcal{H}` respectively.

    The space is ordered as follows:
    - 'chain' format, the default order by kronecker product,
        .. code-block:: text
            |            left sites                         right sites
            |    o --- o  --- o  ---  ...  --- o      o --- o  --- o  ---  ...  --- o
            |    1     2      3                L     L+1   L+2    L+3               2L
            
    - 'ladder' format, the order is like a ladder,
        .. code-block:: text
            |
            |    1     3      5               2L-1   left sites
            |    o --- o  --- o  ---  ...  --- o
            |    o --- o  --- o  ---  ...  --- o     right sites
            |    2     4      6                2L

    Parameters
    ----------
    L : int
        The total number of sites.
    ham : Operator
        The Hamiltonian operator.
    lindblad_ops : list of Operators
        The Lindblad operators.
    format : str, optional
        The format of the Liouvillian operator, by default 'chain'.

    Returns
    -------
    Operator
        The Liouvillian operator.
    """
    res = ham.builder()

    if format == 'chain':
        # transpose ham
        for opstr, posn, coef in ham.each_term():
            # H oxx I
            res += opstr, posn, coef * (-1j)
            # I oxx H^*
            num_y = opstr.count('y')
            res += opstr, [p+L for p in posn], (-1)**num_y * np.conj(coef) * (1j)
        
        for lo in lindblad_ops:
            for opstr, posn, coef in (lo.hc() @ lo).each_term():
                # Ldag L oxx I
                res += opstr, posn, coef * (-0.5)
                # I oxx (Ldag L)^*
                num_y = opstr.count('y')
                res += opstr, [p+L for p in posn], (-1)**num_y * np.conj(coef) * (-0.5)
            
            for opstr1, posn1, coef1 in lo.each_term():
                for opstr2, posn2, coef2 in lo.each_term():
                    # L oxx L^*
                    num_y = opstr2.count('y')
                    res += (
                        opstr1 + opstr2, 
                        list(posn1) + [p+L for p in posn2], 
                        (-1)**num_y * coef1 * np.conj(coef2)
                    )
    
    elif format == 'ladder':
        for opstr, posn, coef in ham.each_term():
            # H oxx I
            res += opstr, [2*p for p in posn], coef * (-1j)
            # I oxx H^*
            num_y = opstr.count('y')
            res += opstr, [2*p+1 for p in posn], (-1)**num_y * np.conj(coef) * (1j)
        
        for lo in lindblad_ops:
            for opstr, posn, coef in (lo.hc() @ lo).each_term():
                # Ldag L oxx I
                res += opstr, [2*p for p in posn], coef * (-0.5)
                # I oxx (Ldag L)^*
                num_y = opstr.count('y')
                res += opstr, [2*p+1 for p in posn], (-1)**num_y * np.conj(coef) * (-0.5)
            
            for opstr1, posn1, coef1 in lo.each_term():
                for opstr2, posn2, coef2 in lo.each_term():
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
    else:
        raise ValueError("format should be 'chain' or 'ladder'")

    return res.build()

