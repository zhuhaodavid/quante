# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-14 22:03:39
# @Last Modified by:   dzwang
# @Last Modified time: 2025-05-18 23:06:52

import numpy as np
import scipy.sparse as sps

from scipy.sparse.linalg import LinearOperator, spsolve, eigsh, svds
from typing import Literal, Callable
from tqdm import tqdm


__all__ = ['SuperOperator', 'Liouvillian']


class SuperOperator:
    # todo: support sparse in different form
    def __init__(self,
                 msmts:list[np.ndarray],
                 msmts2:list[np.ndarray] = None,
                 order:Literal['c', 'f'] = 'c'
        ):
        self.msmts = msmts
        if msmts2 is not None:
            self.msmts2 = msmts2
        else:
            self.msmts2 = [msmt.conj().T for msmt in msmts]
        self.d = msmts[0].shape[0]
        self.order = order
    
    def __call__(self, rho:np.ndarray) -> np.ndarray:
        res = np.zeros_like(rho)
        for msmt1, msmt2 in zip(self.msmts, self.msmts2):
            res += msmt1 @ rho @ msmt2
        return res
    
    def vectorize(self) -> np.ndarray:
        r"""
        Vectorize the super operator.

        Returns
        -------
        np.ndarray
            The vectorized super operator.

        Diagram
        -------
        .. code-block:: none
            .                        ┌─┐ 
                           -- msmt  -┤ │           
            \sum_{msmt}              │ │
                           -- msmt* -┤ │
                                     └─┘
                                     rho
        
        Notes
        -----
        The vectorization is done in the following way:
        - For 'c' order, the action on `rho` is given by:
        >>> (super_oper @ rho.reshape(-1)).reshape(d,d)
        - For 'f' order, the action on `rho` is given by:
        >>> (super_oper @ rho.reshape(-1, order='f')).reshape(d,d, order='f')
        """
        if self.order == 'c':
            return sum(np.kron(msmt1, msmt2.T) for msmt1, msmt2 in zip(self.msmts, self.msmts2))
        elif self.order == 'f':
            return sum(np.kron(msmt2.T, msmt1) for msmt1, msmt2 in zip(self.msmts, self.msmts2))
        else:
            raise ValueError("order must be 'c' or 'f'")
    
    def choi_form(self):
        r"""
        Get the Choi form of the super operator.

        Returns
        -------
        np.ndarray
            The Choi form of the super operator.
        
        Diagram
        -------
        .. code-block:: none
            .   -- ┌─────┐ ---------
                   │  C  │     
               ╭-- └─────┘ -- rho --╮
               ╰--------------------╯
               
        Notes
        -----
        The action of the choi operator `C` on `rho` is given by:
        .. math::
            \tr_{2} C (I \otimes \rho) 
        
        - For 'c' order, the action on `rho` is given by:
        >>> np.einsum("ijkj->ik", (choi_oper @ np.kron(np.eye(d), rho)).reshape(d,d,d,d))
        - For 'f' order, the action on `rho` is given by:
        >>> np.einsum("ijkj->ik", (choi_oper @ np.kron(rho.T, np.eye(d))).reshape(d,d,d,d,order='f'))
        
        This achieved by,
        .. code-block:: none
            .                         
                           -- msmt  --   ---------
            \sum_{msmt}                ╳
                          ╭-- msmt† --   -- rho --╮
                          ╰-----------------------╯
        
        """
        d = self.d
        if self.order == 'c':
            tmp = sum(np.kron(msmt1, msmt2) for msmt1, msmt2 in zip(self.msmts, self.msmts2))
            return tmp.reshape(d,d,d,d).swapaxes(3, 2).reshape(self.d**2, self.d**2)
        elif self.order == 'f':
            tmp = sum(np.kron(msmt2.T, msmt1) for msmt1, msmt2 in zip(self.msmts, self.msmts2))
            return tmp.reshape(d,d,d,d,order='f').transpose(0,2,1,3).reshape(d**2,d**2,order='f')
        else:
            raise ValueError("order must be 'c' or 'f'")
    
    def kraus_form(self, threshold = 1e-10):
        r"""
        Get the Kraus form of the super operator.

        The Hermitian condition is required

        Diagram
        -------
        .. code-block:: none
            \sum_{K}  -- K -- rho -- K† ---
            
        Notes
        -----
        The measurements in this form `K` is orthogonal!
        The action of the choi operator `C` on `rho` is given by:
        .. math::
            \sum_K K \rho K^{\dagger}
        
        - For both 'c'/'f' order, the action on `rho` is given by:
        >>> res = np.zeros((2,2), dtype=complex)
        >>> for kraus_oper in kraus_opers:
        >>>     res += kraus_oper @ rho @ kraus_oper.conj().T
        """
        d = self.d
        if self.order == 'c':
            tmp = sum(np.kron(msmt1, msmt2.T) for msmt1, msmt2 in zip(self.msmts, self.msmts2))
            tmp = tmp.reshape(d,d,d,d).swapaxes(1,2).reshape(d**2,d**2)
            if not np.allclose(tmp, tmp.conj().T):
                raise ValueError("not hermitian")
            val, vec = np.linalg.eigh(tmp)
            return [np.sqrt(vali) * vec[:, i].reshape(2,2)
                for i, vali in enumerate(val) if abs(vali) > threshold]
        elif self.order == 'f':
            tmp = sum(np.kron(msmt2.T, msmt1) for msmt1, msmt2 in zip(self.msmts, self.msmts2))
            tmp = tmp.reshape(d,d,d,d,order='f').transpose(0,2,1,3).reshape(d**2,d**2,order='f')
            if not np.allclose(tmp, tmp.conj().T):
                raise ValueError("ndXY is not hermitian")
            val, vec = np.linalg.eigh(tmp)
            return [np.sqrt(vali) * vec[:, i].reshape(2,2,order='f') 
                            for i, vali in enumerate(val) if abs(vali) > threshold]           
        else:
            raise ValueError("order must be 'c' or 'f'")
    
    def stinespring_form(self, threshold = 1e-10):
        r"""
        Get the Stinesping form of the super operator.

        use this when Hermitian condition is not satisfied!!!
        like `XAY` with `X != Y†`

        Returns
        -------
        np.ndarray, np.ndarray
            The Stinesping form of the super operator.
        
        Diagram
        -------
        .. code-block:: none
               ╭-------------------------╮
               ╰-- ┌───┐         ┌───┐ --╯
                   │ A ├-- rho --┤ B │
                -- └───┘         └───┘ --
               
        Notes
        -----
        The action of the choi operator `A`, `B` on `rho` is given by:
        .. math::
            \tr_{2} (A rho B)
        
        - For 'c' order, the action on `rho` is given by:
        >>> np.einsum('ijik->jk', (A @ ndmat @ B).reshape(d,d,d,d))
        - For 'f' order, the action on `rho` is given by:
        >>> np.einsum("ijkj->ik", (A @ ndmat @ B).reshape(d,d,d,d,order='f'))
        """
        d = self.d
        tmp = sum([np.kron(msmt1, msmt2) for msmt1, msmt2 in zip(self.msmts, self.msmts2)
            ]).reshape(d,d,d,d).swapaxes(1,2).reshape(d**2,d**2)
        U, S, V = np.linalg.svd(tmp)
        nonzero_idxs = S > threshold
        dK = nonzero_idxs.sum()
        S = np.sqrt(S[nonzero_idxs])
        U = (U[:, nonzero_idxs] * S.reshape(1, -1)).reshape(d, d, dK)
        V = (V[nonzero_idxs, :] * S.reshape(-1, 1)).conj().T.reshape(d, d, dK)
        A = np.zeros((d, d*dK), dtype=complex)
        B = np.zeros((d, d*dK), dtype=complex)
        for i in range(dK):
            tmp = np.zeros((dK), dtype=complex)
            tmp[i] = 1.
            A += np.kron(U[:, :, i], tmp)
            B += np.kron(V[:, :, i], tmp)
        if self.order == 'c':
            A = A.reshape(d,d,dK).transpose(2,0,1).reshape(d*dK, d)
            B = B.reshape(d,d,dK).transpose(2,1,0).reshape(d*dK, d)
        elif self.order == 'f':
            A = A.reshape(d,d,dK).transpose(0,2,1).reshape(d*dK, d)
            B = B.reshape(d,d,dK).transpose(1,2,0).reshape(d*dK, d)
        return A, B.conj().T
    


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
        if ham is not None:
            assert sps.issparse(ham), "ham must be sparse matrix"
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
                self._ham_eff = - 1j * sum(lo.conj().T @ lo for lo in self.lindblad_ops)/2
            elif self.lindblad_ops is None:
                self._ham_eff = self.ham
            else:
                self._ham_eff = self.ham - 1j * sum(lo.conj().T @ lo for lo in self.lindblad_ops)/2
        return self._ham_eff

    @property
    def sum_jump(self):
        """将所有的 jump operator 进行求和，得到一个稀疏矩阵"""
        if self._sum_jump is None:
            if self.lindblad_ops is None:
                return None
            # self._sum_jump = sum(sps.kron(lo, lo.conj()) for lo in self.lindblad_ops)
            # 如果 lo 比较多且简单，那么下面的方法会更高效（占用内存会更多）
            from ..generate.basis.symmetry.basis_class_nb import coodiaglists2csr
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

    def toarray(self):
        return self.matrix.toarray()

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
            # from .usenumba.operations_numba import dot_parallel
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

    def trajectory_measure(
        self, 
        inistate:np.ndarray, 
        tlist:list|np.ndarray, 
        measure:Callable|np.ndarray,
        method:Literal['mul-cpu', 'mul-cuda:0', 'linear_operator',
                    'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']='mul-cpu',
        **kwargs
    ):
        for t in tqdm(tlist, ascii=True):

            pass
        # it should look like,
        
        # def integrate(self, t, copy=False):
        #     t_old, y_old = self._integrator.get_state(copy=False)
        #     norm_old = self._prob_func(y_old)
        #     while t_old < t:
        #         t_step, state = self._integrator.mcstep(t, copy=False)
        #         norm = self._prob_func(state)
        #         if norm <= self.target_norm:
        #             t_col, state = self._find_collapse_time(norm_old, norm,
        #                                                     t_old, t_step)
        #             self._do_collapse(t_col, state)
        #             t_old, y_old = self._integrator.get_state(copy=False)
        #             norm_old = 1.
        #         else:
        #             t_old, y_old = t_step, state
        #             norm_old = norm

        #     return t_old, _data.mul(y_old, 1 / self._norm_func(y_old))

        # def run(self, tlist):
        #     for t in tlist[1:]:
        #         yield self.integrate(t, False)
    
        # reference https://qutip.org/docs/4.7/guide/dynamics/dynamics-monte.html
