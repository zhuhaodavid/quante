# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-14 22:03:39
# @Last Modified by:   hzhu
# @Last Modified time: 2025-05-14 22:04:13

import numpy as np
from typing import Literal

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
    