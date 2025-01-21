# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2024-12-11 11:25:59
# @Last Modified by:   dzwang
# @Last Modified time: 2025-01-12 16:42:40
import numpy as np
from ...generate.operas import Oper
from ..networks.mps import MPS


__all__ = ["TEBDEngine"]


class TEBDEngine:
    """Time Evolving Block Decimation (TEBD) algorithm.
    """
    def __init__(self, psi:MPS, model:Oper) -> None:
        self.psi = psi
        self.model = model
        
    def run_GS(self,) -> None:
        """TEBD algorithm in imaginary time to find the ground state.
        """
        pass
    
    def evolve(self, N_steps:int, dt:float):
        """Evolve by `` N_steps * dt``.
        """
        # for j, (pos_cur, gate) in enumerate(zip(pos, gates)):
        #     direction = 1 if pos_cur<=pos[j+1] else 0
        #     self.update_bond(i=pos_cur, U_bond=gate, direction=direction)
        self.evolve_step()

    def evolve_step(self):
        self.update_bond()
        pass
    
    def update_bond(self, i:int, U_bond:np.ndarray, direction:int=0):
        """ Update bond sites MPSs
        .. code-block:: text
        
        theta
                |        |               
                ├-U_bond-┤              
                |        |    
            ----⬜--------⨞---
                i        j

        Parameters
        ----------
        i : int
            we update the matrices at sites "i, i+1"
        U_bond : np.ndarray
            bond operator matrix
        direction : int, optional
            "1" means shift OC from left to right;
            "0" means OC does not move; by default 0

        Returns
        -------
        _type_
            _description_
        """
        self.psi.set_mixed_canonical_form(i)
        j = i+1
        theta = self.psi.contract_U_bond_mps(U_bond, self.psi.Ws[i], self.psi.Ws[j])
        self.psi.Ws[i], self.psi.Ws[j], trunc_err = self.psi.update_two_site(theta, direction)
        self.psi.llim += direction
        self.psi.rlim += direction
        
    def update_imag(self,):
        self.update_bond_imag()
        pass
    
    def update_bond_imag(self):
        pass


    def contract_U_bond_mps(U_bond:np.ndarray, W1:np.ndarray, W2:np.ndarray) -> np.ndarray:
        """
        .. code-block:: text
            
                |         |
               (c)       (f)
                |         |
                ├-gate2_b-┤
                |         |                       |     |   
               (b)       (e)                     (c)   (f)    
                |         |                       |     |    
         --(a)--◻---(d)---⨞--(g)--  ----> --(a)---◻-----◻---(g)-- : theta
                W1        W2                     W1p   W2p
        
        >>> tc.einsum("abd,deg,cfbe->acfg", W1, W2, gate_2b)
        """
        a, b, d = W1.shape
        d, e, g = W2.shape
        W = W1.reshape(-1, d) @ W2.reshape(d, -1)
        W = W.reshape(a,b,e,g).transpose([1,2,0,3]).reshape(b*e, -1)
        W = U_bond @ W
        theta = W.reshape(b,e,a,g).transpose([2,0,1,3]).reshape(a,b,e,g)
        return theta
    
    


def update_two_site(theta, direction:int, 
                    solver="svd",
                    trunc_para:tuple[int,float,float]=(None,None,None),
                    normalize=False,
                    pertube=None,
                    ) -> tuple[None, None]:
    # direction = 0 (left) or 1 (right)
    u, s, vt, trunc_err = svd_tensor(theta, trunc_para=trunc_para)

    if direction == 1:
        W1p = u
        W2p = np.einsum("i,ijk->ijk", s, vt)  # todo need to check
    else:
        W1p = u * s  # todo need to check
        W2p = vt

    return W1p, W2p, trunc_err  
