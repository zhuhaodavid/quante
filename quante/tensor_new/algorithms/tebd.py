# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2024-12-11 11:25:59
# @Last Modified by:   dzwang
# @Last Modified time: 2024-12-13 10:29:54
import numpy as np
from ..networks import MPS
from ...generate.operas import Oper


__all__ = ["TEBDEngine"]


class TEBDEngine:
    def __init__(self, psi:MPS, model:Oper, options:dict) -> None:
        self.psi = psi
        self.model = model
        self.options = options
        pass
    
    def run(self, pos:list[int], gates:list[np.ndarray]):
        """Unitary evolution 
        """        
        trunc_err = None
        pos_pre = 0
        for pos_cur, gate in zip(pos, gates):
            direction = 1 if pos_cur>=pos_pre else 0
            trunc_err += self.update_bond(i=pos_cur, U_bond=gate, direction=direction)
        
        return trunc_err
    
    def update_bond(self, i:int, U_bond:np.ndarray, direction:int=0):
        """
        .. code-block:: text
        
        theta
                |        |               
                ├-U_bond-┤              
                |        |    
            ----⬜--------⨞---
                i        j

        Args:
            i (int): we update the matrices at sites "i, i+1"
            U_bond (np.ndarray): bond opeartor matrix
        """
        self.psi.set_mixed_canonical_form(i, )
        
        j = i+1  # todo infinite
        theta = self.psi.contract_U_bond_mps(self.psi.Ws[i], self.psi.Ws[j], U_bond)
        
        self.psi.Ws[i], self.psi.Ws[j] = self.psi.update_two_site(i, theta, direction)
        self.llim += direction
        self.rlim += direction
        
        trunc_error = None
        return trunc_error
    
    def run_GS(self,):
        """non_Unitary evolution
        """
        pass
