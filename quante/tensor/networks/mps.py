# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-04-19 14:51:03
# @Last Modified by:   dzwang
# @Last Modified time: 2025-04-20 15:24:07
import numpy as np
from quante.tensor import TensorTrain 
from quante.basicfun import println


__all__ = ['MPS']


class MPS(TensorTrain):
    def __init__(self, Ws:list[np.ndarray], llim:int, rlim:int) -> None:
        super().__init__(Ws, llim, rlim)
        
        # properties

    
    def to_vector(self) -> np.ndarray:
        """Convert the MPS to a vector.
        """
        vector = self.to_matrix().squeeze()
        return vector

    @classmethod
    def generate_W_state(cls, N:int, type:str="standard", dtype=np.float64) -> "MPS":
        """ Generate MPS representing W state.

        Parameters
        ----------
        N : int
            Number of sites
        type : str, optional
            standard W state or dual W state
            
        Example
        -------
        >>> N = 4
        >>> mps = qt.tensor.networks.MPS.generate_W_state(N, type="dual", dtype=dtype) 
        >>> mps_vec = mps.to_vector()
        >>> dir_vec = qt.generate.state.w(L=N).astype(mps.dtype).squeeze() # one down spin
        >>> print(np.allclose(mps_vec, dir_vec))
        """
        # judge the type of W state
        if type == "standard":
            index = 1
        elif type == "dual":
            index = 0
        else:
            raise ValueError("'type' must be 'standard' or 'dual'")        
        # initial zero tensor
        Wl = np.zeros((1, 2, 2), dtype=dtype)
        Wm = np.zeros((2, 2, 2), dtype=dtype)
        Wr = np.zeros((2, 2, 1), dtype=dtype)
        # construct three type tensors of W state
        ## left most tensor
        Wl[0, index,  0] = 1.
        Wl[0, 1-index,1] = 1.
        ## middle tensor
        Wm[0, index,  0] = 1.
        Wm[0, 1-index,1] = 1.
        Wm[1, index,  1] = 1.
        ## right most tensor
        Wr[0, 1-index,0] = 1.
        Wr[1, index,  0] = 1.
        ## renormalize
        Wl *= 1/np.sqrt(N)
        # generate MPS
        println(N)
        if N == 2:
            Ws = [Wl, Wr]
        elif N > 2:
            Ws = [Wl] + [Wm]*(N-2) + [Wr]
        return cls(Ws, llim=0, rlim=N-1)
    