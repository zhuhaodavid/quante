# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-04-19 14:51:03
# @Last Modified by:   dzwang
# @Last Modified time: 2025-05-24 12:51:43
import numpy as np
from copy import deepcopy
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
        >>> mps_vec /= np.linalg.norm(mps_vec)
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
        # generate MPS
        if N == 2:
            Ws = [Wl, Wr]
        elif N > 2:
            Ws = [deepcopy(Wl)] + [deepcopy(Wm) for _ in range(N-2)] + [deepcopy(Wr)]
        return cls(Ws, llim=0, rlim=N-1)
    
    @classmethod
    def generate_FullUp_state(cls, N:int, dtype=np.float64) -> "MPS":
        """ Generate MPS representing Full Up state.

        Example
        -------
        >>> N = 4
        >>> mps = qt.tensor.networks.MPS.generate_FullUp_state(N, dtype=dtype)
        >>> mps_vec = mps.to_vector()
        >>> dir_vec = qt.generate.state.product_state(["up"]*N).astype(mps.dtype).squeeze() # one down spin
        >>> print(np.allclose(mps_vec, dir_vec))
        """
        # initial zero tensor
        W = np.zeros((1, 2, 1), dtype=dtype)
        # single tensor 
        W[0, 0, 0] = 1.
        # generate MPS
        Ws = [deepcopy(W) for _ in range(N)]
        return cls(Ws, llim=0, rlim=N-1)
        
    @classmethod
    def generate_SingleUp_state(cls, N:int, dtype=np.float64) -> "MPS":
        """ Generate MPS representing Single Up state.

        Example
        -------
        >>> N = 4
        >>> mps = qt.tensor.networks.MPS.generate_SingleUp_state(N, dtype=dtype)
        >>> mps_vec = mps.to_vector()
        >>> mps_vec /= np.linalg.norm(mps_vec)
        >>> dir_vec = qt.generate.state.product_state(["up", "dn", "dn", "dn"]).astype(mps.dtype).squeeze() # one down spin
        >>> print(np.allclose(mps_vec, dir_vec))
        """
        Wup = np.zeros((1, 2, 1), dtype=dtype)
        Wup[0, 0, 0] = 1.
        Wdn = np.zeros((1, 2, 1), dtype=dtype)
        Wdn[0, 1, 0] = 1.
        # generate MPS
        Ws = [deepcopy(Wup)] + [deepcopy(Wdn) for _ in range(N-1)]
        return cls(Ws, llim=0, rlim=N-1)
    
    @classmethod
    def generate_Neel_state(cls, N:int, dtype=np.float64, *, first_site="up") -> "MPS":
        """ Generate MPS representing Single Up state.

        Example
        -------
        >>> N = 5
        >>> mps = qt.tensor.networks.MPS.generate_Neel_state(N, dtype=dtype, first_site="up")
        >>> mps_vec = mps.to_vector()
        >>> mps_vec /= np.linalg.norm(mps_vec)
        >>> dir_vec = qt.generate.state.product_state(["up", "dn", "up", "dn", "up"]).astype(mps.dtype).squeeze() # >>> one down spin
        >>> print(np.allclose(mps_vec, dir_vec))
        """
        Wup = np.zeros((1, 2, 1), dtype=dtype)
        Wup[0, 0, 0] = 1.
        Wdn = np.zeros((1, 2, 1), dtype=dtype)
        Wdn[0, 1, 0] = 1.
        # generate MPS
        if first_site == "up":
            Ws = [deepcopy(Wup), deepcopy(Wdn)] * (N//2)
            Ws += [deepcopy(Wup)] if N % 2 == 1 else []
        elif first_site == "dn":
            Ws = [deepcopy(Wdn), deepcopy(Wup)] * (N//2)
            Ws += [deepcopy(Wdn)] if N % 2 == 1 else []
        return cls(Ws, llim=0, rlim=N-1)