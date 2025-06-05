# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-04-19 14:51:03
# @Last Modified by:   dzwang
# @Last Modified time: 2025-05-30 21:43:54
import numpy as np
from copy import deepcopy
from quante.tensor import TensorTrain 
from quante.basicfun import println


__all__ = ['MPS']


class MPS(TensorTrain):
    def __init__(self, Ws:list[np.ndarray], llim:int, rlim:int) -> None:
        super().__init__(Ws, llim, rlim)
    
    def to_vector(self) -> np.ndarray:
        """Convert the MPS to a vector.
        """
        return self.to_tensor()
    
    @classmethod
    def generate_from_vector(cls, vector:np.ndarray, d:int) -> "MPS":
        """ Generate MPS from a vector using QR decomposition.
        """
        tt = TensorTrain.tt_decompose(vector, d)
        return cls(tt.Ws, llim=tt.llim, rlim=tt.rlim)
    
    
    @classmethod
    def generate_product_state(cls, config:str|list[str], dtype=np.float64) -> "MPS":
        """Generating MPS representing for product states
        """
        up = np.zeros((1, 2, 1), dtype=dtype)
        up[0, 0, 0] = 1.
        dn = np.zeros((1, 2, 1), dtype=dtype)
        dn[0, 1, 0] = 1.
        Ws = [None] * len(config)
        for i, site in enumerate(config):
            Ws[i] = dn if site == "0" else up
        return cls(Ws, llim=0, rlim=0)


    @classmethod
    def generate_W_state(cls, N:int, dtype=np.float64, *, type:str="single_up") -> "MPS":
        """ Generate MPS representing for W state.

        Parameters
        type : "single_up" or "single_dn"
            up: one up spin
            dn: one down spin
        """
        assert N >= 2, "N must be greater than or equal to 2"
        # judge the type of W state
        if type == "single_up":
            index = 1
        elif type == "single_dn":
            index = 0
        else:
            raise ValueError("type must be 'single_up' or 'single_dn'")
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
            Ws = [Wl] + [deepcopy(Wm) for _ in range(N-2)] + [Wr]        
        return cls(Ws, llim=0, rlim=N-1)
