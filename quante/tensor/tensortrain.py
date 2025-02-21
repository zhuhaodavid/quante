# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-02-19 14:51:17
# @Last Modified by:   dzwang
# @Last Modified time: 2025-02-21 16:13:00
import numpy as np
from quante.basicfun import println
from .linalg import left2right_QR_step, right2left_QR_step, tensor2matrix



class TensorTrain:
    """
    A class for tensor train representation of a tensor network in 1D.
    
    Notes
    -----    
    Only support three or four order tensor.
    
    The order must be: Left - Up - Down - Right.
    
    .. code-block:: text
        .
            |       |       |       |       |       |       |
        ----▷-------⬜-------⬜-------⬜-------⬜-------⬜-------⨞-----
            :       :       :       :       :       :       :
           Ws[0]   Ws[1]   Ws[2]   Ws[3]   Ws[4]   Ws[5]   Ws[6]
                    ↑                               ↑
                llim = 1                         rlim = 5 
    """
    def __init__(self, Ws:list[np.ndarray], llim:int, rlim:int, bc="finite") -> None:
        self.Ws = Ws
        self.llim = llim
        self.rlim = rlim

    @property
    def L(self) -> int:
        """Number of physical sites."""
        return len(self.Ws)

    @property
    def chi(self) -> list[int]:
        """List of bond dimensions."""
        return [self.Ws[i].shape[-1] for i in range(self.L-1)]
    
    def to_matrix(self) -> np.array:
        """
        .. code-block:: text
            .
                    |         |                         |
                   (b)       (e)                       (be)
                    |         |                         |
            --(a)--res--(d)---W---(g)--  --->   --(a)--res--(g)--
                    :         :                         :
                   (c)       (f)                       (cf)
                    |         |                         |
        Examples
        --------
        >>> L = 2
        >>> Ws = []
        >>> for l in range(L):
        >>>     Ws.append(np.random.rand(2, 3, 2).astype(dtype))
        >>>     Ws.append(np.random.rand(2, 4, 5, 2).astype(dtype))
        >>> tt = qt.tensor.TensorTrain(Ws, llim=0, rlim=2*L-1)
        >>> matrix = tt.to_matrix()
        >>> matrix_ = np.einsum("abc,cdef,fgh,hijk->abdgiejk", Ws[0], Ws[1], Ws[2], Ws[3])
        >>> println(np.allclose(matrix, matrix_))
        """
        result = self.Ws[0]
        for i in range(1, self.L):
            # reshape tensor to matrx
            a, *bc, d = result.shape
            d, *ef, g = self.Ws[i].shape
            if len(bc)==1: bc.append(1)
            if len(ef)==1: ef.append(1)
            result_mat, W_mat = result.reshape(-1, d), self.Ws[i].reshape(d, -1)
            # matrix multiplication and reshape back to tensor
            result = (result_mat @ W_mat).reshape(a, bc[0], bc[1], ef[0], ef[1], g)
            result = result.transpose(0, 1, 3, 2, 4, 5).reshape(a, bc[0]*ef[0], bc[1]*ef[1], g)
        return result

    def set_mixed_canonical_form(self, oc:int) -> None:
        """
        .. code-block:: text
            .
                |       |       |       |       |      |       |
            ----▷-------▷-------▷-------⬜-------⨞-------⨞-------⨞-----
                :       :       :       :       :       :      :
               Ws[0]   Ws[1]   Ws[2]   Ws[3]   Ws[4]   Ws[5]   Ws[6]
                                        ↑
                                llim = llim = 3
        """
        assert self.llim <= self.rlim, '"llim" should be at the left side of "rlim"'
        self._move_llim(oc)
        self._move_rlim(oc)
    
    def _move_llim(self, oc:int) -> None:
        """Make all tensors ``Ws[j]`` to the left of ``i`` left-orthogonal"""
        while self.llim < oc:
            a, b = self.llim, self.llim+1
            self.Ws[a], self.Ws[b] = left2right_QR_step(self.Ws[a], self.Ws[b])
            self.llim = self.llim + 1
            if self.rlim < self.llim: self.rlim = self.llim
    
    def _move_rlim(self, oc:int) -> None:
        """Make all tensors ``Ws[j]`` to the right of ``i`` right-orthogonal"""
        while self.rlim > oc:
            a, b = self.rlim-1, self.rlim
            self.Ws[a], self.Ws[b] = right2left_QR_step(self.Ws[a], self.Ws[b])
            self.rlim = self.rlim - 1
            if self.llim > self.rlim: self.llim = self.rlim

    def check_mixed_canonical_form(self) -> None:
        dtype = self.Ws[0].dtype
        for i in range(self.llim):
            dim = self.Ws[i].shape[-1]
            tmp = self.Ws[i].reshape(-1, dim)
            assert np.allclose(tmp.T.conj()@tmp, np.eye(dim, dtype=dtype)), f"{i}-th tensor is not orthogonal."
        for i in range(self.rlim+1, self.L):
            dim = self.Ws[i].shape[0]
            tmp = self.Ws[i].reshape(dim, -1)
            assert np.allclose(tmp@tmp.conj().T, np.eye(dim, dtype=dtype)), f"{i}-th tensor is not orthogonal."