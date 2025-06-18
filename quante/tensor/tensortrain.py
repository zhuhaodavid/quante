# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-02-19 14:51:17
# @Last Modified by:   dzwang
# @Last Modified time: 2025-05-30 16:32:38
import numpy as np
from quante.basicfun import println
from .linalg import left2right_QR_step, right2left_QR_step, inner_initialize, inner_step, add_left, add_mid, add_right


__all__ = ["TensorTrain"]


class TensorTrain:
    """
    A class of Tensor Train representation for vector or matrix.
    
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
    def __init__(self, Ws:list[np.ndarray], llim:int, rlim:int) -> None:
        self.Ws = Ws
        self.llim = llim
        self.rlim = rlim
        self.dtype = Ws[0].dtype

    @property
    def N(self) -> int:
        """Number of tensors."""
        return len(self.Ws)
    
    @property
    def chi(self) -> list[int]:
        """List of bond dimensions."""
        return [self.Ws[i].shape[-1] for i in range(self.N-1)]
    
    @property
    def norm(self) -> float:
        if self.llim == self.rlim:
            W_oc = self.Ws[self.llim]
            norm_tensor = inner_initialize(W_oc, W_oc)
            norm = np.einsum("aabb->", norm_tensor)
        else:
            norm = self.inner(self.Ws, self.Ws)
        assert norm.imag < 1.e-12, "The norm is not real."
        return np.sqrt(norm.real)
    
    def to_tensor(self) -> np.ndarray:
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
        """
        result = self.Ws[0]
        for i in range(1, self.N):
            # reshape tensor to matrx
            a, *bc, d = result.shape
            d, *ef, g = self.Ws[i].shape
            if len(bc)==1: bc.append(1)
            if len(ef)==1: ef.append(1)
            result_mat, W_mat = result.reshape(-1, d), self.Ws[i].reshape(d, -1)
            # matrix multiplication and reshape back to tensor
            result = (result_mat @ W_mat).reshape(a, bc[0], bc[1], ef[0], ef[1], g)
            result = result.transpose(0, 1, 3, 2, 4, 5).reshape(a, bc[0]*ef[0], bc[1]*ef[1], g)
        return result.squeeze()
    
    def get_llim_rlim(self) -> tuple[int, int]:
        for i in range(self.N):
            dim = self.Ws[i].shape[-1]
            tmp = self.Ws[i].reshape(-1, dim)
            Id = np.eye(dim, dtype=self.dtype)
            if np.allclose(tmp.T.conj()@tmp, Id) is False:
                llim = i
                break
            # case of all tensors are left-orthogonal 
            if i==self.N-1: llim = 0
        
        for i in range(self.N-1, -1, -1):
            dim = self.Ws[i].shape[0]
            tmp = self.Ws[i].reshape(dim, -1)
            Id = np.eye(dim, dtype=self.dtype)
            if np.allclose(tmp@tmp.conj().T, Id) is False:
                rlim = i
                break
            # case of all tensors are right-orthogonal
            if i == 0: rlim=0
        if llim > rlim: raise NotImplementedError("发现了特殊情况！！")
        return llim, rlim

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
        self.move_llim(oc)
        self.move_rlim(oc)
    
    def move_llim(self, oc:int) -> None:
        """Make all tensors ``Ws[j]`` to the left of ``i`` left-orthogonal"""
        while self.llim < oc:
            a, b = self.llim, self.llim+1
            self.Ws[a], self.Ws[b] = left2right_QR_step(self.Ws[a], self.Ws[b])
            self.llim = self.llim + 1
            if self.rlim < self.llim: self.rlim = self.llim
    
    def move_rlim(self, oc:int) -> None:
        """Make all tensors ``Ws[j]`` to the right of ``i`` right-orthogonal"""
        while self.rlim > oc:
            a, b = self.rlim-1, self.rlim
            self.Ws[a], self.Ws[b] = right2left_QR_step(self.Ws[a], self.Ws[b])
            self.rlim = self.rlim - 1
            if self.llim > self.rlim: self.llim = self.rlim

    def check_mixed_canonical_form(self) -> None:
        dtype = self.Ws[0].dtype
        for i in range(self.llim):
            println(i)
            dim = self.Ws[i].shape[-1]
            tmp = self.Ws[i].reshape(-1, dim)
            assert np.allclose(tmp.T.conj()@tmp, np.eye(dim, dtype=dtype)), f"{i}-th tensor is not orthogonal."
        for i in range(self.rlim+1, self.N):
            println(i)
            dim = self.Ws[i].shape[0]
            tmp = self.Ws[i].reshape(dim, -1)
            assert np.allclose(tmp@tmp.conj().T, np.eye(dim, dtype=dtype)), f"{i}-th tensor is not orthogonal."
            
    def add(self, other_tt:"TensorTrain", *, c1:float=1., c2:float=1.) -> "TensorTrain":
        """ Add two MPS together with coefficients c1 and c2.
        Notes
        -----
        A = c1 * self + c2 * other
        """
        assert self.Ws[0].shape[0] == other_tt.Ws[0].shape[0] == 1, "Dimension of first tensor should be (1, ...)."
        assert self.Ws[-1].shape[-1] == other_tt.Ws[-1].shape[-1] == 1, "Dimension of last tensor should be (..., 1)."
        Ws_other = other_tt.Ws
        assert self.N == other_tt.N, "The number of tensors in two TensorTrain must be the same."
        Ws_add = [None] * self.N
        ## add the leftmost tensor
        Ws_add[0] = add_left(c1*self.Ws[0], c2*Ws_other[0])
        ## add the middle tensors
        for n in range(1, self.N-1):
            Ws_add[n] = add_mid(self.Ws[n], Ws_other[n])
        ## add the rightmost tensor
        Ws_add[-1] = add_right(self.Ws[-1], Ws_other[-1])
        return TensorTrain(Ws_add, llim=0, rlim=self.N-1)
    
    @classmethod
    def tt_decompose(cls, tensor:np.ndarray, d:int) -> "TensorTrain":
        """ Tensor Train decomposition of a vector or matrix using QR decomposition.
        """
        ## metch the physical dimension 
        ndim = tensor.ndim
        shape = tensor.shape[0]
        assert ndim == 1, "Only support 1- or 2-dimensional tensor Currently."
        # number of tt site
        Nt:float = np.log(shape) / np.log(d)
        Nt=int(Nt) if Nt.is_integer() else ValueError("Physical dimension is not compatible with the Tensor shape.")
        ## QR decomposition
        Ws = [None] * Nt
        for i in range(1, Nt):
            q, r = np.linalg.qr(tensor.reshape(-1, d**(Nt-i)))
            Ws[i-1] = q.reshape(-1, d, r.shape[0])
            tensor = r
        Ws[-1] = tensor.reshape(-1, d, 1)
        return cls(Ws, llim=Nt-1, rlim=Nt-1)
    
    
    @staticmethod
    def inner(Ws1:list[np.ndarray], Ws2:list[np.ndarray]) -> float:
        assert len(Ws1) == len(Ws2), "N in two Ws must be the same."
        assert Ws1[0].shape[0] == Ws1[-1].shape[-1] == Ws2[0].shape[0] == Ws2[-1].shape[-1] == 1, "The first and last tensors in two TensorTrain must be equal to unit."
        ## main
        Lenv = inner_initialize(Ws1[0], Ws2[0])
        for n in range(1, len(Ws1)):
            Lenv = inner_step(Lenv, Ws1[n], Ws2[n])
        return Lenv.squeeze()
