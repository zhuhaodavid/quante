# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2024-03-04 11:37:55
# @Last Modified by:   dzwang
# @Last Modified time: 2024-10-16 15:49:18
import numpy as _np 
import numpy.linalg as _nla
from ..linalg.svd_robust import svd_truncate


__all__ = ["MPS", "canonical_form_mps"]


class MPS():
    def __init__(self, Bs:list[_np.ndarray], Ss:list[_np.ndarray]) ->None:
        self.Ss = Ss
        self.Bs = Bs
        self.L = len(Bs)

    def _add_each(self, self_B:_np.ndarray, other_B:_np.ndarray, d:int) -> _np.ndarray:
        self_D0, self_D1 = self_B.shape[0], self_B.shape[-1]
        other_D0, other_D1 = other_B.shape[0], other_B.shape[-1]
        W = _np.zeros((self_D0+other_D0, d, self_D1+other_D1), dtype=_np.complex128)
        W[:self_D0, :, :self_D1], W[self_D0:, :, self_D1:] = self_B, other_B
        return W
    
    def _add_left(self, self_Bl, other_Bl, d, alpha, beta) -> _np.ndarray:
        self_D1, other_D1 = self_Bl.shape[-1], other_Bl.shape[-1]
        W = _np.zeros((1, d, self_D1+other_D1), dtype=_np.complex128)
        W[:,:,:self_D1], W[:,:,self_D1:] = alpha*self_Bl, beta*other_Bl
        return W
    
    def _add_right(self, self_Br, other_Br, d) -> _np.ndarray:
        self_D1, other_D1 = self_Br.shape[0], other_Br.shape[0]
        W = _np.zeros((self_D1+other_D1, d, 1), dtype=_np.complex128)
        W[:self_D1,:,:], W[self_D1:,:,:] = self_Br, other_Br
        return W

    def add(self, other_Bs:list[_np.ndarray], alpha:float, beta:float) -> list[_np.ndarray]:
        """|psi⟩ = alpha⋅|self> + beta⋅|other>

        Args:
            other_Bs : 3-order tensors of other MPS.
            alpha : Amplitude of self_MPS state.
            beta : Amplitude of other_MPS state.

        Returns:
            Ws : 3-order tensors of result MPS.
        """
        Ws = [None] * self.L
        _, d, _ = self.Bs[0].shape  # physical dimenstion: d
        # * Bs add
        for i, (self_B, other_B) in enumerate(zip(self.Bs, other_Bs)):
            Ws[i] = self._add_each(self_B, other_B, d)
        # * replace the most left B
        Ws[0] = self._add_left(self.Bs[0], other_Bs[0], d, alpha, beta)
        # * replace the most right B
        Ws[-1] = self._add_right(self.Bs[-1], other_Bs[-1], d)
        return Ws


def _left_to_right_QR(r, W) -> tuple[_np.ndarray, _np.ndarray]:
    """
                (c)                     d
                 |          QR          |
    (a)--r--(b)--W--(d)    ---->    D0--A--D1 D1--S--D1
    """
    D0, d, D2 = r.shape[0], W.shape[1], W.shape[-1]
    # rW = _np.einsum("ab,bcd->acd", r, W, optimize=True).reshape(-1, D2)
    rW = (r @ W.reshape(r.shape[1],-1)).reshape(-1, D2)
    q, r = _nla.qr(rW)  # todo 负号可能是从QR分解出来的
    return q.reshape(D0, d, -1), r  


def _right_to_left_SVD(A, S, Dc, eps) ->tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
                    d                  (b)     
                    |       SVD         |       
    D0--A--D1 S D1--B--D2  <----   (a)--Al--(c)--Ar--(d)
    """
    D0, d, D2 = A.shape[0], A.shape[1], A.shape[-1]
    # AS = _np.einsum("abc,cd->abd", A, _np.diag(S), optimize=True)
    AS = A * S
    AS = AS.reshape(D0, -1)
    A, S, B = svd_truncate(AS, Dc, eps)
    # !! psi = [oddA1, ..., S, ..., newB-1] != [newA1, ..., S, ..., newB-1]
    return A.reshape(D0, -1), S, B.reshape(-1, d, D2)


def canonical_form_mps(Ws:list[_np.ndarray], Dc:int=_np.inf, eps:float=1.e-15) ->MPS:
    """
    Args: Ws
    Returns: Ss, Bs
    """
    L = len(Ws)
    # * left ->QR-> right
    As, Ss, Bs = [None]*L, [_np.array([1.])]*(L+1), [None]*L
    r = _np.array([[1.]])
    for i, W in enumerate(Ws):
        As[i], r = _left_to_right_QR(r, W)
    As[-1] = _np.einsum("abc,cd->abd", As[-1], r, optimize=True)
    # * left <-SVD<- right
    for i in range(1, L):
        A_, Ss[-i-1], Bs[-i] = _right_to_left_SVD(As[-i], Ss[-i], Dc, eps)
        As[-i-1] = _np.einsum("abc,cd->abd", As[-i-1], A_, optimize=True)
    # * updata boundar A and B
    Bs[0] = _np.einsum("abc,cd->abd", As[0], _np.diag(Ss[1]))
    return Bs
