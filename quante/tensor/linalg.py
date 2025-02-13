# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-01-27 02:52:23
# @Last Modified by:   dzwang
# @Last Modified time: 2025-02-09 23:59:41
import numpy as np
from ..linalg.svd_robust import TruncationError, svd_truncate


__all__ = ["apply_2b_gate_mps", "update_two_site"]


def apply_2b_gate_mps(W1:np.ndarray, W2:np.ndarray, gate_2b:np.ndarray) -> np.ndarray:
    """
    .. code-block:: text
        
        .      |         |
              (c)       (f)
               |         |
               ├-gate2_b-┤
               |         |                       |    |        
              (b)       (e)                     (b)  (e)       
               |         |                       |    |         
        --(a)--◻---(d)---◻--(g)--  ----> --(a)---◻----◻---(g)-- 
               W1        W2                      theta                 
    
    >>> L = 2
    >>> psi0 = qt.generate.state.w(L, dtype=dtype)
    >>> H = qt.generate.matrix.pauli_matrix("xx+yy+zz")
    >>> U = qt.linalg.expm(H, c=-1.j)
    >>> psi1 = U @ psi0
    >>> d = 2
    >>> u, s, vt = qt.linalg.svd(psi0.reshape(d, d))
    >>> W1 = (u * s).reshape(1, d, d)
    >>> W2 = vt.reshape(d, d, 1)
    >>> theta = qt.tensor.apply_2b_gate_mps(W1, W2, gate_2b=U).reshape(d*d, 1)
    >>> print(np.allclose(psi1, theta))
    """
    assert gate_2b.ndim == 2, "gate_2b must be a matrix."
    a, b, d = W1.shape
    d, e, g = W2.shape
    W = W1.reshape(-1, d) @ W2.reshape(d, -1)
    W = W.reshape(a,b,e,g).transpose(1,2,0,3).reshape(b*e, -1)
    W = gate_2b @ W
    theta = W.reshape(b,e,a,g).transpose(2,0,1,3).reshape(a,b,e,g)
    return theta


def update_two_site(W1W2:np.ndarray, direction:str, lr_index:list=None, trunc_para:tuple=(None, None, None)) -> tuple[np.ndarray, np.ndarray, TruncationError]:
    """
    .. code-block:: text
    
        .      |    |              
              (b)  (e)    
               |    |                 │   │              │   │            |      | 
        --(a)--◻----◻--(g)--   —>  ---▷---◻---   or   ---◻---⨞--- or   ---◻--🔸--◻--- 
               :    :                 :   :              :   :            :      :
              (c)  (f)  
                W1W2                  right              left              mixed
    
    >>> d, chi = 2, 10
    >>> W1W2 = np.random.randn(chi, d, d, chi)
    >>> W1W2 /= np.linalg.norm(W1W2)
    >>> W1, W2, trunc_error = qt.tensor.update_two_site(W1W2, direction="right", trunc_para=(20, None, None))
    >>> W1W2_ = np.einsum("abc,cde->abde", W1, W2)
    >>> print(np.allclose(W1W2, W1W2_))
    >>> print(trunc_error.eps, trunc_error.ov)
    >>> print("-"*20)
    >>> W1W2 = np.random.randn(chi, d, d, d, chi)
    >>> W1W2 /= np.linalg.norm(W1W2)
    >>> W1, W2, trunc_error = qt.tensor.update_two_site(W1W2, direction="left", lr_index=[[0,1],[2,3,4]], trunc_para=(20, None, None))
    >>> W1W2_ = np.einsum("abc,cdef->abdef", W1, W2)
    >>> print(np.allclose(W1W2, W1W2_))
    >>> print(trunc_error.eps, trunc_error.ov)
    >>> print("-"*20)
    >>> W1W2 = np.random.randn(chi, d, d, d, chi)
    >>> W1W2 /= np.linalg.norm(W1W2)
    >>> W1, s, W2, trunc_error = qt.tensor.update_two_site(W1W2, direction="mixed", lr_index=[[0,1],[2,3,4]], trunc_para=(20, None, None))
    >>> W1W2_ = np.einsum("abc,c,cdef->abdef", W1, s, W2)
    >>> print(np.allclose(W1W2, W1W2_))
    >>> print(trunc_error.eps, trunc_error.ov)   
    """
    u, s, vt, trunc_error = svd_truncate(W1W2, lr_index, trunc_para=trunc_para)

    if direction=="right":
        return u, s.reshape(-1, *([1]*(vt.ndim-1)))*vt, trunc_error
    elif direction=="left":
        return u*s, vt, trunc_error
    elif direction=="mixed":
        return u, s, vt, trunc_error
    else:
        raise ValueError("direction must be 'right', 'left' or 'mixed'.")    

