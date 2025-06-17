# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-01-27 02:52:23
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:44:12
import numpy as np
from ..linalg.decomp import TruncationError, svd_truncate
from quante.basicfun import println


__all__ = ["invert_transpose", "tensor2matrix", "qr", "rq"]
__all__ += ["left2right_QR_step", "right2left_QR_step", "apply_2b_gate_mps", "update_two_site"]


def invert_transpose(axes) -> tuple:
    """Calculate the inverse permutation of given transpose axes.
        
    Example
    ----------
    >>> tensor = np.random.rand(2, 3, 4, 5)
    >>> perm = (2, 0, 1, 3)
    >>> transposed = tensor.transpose(perm)
    >>> original = transposed.transpose(invert_transpose(perm))
    >>> print(np.array_equal(tensor, original))
    """
    axes = tuple(axes)  # Convert to tuple if list
    inverse = [-1] * len(axes)
    for i, ax in enumerate(axes):
        inverse[ax] = i
    return tuple(inverse)


def tensor2matrix(tensor:np.ndarray, lr_index:list=None) -> tuple[np.ndarray, list, list]:
    """Reshape a tensor to a matrix along the given left/right bond index.

    Examples
    ----------
    >>> tensor = np.random.randn(2, 3, 4, 5)
    >>> matrix, l_shape, r_shape = qt.tensor.tensor2matrix(tensor)
    >>> print(matrix.shape, l_shape, r_shape)
    >>> matrix, l_shape, r_shape = qt.tensor.tensor2matrix(tensor, lr_index=[[0, 2], [1, 3]])
    >>> print(matrix.shape, l_shape, r_shape)
    """
    ndim = tensor.ndim
    shape = tensor.shape
    if lr_index == None:
        l_shape, r_shape = [*shape[:ndim//2]], [*shape[ndim//2:]]
        matrix = tensor.reshape(np.prod(l_shape), np.prod(r_shape))
    else:
        l_index, r_index = lr_index
        l_shape = [shape[i] for i in l_index]
        r_shape = [shape[i] for i in r_index]
        matrix = tensor.transpose(*(l_index+r_index)).reshape(np.prod(l_shape), np.prod(r_shape))
    return matrix, l_shape, r_shape
    
    
def qr(tensor:np.ndarray, lr_index:list=None) -> tuple[np.ndarray, np.ndarray]:
    """
    .. code-block:: text
        QR decomposition of a tensor.
               |                           |
              (b)                         (b)
               |            QR             |
        --(a)--⬜--(c)--    ---->    --(a)--▷--(c)--⬜--(d)--
               W                           A       S

    Examples
    ----------
    >>> tensor = np.random.randn(2, 3, 4, 5)
    >>> A, S = qt.tensor.qr(tensor)
    >>> print(A.shape, S.shape)
    >>> print(np.allclose(np.einsum("abc,cde->abde", A, S), tensor))
    >>> A, S = qt.tensor.qr(tensor, lr_index=[[1, 3], [0, 2]])
    >>> print(A.shape, S.shape)
    >>> print(np.allclose(np.einsum("abc,cde->abde", A, S), tensor.transpose(1, 3, 0, 2)))
    """
    # a. reshape tensor to a matrix
    matrix, l_shape, r_shape = tensor2matrix(tensor, lr_index)
    # b. QR decomposition
    A, S = np.linalg.qr(matrix)
    # c. reshape A back to tensor
    A = A.reshape(*l_shape, -1)
    S = S.reshape(-1, *r_shape)
    
    return A, S


def rq(tensor:np.ndarray, lr_index:list=None) -> tuple[np.ndarray, np.ndarray]:
    """
    .. code-block:: text
        RQ decomposition of a tensor.
                       |                       |         
                      (b)                     (b)        
                       |           QR          |         
        --(a)--⬜--(d)--⨞--(c)--   <---  --(a)--⬜--(c)--    
               L       U                     tensor
               
    >>> tensor = np.random.randn(2, 3, 4, 5)
    >>> L, U = qt.tensor.rq(tensor)
    >>> print(L.shape, U.shape)
    >>> print(np.allclose(np.einsum("abc,cde->abde", L, U), tensor))
    >>> L, U = qt.tensor.rq(tensor, lr_index=[[1, 3], [0, 2]])
    >>> print(L.shape, U.shape)
    >>> print(np.allclose(np.einsum("abc,cde->abde", L, U), tensor.transpose(1, 3, 0, 2)))
    """
    # a. reshape tensor to a matrix
    matrix, l_shape, r_shape = tensor2matrix(tensor, lr_index)
    # b. QR decomposition the Transpose of matrix
    A, S = np.linalg.qr(matrix.T)
    # c. transpose back
    L, U = S.T, A.T
    # c. reshape A back to tensor
    L = L.reshape(*l_shape, -1)
    U = U.reshape(-1, *r_shape)
    return L, U


def left2right_QR_step(W1:np.ndarray, W2:np.ndarray, lr_index1:list=None, lr_index2:list=None) -> tuple[np.ndarray, np.ndarray]:
    """
    .. code-block:: text
        .
               |       |                         |       |
              (b)     (d)                       (b)     (d)
               |       |           QR            |       |
        --(a)--◻--(c)--◻--(e)--   ---->   --(a)--▷--(f)--◻--(e)-- 
               :       :                         :       :
               W1      W2                       W1p     W2p

    Examples
    ----------
    >>> W1 = np.random.rand(2, 3, 4, 5) + 1j*np.random.rand(2, 3, 4, 5)
    >>> W2 = np.random.rand(2, 3, 4, 5) + 1j*np.random.rand(2, 3, 4, 5)
    >>> lr_index1 = [[3, 0, 2], [1]]
    >>> lr_index2 = [[1], [0, 2, 3]]
    >>> W1p, W2p = qt.tensor.left2right_QR_step(W1, W2, lr_index1, lr_index2)
    >>> print(W1p.shape, W2p.shape)
    >>> W1W2 = np.einsum("abcd,ebfg->aecfdg", W1, W2)
    >>> W1pW2p = np.einsum("abcd,ebfg->aecfdg", W1p, W2p)
    >>> print(np.allclose(W1W2, W1pW2p))
    >>> W1p_ = W1p.transpose(3, 0, 2, 1)
    >>> print(np.einsum("abcd,abce->de", W1p_, W1p_.conj()))
    """
    # a. QR decomposition W1 and revert its order
    if lr_index1 == None:
        lr_index1 = [list(range(W1.ndim-1)), [W1.ndim-1]]
    W1_update, S = qr(W1, lr_index=lr_index1)
    index1_order = sum(lr_index1, [])
    W1_update = W1_update.transpose(invert_transpose(index1_order))
    # b. shift OC into W2
    if lr_index2 == None:
        lr_index2 = [[0], list(range(1, W2.ndim))]
    W2_mat, l_shape, r_shape = tensor2matrix(W2, lr_index2)
    W2_update = S @ W2_mat
    assert len(lr_index1[-1]) == len(lr_index2[0]) == 1, "The last index of lr_index1 and the first index of lr_index2 must be the one index."
    # c. reshape W2 back to tensor
    W2_update = W2_update.reshape(*l_shape, *r_shape)
    # d. revert the order of W2
    index2_order = sum(lr_index2, [])
    W2_update = W2_update.transpose(invert_transpose(index2_order))
    return W1_update, W2_update


def right2left_QR_step(W1:np.ndarray, W2:np.ndarray, lr_index1:list=None, lr_index2:list=None) -> tuple[np.ndarray, np.ndarray]:
    """
    .. code-block:: text
        .
               |       |                        |       |        
              (b)     (d)                      (b)     (d)       
               |       |           QR           |       |        
        --(a)--◻--(f)--⨞--(e)--   <----  --(a)--◻--(c)--◻--(e)-- 
               :       :                        :       :
              W1p     W2p                       W1      W2
              
    Examples
    ----------
    >>> W1 = np.random.rand(2, 3, 4, 5) + 1j*np.random.rand(2, 3, 4, 5)
    >>> W2 = np.random.rand(2, 3, 4, 5) + 1j*np.random.rand(2, 3, 4, 5)
    >>> lr_index1 = [[3, 0, 2], [1]]
    >>> lr_index2 = [[1], [0, 2, 3]]
    >>> W1p, W2p = qt.tensor.right2left_QR_step(W1, W2, lr_index1, lr_index2)
    >>> print(W1p.shape, W2p.shape)
    >>> W1W2 = np.einsum("abcd,ebfg->aecfdg", W1, W2)
    >>> W1pW2p = np.einsum("abcd,ebfg->aecfdg", W1p, W2p)
    >>> print(np.allclose(W1W2, W1pW2p))
    >>> W2p_ = W2p.transpose(1, 0, 2, 3)
    >>> print(np.einsum("abcd,ebcd->ae", W2p_, W2p_.conj()))
    """
    # a. RQ decomposition W2 and revert its order
    if lr_index2 == None:
        lr_index2 = [[0], list(range(1, W2.ndim))]
    S, W2_update = rq(W2, lr_index=lr_index2)
    index2_order = sum(lr_index2, [])
    W2_update = W2_update.transpose(invert_transpose(index2_order))
    # b. shift OC into W1
    if lr_index1 == None:
        lr_index1 = [list(range(W1.ndim-1)), [W1.ndim-1]]
    W1_mat, l_shape, r_shape = tensor2matrix(W1, lr_index1)
    W1_update = W1_mat @ S
    assert len(lr_index1[-1]) == len(lr_index2[0]) == 1, "The last index of lr_index1 and the first index of lr_index2 must be the one index."
    # c. reshape W1 back to tensor
    W1_update = W1_update.reshape(*l_shape, *r_shape)
    # d. revert the order of W1
    index1_order = sum(lr_index1, [])
    W1_update = W1_update.transpose(invert_transpose(index1_order))
    return W1_update, W2_update


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
    # s /= np.linalg.norm(s)
    
    if direction=="right":
        return u, s.reshape(-1, *([1]*(vt.ndim-1)))*vt, trunc_error
    elif direction=="left":
        return u*s, vt, trunc_error
    elif direction=="mixed":
        return u, s, vt, trunc_error
    else:
        raise ValueError("direction must be 'right', 'left' or 'mixed'.")    

