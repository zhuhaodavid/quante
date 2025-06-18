# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-01-27 02:52:23
# @Last Modified by:   dzwang
# @Last Modified time: 2025-06-18 13:00:15
import numpy as np
from ..linalg.decomp import TruncationError, svd_truncate
from quante.basicfun import println

__all__ = ["generate_delta_tensor"]
__all__ += ["invert_transpose_order", "tensor2matrix", "qr", "rq"]
__all__ += ["left2right_QR_step", "right2left_QR_step", "apply_2b_gate_mps", "update_two_site"]
__all__ += ["inner_initialize", "inner_step"]
__all__ += ["add_mid", "add_left", "add_right"]


def generate_delta_tensor(shape:tuple) -> np.ndarray:  # todo 弄懂
    """
    生成一个 n 阶 delta 张量，只有所有索引都相等时为 1
    shape: (d, d, ..., d) 长度为阶数
    """
    grids = np.indices(shape)  # shape: (n, d, d, ..., d)
    # 比较是否所有坐标相等
    equal_mask = np.all(grids == grids[0], axis=0)
    return equal_mask.astype(int)


def invert_transpose_order(order) -> tuple:
    """Invert the order of a transpose operation.
    """
    order = tuple(order)  # Convert to tuple if list
    inverse = [-1] * len(order)
    for i, ax in enumerate(order):
        inverse[ax] = i
    return tuple(inverse)


def tensor2matrix(tensor:np.ndarray, lr_index:list[list, list]=None) -> tuple[np.ndarray, list, list]:
    """Reshape a tensor to a matrix along the given left/right bond index.

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
    """
    # QR decomposition W1 and revert its order
    if lr_index1 == None:
        lr_index1 = [list(range(W1.ndim-1)), [W1.ndim-1]]
    W1_update, S = qr(W1, lr_index=lr_index1)
    index1_order = sum(lr_index1, [])
    # revert the order of W1
    W1_update = W1_update.transpose(invert_transpose_order(index1_order))
    # shift OC into W2
    if lr_index2 == None:
        lr_index2 = [[0], list(range(1, W2.ndim))]
    W2_mat, l_shape, r_shape = tensor2matrix(W2, lr_index2)
    W2_update = S @ W2_mat
    l_shape[0] = W2_update.shape[0]  # for the case of left-dim > right-dim about W1
    assert len(lr_index1[-1]) == len(lr_index2[0]) == 1, "The last index of lr_index1 and the first index of lr_index2 must be the one index."
    # reshape W2 back to tensor
    W2_update = W2_update.reshape(*l_shape, *r_shape)
    # revert the order of W2
    index2_order = sum(lr_index2, [])
    W2_update = W2_update.transpose(invert_transpose_order(index2_order))
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
    """
    # RQ decomposition W2 and revert its order
    if lr_index2 == None:
        lr_index2 = [[0], list(range(1, W2.ndim))]
    S, W2_update = rq(W2, lr_index=lr_index2)
    index2_order = sum(lr_index2, [])
    W2_update = W2_update.transpose(invert_transpose_order(index2_order))
    # shift OC into W1
    if lr_index1 == None:
        lr_index1 = [list(range(W1.ndim-1)), [W1.ndim-1]]
    W1_mat, l_shape, r_shape = tensor2matrix(W1, lr_index1)
    W1_update = W1_mat @ S
    r_shape[-1] = W1_update.shape[-1]  # for the case of right-dim > left-dim about W2
    assert len(lr_index1[-1]) == len(lr_index2[0]) == 1, "The last index of lr_index1 and the first index of lr_index2 must be the one index."
    # reshape W1 back to tensor
    W1_update = W1_update.reshape(*l_shape, *r_shape)
    # d. revert the order of W1
    index1_order = sum(lr_index1, [])
    W1_update = W1_update.transpose(invert_transpose_order(index1_order))
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
    """
    assert gate_2b.ndim == 2, "gate_2b must be a matrix."
    a, b, d = W1.shape
    d, e, g = W2.shape
    W = W1.reshape(-1, d) @ W2.reshape(d, -1)
    W = W.reshape(a,b,e,g).transpose(1,2,0,3).reshape(b*e, -1)
    W = gate_2b @ W
    theta = W.reshape(b,e,a,g).transpose(2,0,1,3).reshape(a,b,e,g)
    return theta


def update_two_site(theta:np.ndarray, direction:str, lr_index:list=None, trunc_para:tuple=(None, None, None)) -> tuple[np.ndarray, np.ndarray, TruncationError]:
    """
    .. code-block:: text
    
        .      |    |              
              (b)  (e)    
               |    |                 │   │              │   │            |      | 
        --(a)--◻----◻--(g)--   —>  ---▷---◻---   or   ---◻---⨞--- or   ---◻--🔸--◻--- 
               :    :                 :   :              :   :            :      :
              (c)  (f)  
                theta                 right              left              mixed
    """
    u, s, vt, trunc_error = svd_truncate(theta, lr_index, trunc_para=trunc_para)
    if direction=="right":
        return u, s.reshape(-1, *([1]*(vt.ndim-1)))*vt, trunc_error
    elif direction=="left":
        return u*s, vt, trunc_error
    elif direction=="mixed":
        return u, s, vt, trunc_error
    else:
        raise ValueError("direction must be 'right', 'left' or 'mixed'.")    


def inner_initialize(W1:np.ndarray, W2:np.ndarray) -> np.ndarray:
    """
    .. code-block:: text
        .
                    :                        
             --(a)--W1--(c)--         --(a)--┬--(c)--
                    |                        |
                   (b)          --->        Lenv
                    |                        |
             --(d)--W2--(e)--         --(d)--┴--(e)--
                    :                        
    Note
    ----
    Order of the indices must be left - up - down - right.
    """
    a, *_, c = W1.shape
    W1_mat = W1.reshape(a, -1, c)
    W1_mat = W1_mat.swapaxes(1,2).reshape(a*c, -1)    
    d, *_, e = W2.shape
    W2_mat = W2.reshape(d, -1, e)
    W2_mat = W2_mat.swapaxes(0,1).reshape(-1, d*e)
    return (W1_mat.conj() @ W2_mat).reshape(a, c, d, e).swapaxes(1,2)


def inner_step(Lenv:np.ndarray, W1:np.ndarray, W2:np.ndarray) -> np.ndarray:
    """
    .. code-block:: text
       . 
                           :   
            --(a)--┬--(c)--W1--(f)--         --(a)--┬--(c)--
                   |       |                        |
                  Lenv    (b)          --->        Lenv
                   |       |                        |
            --(d)--┴--(e)--W2--(g)--         --(d)--┴--(e)--
                           :
    """
    e, *b, g = W2.shape
    c, *b, f = W1.shape
    a, d, c, e = Lenv.shape
    Lenv = Lenv.reshape(-1, e) @ W2.reshape(e, -1)
    Lenv = Lenv.reshape(a*d, -1, g).swapaxes(1,2).reshape(a*d*g, -1)
    Lenv = Lenv @ W1.conj().reshape(-1, f)
    return Lenv.reshape(a, d, g, f).swapaxes(2,3)



def add_mid(W1:np.ndarray, W2:np.ndarray) -> np.ndarray:
    """
    .. code-block:: text
    
        .      |                  |                         |
              (b)                (e)                       (b)
               |                  |                         |
        --(a)--W1--(c)-- + --(d)--W2--(f)--  --->  --(a+d)--W--(c+f)--
               :                  :                         :
    """
    assert W1.dtype == W2.dtype, "W1 and W2 must have the same dtype."
    a, *b, c = W1.shape
    d, *e, f = W2.shape
    assert b == e, "The shape of W1 and W2 must be compatible for physical dimension."
    W1p2 = np.zeros((a+d, *b, c+f), dtype=W1.dtype)
    W1p2[:a, ..., :c] = W1
    W1p2[a:, ..., c:] = W2
    return W1p2


def add_left(W1:np.ndarray, W2:np.ndarray) -> np.ndarray:
    """
    .. code-block:: text
    
        .      |                  |                       |
              (b)                (e)                     (b)
               |                  |                       |
        --(1)--W1--(c)-- + --(1)--W2--(f)--  --->  --(1)--W--(c+f)--
               :                  :                       :
    """
    assert W1.dtype == W2.dtype, "W1 and W2 must have the same dtype."
    a, *b, c = W1.shape
    d, *e, f = W2.shape
    assert a == d == 1, "The first dimension of W1 and W2 must be 1 for left dimension."
    assert b == e, "The shape of W1 and W2 must be compatible for physical dimension."
    W1p2 = np.zeros((1, *b, c+f), dtype=W1.dtype)
    W1p2[..., :c] = W1
    W1p2[..., c:] = W2
    return W1p2


def add_right(W1:np.ndarray, W2:np.ndarray) -> np.ndarray:
    """
    .. code-block:: text
    
        .      |                  |                         |
              (b)                (e)                       (b)
               |                  |                         |
        --(a)--W1--(1)-- + --(d)--W2--(1)--  --->  --(a+d)--W--(1)--
               :                  :                         :
    """
    assert W1.dtype == W2.dtype, "W1 and W2 must have the same dtype."
    a, *b, c = W1.shape
    d, *e, f = W2.shape
    assert c == f == 1, "The last dimension of W1 and W2 must be 1 for left dimension."
    assert b == e, "The shape of W1 and W2 must be compatible for physical dimension."
    W1p2 = np.zeros((a+d, *b, 1), dtype=W1.dtype)
    W1p2[:a, ...] = W1
    W1p2[a:, ...] = W2
    return W1p2


