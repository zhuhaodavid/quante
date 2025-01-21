# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2024-12-11 11:26:43
# @Last Modified by:   dzwang
# @Last Modified time: 2025-01-21 13:12:35
import numpy as np 
from ...generate.state import spin_down, spin_up, xplus, xminus, yplus, yminus
from ..linalg.decomposer import qr_tensor, rq_tensor, svd_tensor


# todo list: solver "qr", "eig", "get_bond_dimension" 


__all__ = ["BaseMPSExpectationValue", "MPS"]


class BaseMPSExpectationValue:
    """
    .. code-block:: text
             |      |      |      |      |     |      |
        ——-——⨞—————-⨞——-————⨞——————⨞——————⨞——————⨞——————⨞————
            Bs[0]  Bs[1]  Bs[2]  Bs[3]  Bs[4]  Bs[5]  Bs[6]
        ∘      ∘      ∘      ∘      ∘      ∘      ∘      ∘
       Ss[0]  Ss[1]  Ss[2]  Ss[3]  Ss[4]  Ss[5]  Ss[6]  Ss[7]
    """        
    def __init__(self, Ws:list[np.ndarray], bc:str="finite"):
        self.Ws = Ws
        self.bc = bc
        
        self.dtype = self.Ws[0].dtype
        
    def check_right_canonical_form(self) -> None:
        assert self.llim == self.rlim == 1        
        for i in range(1, self.L):
            dim = self.Bs[i].shape[0]
            tmp = self.Bs[i].reshape(dim, -1)
            assert np.allclose(tmp @ tmp.conj().T, np.eye(dim, dtype=self.dtype)), f"第 {i} 个张量不是正交的"

    def set_right_canonical_form(self, qrnormalize=False) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        将任意的 MPS/MPO Ws 变为标准正交的 MPS/MPO (Bs, Ss)
        """
        assert self.bc=="finite", "正则形式只对开边界mps有定义！"
        self.set_mixed_canonical_form(i=-1)
        As = _left2right_QR(self.Ws, self.L, qrnormalize=qrnormalize)
        Bs, Ss = _right2left_SVD(As, self.L)
        Ss[0] = Ss[-1] = np.array(1., dtype=Ss[1].dtype)
        self.llim = self.rlim = 1
        self.Bs = Bs
        self.Ss = Ss
        
    def set_mixed_canonical_form(self, i:int) -> None:
        raise NotImplementedError("Subclasses should implement this.")


class MPS(BaseMPSExpectationValue):
    """
        .. code-block:: text
            
            |       |       |       |       |       |       |
        ----▷-------⬜-------⬜-------⬜-------⬜-------⬜-------⨞-----
           Ws[0]   Ws[1]   Ws[2]   Ws[3]   Ws[4]   Ws[5]   Ws[6]
                    ↑                               ↑
                llim = 1                         rlim = 5 

    """
    def __init__(self, Ws:list[np.ndarray], llim:int, rlim:int, bc="finite") -> None:
        super().__init__(Ws, bc)
        self.llim = llim
        self.rlim = rlim

    @property
    def L(self) -> int:
        """Number of physical sites; for an iMPS the len of the MPS unit cell."""
        return len(self.sites)

    @property
    def chi(self) -> list[int]:
        """List of local physical dimensions."""
        return [self.Ws[i].shape[-1] for i in range(self.L-1)]

    def check_mixed_canonical_form(self) -> None:
        dtype = self.Ws[0].dtype
        for i in range(self.llim):
            dim = self.Ws[i].shape[-1]
            tmp = self.Ws[i].reshape(-1, dim)
            assert np.allclose(tmp.T.conj() @ tmp, np.eye(dim, dtype=dtype)), f"第 {i} 个张量不是正交的"
        
        for i in range(self.rlim+1, self.L):
            dim = self.Ws[i].shape[0]
            tmp = self.Ws[i].reshape(dim, -1)
            assert np.allclose(tmp @ tmp.conj().T, np.eye(dim, dtype=dtype)), f"第 {i} 个张量不是正交的"

    def set_mixed_canonical_form(self, i:int):
        """
        .. code-block:: text
            
            |       |       |       |       |      |       |
        ----▷-------▷-------▷-------⬜-------⨞-------⨞-------⨞-----
           Ws[0]   Ws[1]   Ws[2]   Ws[3]   Ws[4]   Ws[5]   Ws[6]
                                    ↑
                            llim = llim = 3
        """
        assert self.llim <= self.rlim, "'llim' should at the left side of 'rlim'"
        self._move_llim(i)
        self._move_rlim(i)
    
    def _move_llim(self, i:int) -> None:
        """使得 self.Ws[j] 左侧全部为左正交的"""
        while self.llim < i:
            a, b = self.llim, self.llim+1
            self.Ws[a], self.Ws[b] = _left2right_QR_step(self.Ws[a], self.Ws[b])
            self.llim = self.llim + 1
            if self.rlim < self.llim: self.rlim = self.llim
    
    def _move_rlim(self, i:int) -> None:
        """使得 self.Ws[j] 右侧全部为右正交的"""
        while self.rlim > i:
            a, b = self.rlim-1, self.rlim
            self.Ws[a], self.Ws[b] = _right2left_QR_step(self.Ws[a], self.Ws[b])
            self.rlim = self.rlim - 1
            if self.llim > self.rlim: self.llim = self.rlim


    @classmethod
    def from_product_state(cls, L:int, state:list[str], bc="finite", dtype=np.float64) -> "MPS":
        """
        Examples
        --------
        Example to get a MPS of Neel state.
        
        >>> import quante as qt
        >>> L = 10
        >>> state = ["spin_up", "spin_down"] * (L//2)
        >>> psi = qt.tensor_new.networks.MPS.from_product_state(L, state)
        """
        str_to_vector = {
            "spin_down": spin_down(dtype),
            "spin_up": spin_up(dtype),
            "xplus": xplus(dtype),
            "xminus": xminus(dtype),
            "yplus": yplus(),
            "yminus": yminus()
        }
        Ws = [str_to_vector[site].reshape(1, -1, 1) for site in state]
        return cls(Ws, llim=0, rlim=L-1, bc=bc)



def _left2right_QR_step(W1:np.ndarray, W2:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    .. code-block:: text
        
               |       |                         |       |
              (b)     (d)                       (b)     (d)
               |       |           QR            |       |
        --(a)--◻--(c)--◻--(e)--   ---->   --(a)--▷--(f)--◻--(e)-- 
               W1      W2                       W1p     W2p
    
    """
    W1p, R = qr_tensor(W1)
    c, *e = W2.shape
    W2p = R @ W2.reshape(c, -1)
    return W1p, W2p.reshape(-1, *e)


def _right2left_QR_step(W1:np.ndarray, W2:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    .. code-block:: text
        
        .
               |       |                        |       |        
              (b)     (d)                      (b)     (d)       
               |       |           QR           |       |        
        --(a)--◻--(f)--⨞--(e)--   <----  --(a)--◻--(c)--◻--(e)-- 
              W1p     W2p                       W1      W2       
    
    MPS MPO 都可以
    """
    R, W2p = rq_tensor(W2)
    *a, f = W1.shape
    W1p = W1.reshape(-1, f) @ R
    return W1p.reshape(*a, -1), W2p


def _left2right_QR(Ws:np.ndarray, L:int, qrnormalize=False) -> tuple[np.ndarray, np.ndarray]:
    As, lognm = [None] * L, 0.0
    W1 = Ws[0]
    for i in range(L-1):
        As[i], W1 = _left2right_QR_step(W1, Ws[i+1])
        if qrnormalize:
            nm = np.linalg.norm(W1)
            lognm = np.log(nm) + lognm
            W1 = W1 / nm
    As[-1] = W1
    if not qrnormalize:
        nm = np.linalg.norm(As[-1])
        lognm = np.log(nm)
        As[-1] = As[-1] / nm
    return As


def _right2left_SVD(As, L, trunc_para=(None, None, None)) -> tuple[np.ndarray, np.ndarray]:
    """
    .. code-block:: text

        .                      |                        |
                              (b)                      (b)
                               |          SVD           |
        --(a)--▷--(d)--◇--(e)--⨞--(c)--  <----   --(a)--◻--(c)--
               U       S       B                        W
    """
    Ss, Bs = [None]*(L+1), [None] * L
    # trunc_err_sum = TruncationError(0.0, 1.0)
    lr_dims = [[0], list(range(1, As[0].ndim))]
    for i in range(L-1, 0, -1):
        U, Ss[i], Bs[i], trunc_err = svd_tensor(As[i], lr_index=lr_dims, trunc_para=trunc_para)
        As[i-1] = _SVD_contract_right(As[i-1], U, Ss[i])
    Bs[0] = As[0]
    return Bs, Ss


def _SVD_contract_right(A, U, S):
    """
    .. code-block:: text
        
        .      |                                      |
              (b)                                    (b)
               |                                      |
        --(a)--A--(c)--U--(d)--S--(e)  ---->   --(a)--W--(e)--
    
    MPS MPO 都可以
    
    >>> tc.einsum("abc,cd,de->abe", A, U, S)
    """
    *a, c = A.shape
    W = (A.reshape(-1, c) @ U) * S
    return W.reshape(*a, -1)
