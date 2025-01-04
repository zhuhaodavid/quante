# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2023-09-28 15:06:12
# @Last Modified by:   dzwang
# @Last Modified time: 2024-10-16 15:00:15
import copy
from ..generate import *
import numpy as _np
import numpy.linalg as _nla
from ..linalg.svd_robust import svd_truncate
from ..basicfun import load_hdf5, save_hdf5
from .automata import contract
from ..generate.matrix import pauli_matrix


__all__ = ["MPO", "canonical_form_mpo", "add_mpo"]


class MPO():
    def __init__(self, Ss:list[_np.ndarray], Ws:list[_np.ndarray], llim:int, rlim:int) -> None:
        """MPO construction

                          llim=3 --> <-- rlim=3
            |       |       |       |       |       |       |
        ----▷-------▷-------▷-------⬜-------⨞-------⨞-------⨞-----
            |       |       |       |       |       |       |
           Ws[0]   Ws[1]   Ws[2]   Ws[3]   Ws[4]  Ws[5]   Ws[6]
           
         ∘      ∘       ∘       ∘       ∘      ∘       ∘      ∘
        Ss[0]  Ss[1]   Ss[2]   Ss[3]  Ss[4]   Ss[5]   Ss[6]  Ss[7]

        Args:
            Ws (list[_np.ndarray]): All of these 4-order tensors
            Ss (list[_np.ndarray]): All of these spectrum at bond

        我们的MPO和MPS只是格点中心正交形式，构成量子态时只需要W和llim和rlim不需要也不能用S。
        对MPO进行操作时，必须移动正交中心。
        在移动正交中心时有一个技巧，MPO是通过SVD分解构造的右正交形式，所以当操作给定位置时，仅需找到指定位置的 W 以及对应的S即可，但是操作完必须保证可以还原出S使得左边的左正交形式不被破坏，如果还原不出来S，则意味着左边的正交形式被影响，此时一旦在此处涉及裁剪操作，就会因为不是正交中心而人为引入误差，因此技巧失效，需要手动移动正交中心。
        """
        self.Ss = Ss
        self.Ws = Ws
        self.llim = llim
        self.rlim = rlim
        self.L = len(Ws)

        self.left_traces = None
        self.right_traces = None

    def get_Ss(self) -> list[_np.ndarray]:
        return [copy.deepcopy(S) for S in self.Ss]

    def get_Ws(self) -> list[_np.ndarray]:
        return [copy.deepcopy(W) for W in self.Ws]

    def get_bond_dimension(self) -> list[int]:
        return [len(S) for S in self.Ss]

    def get_minimal_singular_value(self) -> list[str]:
        return [f"{S[-1]:.0e}" for S in self.Ss]

    def get_trace(self) -> float:
        matrix_Ws = [self.Ss[0]] + [_np.einsum("abbc->ac", W) for W in self.Ws] + [self.Ss[-1]]
        return _np.linalg.multi_dot(matrix_Ws) 

    def check_Hermitian(self) -> float:  # todo 用 MPO 加法检查
        """
        ||ρ - ρ†||. Only for little size
        """
        rho = contract(self.Bs, type="mpo")
        return _nla.norm(rho-rho.conj().transpose())

    def one_site_SW(self, i:int) -> _np.ndarray:
        """
                     (c)
                      |      
        (a)--Si--(b)--Wi--(e)  
                      |
                     (d)
        """
        Si = self.Ss[i] 
        Wi = self.Ws[i]
        a, = Si.shape
        b, c, d, e = Wi.shape
        # einsum("ab,bcde->acde", Si, Wi)
        return (Si.reshape(-1, 1) * Wi.reshape(b, -1)).reshape(a, c, d, e)

    def two_site_SWW(self, i:int) -> _np.ndarray:
        assert i+1 <= self.L-1, "PBC is NOT done."
        """
             (b)     (e)
              |       |
        (a)--SW--(d)--Wj--(g)
              |       |
             (c)     (f)
        """
        SW = self.one_site_SW(i)
        Wj = self.Ws[i + 1]
        a, b, c, d = SW.shape
        d, e, f, g = Wj.shape
        # einsum("abcd,defg->abcefg", SW, Wj)
        return (SW.reshape(-1, d) @ Wj.reshape(d, -1)).reshape(a, b, c, e, f, g)
    
    def two_site_WW(self, i:int) -> _np.ndarray:
        assert i+1 <= self.L-1, "PBC is NOT done."
        """
             (b)     (e)
              |       |
         (a)--W--(d)--Wj--(g)
              |       |
             (c)     (f)
        """
        j = i + 1
        Wi = self.Ws[i]
        Wj = self.Ws[j]
        a, b, c, d = Wi.shape
        d, e, f, g = Wj.shape
        return (Wi.reshape(-1, d) @ Wj.reshape(d, -1)).reshape(a, b, c, e, f, g)

    def get_trace_all(self, ) -> list[_np.ndarray]:
        # [S0, W0, W0W1, W0W1W2, ..., W0W1W2W3-WL]
        left_traces = [None] * len(self.Ss)
        left_traces[0] = self.Ss[0]
        left_trace = self.Ss[0]
        for n in range(self.L):
            left_trace = left_trace @ _np.einsum("abbc->ac", self.Ws[n], optimize=True)
            left_traces[n+1] = left_trace / _nla.norm(left_trace)
        self.left_traces = left_traces
        
        # [W0W1W2W3-WL, ..., W(L-1)WL, WL, S0]
        right_traces = [None] * len(self.Ss)
        right_traces[-1] = self.Ss[0]
        right_trace = self.Ss[0]
        for n in range(self.L-1, -1, -1):
            right_trace = _np.einsum("abbc->ac", self.Ws[n], optimize=True) @ right_trace
            right_traces[n] = right_trace / _nla.norm(right_trace)
        self.right_traces = right_traces
    
    def _two_sites_operator(self, operator:_np.ndarray, i:int):
        """                
                           ╭╮         ╭╮
                            |          |
                           (c)        (f)
                            |          |
        theta  =            ├---gate---┤
                            |          |                           
                           (b)        (e)                     
                            |          |                        
                    --(a)---Wi---(d)---Wj--(g)--  ----> --(a)---⬜---⬜---(g)--
                            |          |                         
                           (c)        (f)                      
                            |          |                      
                           ╰╯         ╰╯ 
        """
        # einsum("abcd,defg->abcefg", Wi, Wj)
        # einsum("cfbe,abcefg->ag", operator, WiWj)
        Wi, Wj = self.Ws[i], self.Ws[i+1]
        a, b, c, d = Wi.shape
        d, e, f, g = Wj.shape
        WiWj = (Wi.reshape(-1, d) @ Wj.reshape(d, -1)).reshape(a, b, c, e, f, g)
        O_WiWj = operator @ WiWj.transpose(1, 3, 0, 2, 4, 5).reshape(b*e, -1)
        O_WiWj = O_WiWj.reshape(b, e, a, c, f, g).transpose(2, 0, 3, 1, 4, 5)
        return _np.einsum("abbccd->ad", O_WiWj), _np.einsum("abbccd->ad", WiWj)

    def get_observable_two_site(self, operator:_np.ndarray, i:int):
        # 算近邻的快，那种长程关联函数还需要其他的构造 trace 的方法
        # [S0,              W0,            W0W1,        ..., W0W1W2W3-W(L-2), W0W1W2W3-W(L-1)]
        # [W0W1W2W3-W(L-1), W1W2W3-W(L-1), W2W3-W(L-1), ...,          W(L-1),             S0]
        trace_operator_DM, trace_DM = self._two_sites_operator(operator, i)   
        trace_operator_DM = self.left_traces[i] @ trace_operator_DM @ self.right_traces[i+2]
        trace_DM = self.left_traces[i] @ trace_DM @ self.right_traces[i+2]
        return trace_operator_DM/trace_DM

    def save(self, filename:str, group:str) -> None:
        L = len(self.Ws)
        Ss, Ws = dict(), dict()
        for i in range(L):
            Ss[f"{i}"] = self.Ss[i]
            Ws[f"{i}"] = self.Ws[i]
        Ss[f"{L}"] = self.Ss[L]
        save_hdf5(filename, group+"/rho", data={"Ss":Ss, "Ws":Ws, "llim":self.llim, "rlim":self.rlim})

    @classmethod
    def load(cls, filename:str, group:str) -> "MPO":
        group = group + "/rho"
        Ws_dict = load_hdf5(filename, group, dataname="Ws")
        Ss_dict = load_hdf5(filename, group, dataname="Ss")
        L = len(Ws_dict)
        Ws, Ss = [None]*L, [None]*(L+1)
        for i in range(L):
            Ws[i] = Ws_dict[f"{i}"]
            Ss[i] = Ss_dict[f"{i}"]
        Ss[L] = Ss_dict[f"{L}"]
        llim = load_hdf5(filename, group, dataname="llim")
        rlim = load_hdf5(filename, group, dataname="rlim")
        return MPO(Ss, Ws, llim, rlim)
    
    @classmethod
    def get_initial_DM(cls, L:int, init_rho:str, D:int=1) -> "MPO":
        from ..linalg.operations import kron
        d = 2
        load_shape = [D, d, d, D]
        Id = pauli_matrix("I")
        up = pauli_matrix("u")
        down = pauli_matrix("d")
        if init_rho=="Id":
            Ws = [Id.reshape(load_shape)/2] * L
        elif init_rho=="Up":
            Ws = [kron(up, up).reshape(load_shape)] * L
        elif init_rho == "Down":
            Ws = [kron(down, down).reshape(load_shape)] * L
        elif init_rho == "Right":
            right = _np.sqrt(1/2)*(up+down) 
            Ws = [kron(right, right).reshape(load_shape)] * L
        elif init_rho=="Neel":
            assert L%2 == 0, "Size L must be odd due to Neel state!"
            W1 = kron(up, up).reshape(load_shape)
            W2 = kron(down, down).reshape(load_shape)
            Ws = [W1, W2] * (L//2)
        Ss = [_np.ones(1)] * (L+1)
        return MPO(Ss, Ws, llim=0, rlim=0)

    

def _add_each(W1, W2)->_np.ndarray:
    W1_D0, d, d, W1_D1 = W1.shape
    W2_D0, d, d, W2_D1 = W2.shape
    W = _np.zeros((W1_D0+W2_D0, d,d, W1_D1+W2_D1), dtype=W1.dtype)
    W[:W1_D0, :,:, :W1_D1], W[W1_D0:, :,:, W1_D1:] = W1, W2
    return W


def _add_left(W1l, W2l, alpha, beta) -> _np.ndarray:
    W1_D0, d, d, W1_D1 = W1l.shape
    W2_D0, d, d, W2_D1 = W2l.shape
    W = _np.zeros((1, d,d, W1_D1+W2_D1), dtype=W1l.dtype)
    W[:,:,:,:W1_D1], W[:,:,:,W1_D1:] = alpha*W1l, beta*W2l
    return W


def _add_right(W1r, W2r) -> _np.ndarray:
    W1_D0, d, d, W1_D1 = W1r.shape
    W2_D0, d, d, W2_D1 = W2r.shape
    W = _np.zeros((W1_D0+W2_D0, d,d, 1), dtype=W1r.dtype)
    W[:W1_D0,:,:,:], W[W1_D0:,:,:,:] = W1r, W2r
    return W


def add_mpo(W1s:list[_np.ndarray], W2s:list[_np.ndarray], alpha:float=1., beta:float=1.)->list[_np.ndarray]:
    """
    Args: W1s, W2s, alpha, beta
    Returns: Ws
    """
    Ws = [None]*len(W1s)
    # * add every one
    for i, (W1, W2) in enumerate(zip(W1s, W2s)):
        Ws[i] = _add_each(W1, W2)
    # * updata the most left
    Ws[0] = _add_left(W1s[0], W2s[0], alpha, beta)
    # * updata the most right
    Ws[-1] = _add_right(W1s[-1], W2s[-1])
    return Ws


def _left_to_right_QR(r, W) ->tuple[_np.ndarray, _np.ndarray]:
    """
                (c)                     d
                 |          QR          |
    (a)--r--(b)--W--(e)    ---->    D0--q--D1 D1--r--D1
                 |                      |
                (d)                     d
    """
    D0, D1 = r.shape
    D1, d, d, D2 = W.shape
    rW = (r@W.reshape(D1, -1)).reshape(-1, D2)
    q, r = _nla.qr(rW)  # todo 负号可能是从QR分解出来的
    return q.reshape(D0, d, d, -1), r


def _right_to_left_SVD(A, S, Dc, eps) ->tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """
                              d                   (b)
                              |        SVD         |
    D0--A--D1  D1--S--D1  D1--B--D2   <----   (a)--A--(d)--S--(e)
                              |                    |           
                              d                   (c)      
    """
    D0, d, d, D1 = A.shape
    # AS = (A.reshape(-1, D1)@_np.diag(S)).reshape(D0, -1)
    AS = (A.reshape(-1, D1) * S).reshape(D0, -1)
    A, S, B = svd_truncate(AS, Dc, eps)
    return A.reshape(D0, -1), S, B.reshape(-1, d, d, D1)


def canonical_form_mpo(Ws:list[_np.ndarray], Dc:int=_np.inf, eps:float=1.e-15) ->MPO:
    """
    Args: Ws
    Update: self.Ss, self.Bs
    """
    L = len(Ws)
    
    # * left ->QR-> right
    As, Ss, Bs = [None]*L, [_np.array([1.])]*(L+1), [None]*L
    r = _np.diag(Ss[0])
    for i, W in enumerate(Ws):
        As[i], r = _left_to_right_QR(r, W)
    As[-1] = _np.einsum("abcd,de->abce", As[-1], r, optimize=True)  # set the most left S is [[1.]]
    
    # * left <-SVD<- right
    for i in range(1, L):
        A_, Ss[-i-1], Bs[-i] = _right_to_left_SVD(As[-i], Ss[-i], Dc, eps)
        D0, d, d, D1 = As[-i-1].shape
        As[-i-1] = (As[-i-1].reshape(-1, D1)@A_).reshape(D0, d, d, -1)
        
    # * updata boundar B
    Bs[0] = _np.einsum("abcd,de->abce", As[0], _np.diag(Ss[1]), optimize=True)    
    return MPO(Ss, Bs)



