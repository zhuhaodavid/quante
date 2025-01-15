# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-07-10 21:48:14
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-18 03:36:34

# TODO: swap sites, apply submpo

import torch as tc
from typing import Union, TypeVar, Optional, TYPE_CHECKING, Generator

from . import tnfuncs as tf
from ..utils import clone, promote_dtype
from ..linalg.decomp import eigh, qr, rq, svd, truncate, log_or_not_update
from ...generate.matrix import pauli_matrix
from ...linalg.svd_robust import TruncationError

import math as math_lib # type: ignore
import copy
import warnings
import numpy as np
from numbers import Number

if TYPE_CHECKING:  # 类型检查时，导入 torch
    from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState

__all__ = [
    "MPS",
    "MPO",
]

T = TypeVar('T')

class TensorTrain:
    def __init__(self, Ws: list[tc.Tensor], Ss: Optional[list[tc.Tensor]] = None, llim: Optional[int] = None, rlim: Optional[int] = None, lognm: Optional[float] = None, L:Optional[int] = None) -> None:
        """
        .. code-block:: text
            
            =>       |      |      |      |      |      |      |
                —————▷——————▷——————◻——————◻——————◻——————⨞——————⨞—————
                   Ws[0]  Ws[1]  Ws[2]  Ws[3]  Ws[4]  Ws[5]  Ws[6]
                ∘       ∘       ∘      ∘       ∘      ∘      ∘      ∘
              Ss[0]  Ss[1]   Ss[2]   Ss[3]  Ss[4]  Ss[5]  Ss[6]  Ss[7]
                                 ↑             ↑
                              llim=2         rlim=4
        
        无论是 MPO 还是 MPS 都遵从这个格式
        对于标准的正则形式，llim=-1, rlim=1
        """
        assert isinstance(Ws[0], tc.Tensor)
        self.data = Ws
        
        self.L = len(Ws) if L is None else L
        assert self.L == len(Ws) or self.L == tc.inf
        
        if self.L == tc.inf:
            self.llim, self.rlim = None, None
        else:
            self.llim = llim if llim is not None else 0
            self.rlim = rlim if rlim is not None else len(Ws) - 1
            
        self.Ss = Ss if Ss is not None else [None] * (len(Ws) + 1)
        self.dtype = Ws[0].dtype
        self.device = Ws[0].device
        self.lognm = lognm if lognm is not None else tc.tensor(0.0, dtype=tc.float64,device=self.device)
    
    def __len__(self):
        return self.L
    
    def __getitem__(self, key):
        return self.data[key]
    
    def add_(self, anotherTT):
        """Tensor Train 的加法，MPO,MPS 都可用
        todo 加法不能保证正交形式，但是可以通过添加辅助自旋保持正交形式，是否有意义？
        """
        # 先拿到 lognorm
        a = self.lognm
        b = anotherTT.lognm
        # 新的 norm 为 log(exp(a) + exp(b)) = logaddexp(a, b)
        newlognm = tc.logaddexp(a, b)
        # 这样 alpha, beta 都在 0-1 之间
        alpha = tc.exp(a - newlognm)
        beta = tc.exp(b - newlognm)
        # 跟新数据
        self.data = tf.add([self.data, anotherTT.data], [alpha, beta])
        self.lognm = newlognm
    
    def copy(self):
        return type(self)(clone(self.data), clone(self.Ss), copy.deepcopy(self.llim), copy.deepcopy(self.rlim), copy.deepcopy(self.lognm))
    
    def __add__(self, anotherTT):
        """Tensor Train 的自加，会改变 self 的数据！"""
        out = self.copy()
        out.add_(anotherTT)
        return out

    def sub_(self, anotherTT):
        """Tensor Train 的自减，类似加法，MPO,MPS 都可用"""
        # 先拿到 lognorm
        a = self.lognm
        b = anotherTT.lognm
        # 新的 norm 为 log(exp(a) + exp(b)) = logaddexp(a, b)
        newlognm = tc.logaddexp(a, b)
        # 这样 alpha, beta 都在 0-1 之间
        alpha = tc.exp(a - newlognm)
        beta = tc.exp(b - newlognm)
        # 跟新数据
        self.data = tf.add([self.data, anotherTT.data], [alpha, -beta])
        self.lognm = newlognm
    
    def __sub__(self, antoherTT):
        """Tensor Train 的减法，类似加法，MPO,MPS 都可用"""
        out = self.copy()
        out.sub_(antoherTT)
        return out
    
    def __rmul__(self, number:tc.Tensor):
        """乘以数，为了效率，不复制张量！！！
        
        #!!! 因此改变 data 的值会影响原始张量
        
        Examples
        --------
        >>> 𝜓1 = tn.MPS.random(4, linkdims=3, dtype=tc.float64)
        >>> print(𝜓1.data[0][0,1,0])
        >>> 𝜓2 = 2 * 𝜓1
        >>> 𝜓2.data[0][0,1,0] = 1.
        >>> print(𝜓1.data[0][0,1,0])
        tensor(1.0229, dtype=torch.float64)
        tensor(1., dtype=torch.float64)
        """
        try:
            lognmber = tc.log(number) + self.lognm
        except TypeError:  # when number is pure float form,
            import numpy as np
            lognmber = np.log(number) + self.lognm
        return type(self)(self.data, lognm=lognmber)

    def maxbonddim(self):
        return max([i.shape[0] for i in self.data[1:]])
    
    def bonddims(self):
        return [i.shape[0] for i in self.data[1:]]
    
    def to(self, dtype=None, device=None):
        for i in range(len(self.data)):
            self.data[i] = self.data[i].to(dtype=dtype, device=device)
        for i in range(len(self.data)+1):
            if self.Ss[i] is not None:
                self.Ss[i] = self.Ss[i].to(device=device)
        self.device = self.data[0].device
        self.dtype = dtype
        
    def save(self, path:str):
        data_dic = dict()
        L = len(self.data)
        data_dic["type"] = type(self)
        data_dic["L"] = L
        for i in range(L):
            data_dic[f'Ws/{i}'] = self.data[i]
            data_dic[f'Ss/{i}'] = self.Ss[i]
        data_dic["llim"] = self.llim
        data_dic["rlim"] = self.rlim
        data_dic["lognm"] = self.lognm
        tc.save(data_dic, path)

    @classmethod
    def loadpth(cls, path):
        data = tc.load(path)
        return data["type"](data["Ws"], data["Ss"], data["llim"], data["rlim"], data["lognm"])

    def to_matrix(self):
        if isinstance(self, MPS):
            return tf._full_contract_mps(self.data) * tc.exp(self.lognm)
        elif isinstance(self, MPO):
            return tf._full_contract_mpo(self.data) * tc.exp(self.lognm)
        raise AttributeError("TensorTrain can not be full contract")

    def inner(self, anotherTT, logscale=False, conj=True):
        """计算 <ψ|ϕ>: ψ.inner(ϕ)

        如何使 MPO，那么等价于 tr(ψ^† ϕ)

        Parameters
        ----------
        anotherMPS : MPS
            另外一个 tt
        logscale : bool, optional
            是否计算对数模, by default False
        conj : bool, optional
            第一个位置是否去共轭, by default True

        Returns
        -------
        tc.Tensor
            计算结果
        """
        assert isinstance(anotherTT, type(self))
        if conj:
            conjdata = [i.conj() for i in self.data]
        else:
            conjdata = self.data
        if logscale:
            coef, lognm = tf.tn_inner(conjdata, anotherTT.data, logscale=True) 
            return tc.log(coef) + lognm + self.lognm + anotherTT.lognm
        else:
            return tf.tn_inner(conjdata, anotherTT.data, logscale=False) * tc.exp(self.lognm) * tc.exp(anotherTT.lognm)

    def norm(self, lognorm=False):
        if self.is_canonical_form():
            return self.lognm if lognorm else tc.exp(self.lognm)
        elif self.is_orthogonal_form():
            nm = tc.norm(self.data[self.llim])
            if lognorm:
                return tc.log(nm) + self.lognm
            else:
                return nm * tc.exp(self.lognm)
        else:
            if lognorm:
                coef, lognm = tf.tn_norm(self.data, lognorm=True)
                return tc.log(coef) + lognm + self.lognm
            else:
                coef = tf.tn_norm(self.data, lognorm=False)
                return coef * tc.exp(self.lognm)
    
    def normalize_(self):
        if self.is_canonical_form():
            self.lognm *= 0.
        elif self.is_orthogonal_form():
            nm = tc.norm(self.data[self.llim])
            self.data[self.llim] /= nm
            self.lognm *= 0.0
        else:
            self.orthogonalize_(0)
            self.lognm *= 0.0
        
    def entanglement_entropy(self, bonds: list[int]):
        if self.is_canonical_form():
            res = []
            for bond in range(bonds):
                s = self.Ss[bond]
                assert tc.norm(s) - 1 < 1E-8
                res.append(-tc.sum(s**2 * tc.log(s**2)))
            return res
        
        psi = self.copy()
        res = []
        for bond in bonds:
            psi.orthogonalize_(bond)
            data = psi.data[bond]
            _, s, _ = tc.linalg.svd(data.reshape(data.shape[0], -1))
            assert tc.norm(s) - 1 < 1E-8
            res.append(-tc.sum(s**2 * tc.log(s**2)))
            
        return res
    
    ####################################
    # 以下是张量网络的核心方法
    ####################################

    def move_llim_(self, j):
        """使得 self.data[j] 左侧全部为左正交的"""
        while self.llim < j:
            a, b = self.llim, self.llim + 1
            self.data[a], self.data[b] = tf._left2right_QR_step(self.data[a], self.data[b])
            self.llim = self.llim + 1
            if self.rlim < self.llim:
                self.rlim = self.llim

    def move_rlim_(self, j):
        """使得 self.data[j] 右侧全部为右正交的"""
        while self.rlim > j:
            a, b = self.rlim - 1, self.rlim
            self.data[a], self.data[b] = tf._right2left_QR_step(self.data[a], self.data[b])
            self.rlim = self.rlim - 1
            if self.llim > self.rlim:
                self.llim = self.rlim
    
    def is_orthogonal_form(self):
        return self.llim == self.rlim

    def is_canonical_form(self):
        return self.llim is None and self.rlim is None
    
    def check_canonical_form(self):
        for i in range(len(self.data)):
            dim = self.data[i].shape[0]
            tmp = self.data[i].reshape(dim, -1)
            assert tc.allclose(tmp @ tmp.H, tc.eye(dim, dtype=tmp.dtype, device=tmp.device)), f"第 {i} 个张量不是正交的, diff = {tc.dist(tmp @ tmp.H, tc.eye(dim, dtype=tmp.dtype, device=tmp.device))}"
    
    def check_orthogonal_form(self):
        if self.llim is None:
            self.check_canonical_form()
            
        for i in range(self.llim):
            dim = self.data[i].shape[-1]
            tmp = self.data[i].reshape(-1, dim)
            assert tc.allclose(tmp.H @ tmp, tc.eye(dim, dtype=tmp.dtype, device=tmp.device)), f"第 {i} 个张量不是正交的"
        
        for i in range(self.rlim+1, len(self.data)):
            dim = self.data[i].shape[0]
            tmp = self.data[i].reshape(dim, -1)
            assert tc.allclose(tmp @ tmp.H, tc.eye(dim, dtype=tmp.dtype, device=tmp.device)), f"第 {i} 个张量不是正交的"

    def orthogonalize_(self, j: int, normalize=True):
        """
        .. code-block:: text

          .      |      |      |      |      |      |      |
            -----▷------▷------▷------◻------⨞------⨞------⨞-----
               Ws[0]  Ws[1]  Ws[2]  Ws[3]  Ws[4]  Ws[5]  Ws[6]
                                      ↑
                               llim = rlim = 3
        """
        self.move_llim_(j)
        self.move_rlim_(j)
        if normalize:
            oldnm = tc.norm(self.data[j])
            self.data[j] = self.data[j] / oldnm
            self.lognm = self.lognm + tc.log(oldnm)

    def canonicalize_(self, trunc_para:tuple[int,float,float]=(None,None,None), qrnormalize=False):
        """正则化，即：
        
        .. code-block:: text
        
            .     |      |      |      |
            ------⨞------⨞------⨞------⨞-----
                Ws[0]  Ws[1]  Ws[2]  Ws[3]
              ∘       ∘       ∘      ∘       ∘
            Ss[0]  Ss[1]   Ss[2]   Ss[3]  Ss[4]
            
            llim = rlim = None

        一旦进入这个形式不能再移动正交中心
        
        前从左到右 qr，再从右到左 svd
        
        Parameters
        ----------
        trunc_para: tuple[int,float,float], optional
            截断参数，默认 (None,None,None) 表示不截断。
            第一个参数 chi_max 表示最大的奇异值数目，第二个参数 svd_min 表示最小的奇异值，第三个参数 trunc_cut 表示截断阈值。
            例如 trunc_para=(10, 1E-10, 1E-12) 表示截断前 10 个奇异值，最小的奇异值大于 1E-10，截断阈值大于 1E-12。
            截断参数仅对 svd 有效。
        qrnormalize: bool, optional
            是否对 qr 得到的矩阵进行归一化，默认 False。
        
        """
        self.data, self.Ss, lognm, trunc_err = tf.canonicalize(self.data, trunc_para, qrnormalize)
        self.llim = self.rlim = None
        self.lognm = self.lognm + lognm
        return trunc_err


    def apply_gate_(
        self,
        pos:int,
        gate:Union[tc.Tensor, tuple[tc.Tensor, str]],
        *,
        direction: str = None,
        svd_alg: str = "svd", 
        trunc_para: tuple[int, float, float] = (None, None, None),
        updateS=False,
        normalize=False,
        gate_range=2,
        unitary_gate=False,
    ):
        """作用单体或两体门.
        
        Parameters
        ----------
        pos: int
            作用在第 pos 格点上的门
        gate: Union[tc.Tensor, tuple[tc.Tensor, str]]
            门的矩阵表示，或者是 (矩阵, 方向) 的元组
        direction: str, optional
            作用方向，默认 None 表示作用在左边
        svd_alg: str, optional
            用于 SVD 的算法，默认 "svd"，可选 "eig"
        trunc_para: tuple[int, float, float], optional
            截断参数，默认 (None,None,None) 表示不截断。
            第一个参数 chi_max 表示最大的奇异值数目，第二个参数 svd_min 表示最小的奇异值，第三个参数 trunc_cut 表示截断阈值。
            例如 trunc_para=(10, 1E-10, 1E-12) 表示截断前 10 个奇异值，最小的奇异值大于 1E-10，截断阈值大于 1E-12。
            截断参数仅对 svd 有效。
        updateS: bool, optional
            是否保存过程的中的奇异谱，默认 False
        normalize: bool, optional
            是否对矩阵进行归一化，默认 False
        gate_range: int, optional
            门的作用范围，默认 2 表示作用在两格点上
        unitary_gate: bool, optional
            门是否为幺正，默认 False
        
        Returns
        -------
        TruncationError
        
        Note
        ----
        
        .. code-block:: text
        
            .      |         |               
                  (1)       (2)                  
                   |         |                
                   ├-gate2_b-┤              
                   |         |                      |         |        
                  (3)       (4)                    (c)       (f)       
                   |         |                      |         |            
            --(a)--⨞---(d)---⨞--(g)--  -->   --(a)--⨞---(d)---⨞--(g)-- 
        
        Examples
        --------
        >>> gates = ham.trotter_gates(L, tau=tau, order='2', evolve_type='time', pauli=False)
        >>> U_tau = MPO.eye(L)
        >>> for pos_cur, gate in zip(*gates):
        >>>     U_tau.apply_gate_(pos_cur, gate)
        """
        if gate_range == 1:
            if not unitary_gate:
                self.move_llim_(pos)
                self.move_rlim_(pos)
            
            # ------- main part -------
            W = self._apply_1b_gate(pos, gate)
            return self.update_single_site_(pos, W, normalize=normalize, unitary_gate=unitary_gate)
            # -------------------------
        
        else:
            # todo 幺正门作用的时候，同样可以不移动正交中心，是否会导致数值误差？
            # 不移动正交中心，那就和 cannocalize 没有什么区别，这里不单独写了
            if self.is_canonical_form():
                assert unitary_gate, "正则形式下只能使用 unitary_gate"
            else:
                self.move_llim_(pos)
                self.move_rlim_(pos+1)
            
            # ------- main part -------
            W = self._apply_2b_gate(pos, gate)
            return self.update_two_site_(pos, W, direction=direction, svd_alg=svd_alg, trunc_para=trunc_para, normalize=normalize, updateS=updateS)
            # -------------------------


    def _convert_gate(self, gate, site_num):
        if gate.ndim == site_num == 2:
            try:
                gate = gate.reshape(2,2,2,2)
            except Exception as e:
                raise ValueError(f"gate shape error: {gate.shape}, site_num={site_num}")
        if isinstance(gate, np.ndarray):
            if np.iscomplexobj(gate):
                gate = tc.tensor(gate, dtype=tc.complex128, device=self.device)
                if not self.dtype.is_complex:
                    self.to(dtype=tc.complex128, device=self.device)
            else:
                if self.dtype.is_complex:
                    gate = tc.tensor(gate, dtype=tc.complex128, device=self.device)
                else:
                    gate = tc.tensor(gate, dtype=tc.float64, device=self.device)
        else:
            if gate.dtype.is_complex and not self.dtype.is_complex:
                self.to(dtype=tc.complex128, device=self.device)
            elif not gate.dtype.is_complex and self.dtype.is_complex:
                gate = gate.to(dtype=tc.complex128, device=self.device)
        return gate


    def update_single_site_(self, pos, phi, normalize=False, unitary_gate=False):
        if not unitary_gate:
            self.move_llim_(pos)
            self.move_rlim_(pos)
        
        assert self.data[pos].ndim == phi.ndim, "维度不匹配"
        phi, self.lognm = log_or_not_update(phi, self.lognm, use_log=normalize)
        self.data[pos] = phi
        return TruncationError(0.0, 1.0)
    
    def update_two_site_(self, pos, W, direction, *,
                   svd_alg="eig",
                   trunc_para:tuple[int,float,float]=(None,None,None),
                   normalize=False,
                   pertube=None,
                   updateS=True
                   ) -> TruncationError:
        """
        将两格点上的张量还原到 self.data 中
        
        可以使用的方法 method 包括："qr", "svd", "eig"
        
        trunc_para 是一个包含三个数的元组，分别表示:
    
        :chi_max: int, 截断的最大值
        
        :svd_min: float, SVD 的最小值
        
        :trunc_cut: float, 截断的阈值
            
        - direction 指的是作用后得到:
                                
            .. code-block:: text
            
                right               left                center
                    │   │               │   │               │   │   
                 ---▷---◻---         ---◻---⨞---         ---◻---◻---
        """
        # todo 正则形式如何更新
        if self.is_canonical_form():
            return self.update_two_site_cannonical_(pos, W, svd_alg=svd_alg, trunc_para=trunc_para, normalize=normalize)
        
        self.move_llim_(pos)
        self.move_rlim_(pos+1)
        
        # -------------- 使用 qr ------------
        if svd_alg == "qr":
            assert trunc_para == (None, None, None) and pertube is None, 'qr method do not need trunc_para, normalize and pertube'
            Wshape = W.shape
            halfdim = W.ndim // 2
            direction = "right" if direction is None else direction
            if direction == "right":
                U, A = qr(W.reshape(*Wshape[:halfdim],-1))
                A, self.lognm = log_or_not_update(A, self.lognm, use_log=normalize)
                self.data[pos] = U
                self.data[pos + 1] = A.reshape(-1,*Wshape[halfdim:])
                self.llim = self.rlim = pos + 1
            elif direction == "left":
                B, U = rq(W.reshape(-1,*Wshape[halfdim:]))
                B, self.lognm = log_or_not_update(B, self.lognm, use_log=normalize)
                self.data[pos] = B.reshape(*Wshape[:halfdim],-1)
                self.data[pos + 1] = U
                self.llim = self.rlim = pos
            else:
                raise ValueError(f"not defined direction (left or right): {direction}")
            return TruncationError(0.0, 1.0)
        
        # -------------- 使用 svd ------------
        elif svd_alg == "svd":
            assert pertube is None, 'svd method do not need pertube'
            W1, S, W2, trunc_err = svd(W, trunc_para=trunc_para)
            S, self.lognm = log_or_not_update(S, self.lognm, use_log=normalize)
                
            direction = "right" if direction is None else direction
            if direction == "right":
                self.data[pos] = W1
                self.data[pos + 1] = S.reshape(-1, *([1]*(W2.ndim-1))) * W2
                self.llim = self.rlim = pos + 1
            elif direction == "left":
                self.data[pos] = W1 * S
                self.data[pos + 1] = W2
                self.llim = self.rlim = pos
            elif direction == "center":
                sqrtS = tc.sqrt(S)
                self.data[pos] = W1 * sqrtS
                self.data[pos + 1] = sqrtS.reshape(-1, *([1]*(W2.ndim-1))) * W2
                self.llim, self.rlim = pos, pos + 1
            else:
                raise ValueError(f"not defined direction (left or right): {direction}")
            
            if updateS:
                self.Ss[pos + 1] = S
                
            return trunc_err
        
        # -------------- 使用 eig ------------
        elif svd_alg == "eig":
            # todo 目前 eig 使用的是自动判断方向，是否有更好的做法？
            W1, S, W2, trunc_err, eigdirection = eigh(W, direction=None, trunc_para=trunc_para, pertube=pertube)
            
            # 如果 eig 选择的方向与需要的方向不一样，通过 qr 调整回来
            if direction is None:
                direction = eigdirection
            elif direction == "right" and eigdirection == "left":
                W1, R = qr(W1)
                W2 = (R @ W2.reshape(W2.shape[0],-1)).reshape(*W2.shape)
            elif direction == "left" and eigdirection == "right":
                L, W2 = rq(W2)
                W1 = (W1.reshape(-1, W1.shape[-1]) @ L).reshape(*W1.shape)
            
            self.data[pos] = W1
            self.data[pos + 1] = W2
            if direction == "right":
                self.llim = self.rlim = pos + 1
                self.data[pos + 1], self.lognm = log_or_not_update(
                    self.data[pos + 1], self.lognm, use_log=normalize)
            elif direction == "left":
                self.llim = self.rlim = pos
                self.data[pos], self.lognm = log_or_not_update(
                    self.data[pos], self.lognm, use_log=normalize)
            else:
                raise ValueError(f"not defined direction (left or right): {direction}")
            
            if updateS:
                self.Ss[pos + 1] = S if not normalize else S / tc.norm(S)
                
            return trunc_err

    def update_two_site_cannonical_(self, pos, W,
                   svd_alg="eig",
                   trunc_para:tuple[int,float,float]=(None,None,None),
                   normalize=False) -> TruncationError:
        """
        警告：W 不应该包含 S
        
        trunc_para 是一个包含三个数的元组，分别表示:
        
        :chi_max: int, 截断的最大值
            
        :svd_min: float, SVD 的最小值
            
        :trunc_cut: float, 截断的阈值
        
        .. code-block:: text
        
            eig method
                                |         |
                               (c)       (f)
                                |         |
            W      =            ├-gate2_b-┤
                                |         | 
                               (b)       (e)
                                |         |                  
                         --(a)--⨞---(d)---⨞--(g)--  

                                |         |
                               (c)       (f)
                                |         |
            theta  =            ├-gate2_b-┤
                                |         |                       |     |     
                               (b)       (e)                     (c)   (f)  
                                |         |                       |     |   
                      --◇--(a)--⨞---(d)---⨞--(g)--  ----> --(a)---▷--◇--⨞---(g)-- 
                       S1                                        W1  S  W2

            S1 * W = theta
                                                                      |    
                                                                     (c)   
                                                                      |    
            S1 * W * W2.dagger()  =  theta * W2.dagger()  =   --(a)---▷--◇-
                                                                     W1  S

                                                                       |    
                                                                      (c)   
                                                                       |    
            W * W2.dagger()  =  S1^-1 * theta * W2.dagger()  =   --◇---▷--◇-
                                                                S1^-1  W1  S
        """
        next_pos = pos + 1 if self.L != tc.inf else (pos + 1) % len(self.data)
        theta = self.Ss[pos].reshape(-1, *[1]*(W.ndim-1)) * W
        if svd_alg == 'eig':
            
            if self.L == tc.inf:
                raise ValueError("正则形式下不能使用 eig 方法，因为证明中用到了本征分解的正确性，如果有裁剪，会破坏左正交的正交性质，并且在无穷长链中，这个破坏会逐步积累。")
            
            W1, S, W2, err, direction = eigh(theta, trunc_para=trunc_para)
            if direction == 'right':
                _, W2 = rq(W2)
            W1 = self._resume_canonical(W, W2)
            W1, self.lognm = log_or_not_update(W1, self.lognm, use_log=normalize)
            self.data[pos] = W1
            self.Ss[next_pos], self.data[next_pos] = S/tc.norm(S) if normalize else S, W2
            
        elif svd_alg == 'svd':
            W1, S, W2, err = svd(theta, trunc_para=trunc_para)
            S, self.lognm = log_or_not_update(S, self.lognm, use_log=normalize)
            self.Ss[next_pos] = S
            self.data[pos] = W1 * S / self.Ss[pos].reshape(-1,*([1]*(W1.ndim-1)))
            self.data[next_pos] = W2
        else:
            raise ValueError("method 只能为 eig 或 svd")
        return err

    def apply_mpo_(
                self,
                Ws_mpo: 'MPO',
                trunc_para: tuple[int, float, float] = (None, None, None),
                updateS = True,
                normalize = False,
            ) -> T:
        r"""density matrix method
        
        Parameters
        ----------
        Ws_mpo: MPO
            要作用到 self 的 MPO
        trunc_para : tuple[int, float, float], optional
            截断参数，包含以下元素：
            - `chi_max` (int): 截断的最大值。
            - `svd_min` (float): SVD 的最小值。
            - `trunc_cut` (float): 截断的阈值。
            默认为 `(None, None, None)`。
        dtype: torch.dtype
            计算的精度
        updateS: bool
            是否更新奇异值
        normalize: bool
            是否归一化
        
        Returns
        -------
        TruncationError
            
        Notes
        -----
        
        .. code-block:: text
        
            收缩如下网络：
                    |   |       |
                Ws  |---|--...--|
                ψ   └---┴--...--┘
        
            利用 density matrix 的方法进行收缩：

            1). 首先得到最右侧的约化密度矩阵，裁剪：
                    ╭╮  ╭╮    ╭╮  |         |
                Ws  |---|--...|---|         ▽ V1
                ψ   └---┴--...┴---┘         |
                ψ†  ┌---┬--...┬---┐  -->    ◇        记录：  -⨞-
                Ws† |---|--...|---|         |                V1
                    ╰╯  ╰╯    ╰╯  |         △
                                            |
            
            2). 计算右侧第二个位置约化密度矩阵，裁剪：
                                   ┌-╨-┐
                     ╭╮  ╭╮    ╭╮  |   △ V1      ║
                 Ws  |---|--...|---|---|         ▽ V2
                 ψ   └---┴--...┴---┴---┘         |              |
                 ψ†  ┌---┬--...┬---┬---┐  -->    ◇       记录： -⨞--⨞-
                 Ws† |---|--...|---|---|         |              V2 V1
                     ╰╯  ╰╯    ╰╯  |   ▽         △
                                   └-╥-┘         ║
            
            3). 计算右侧第二个位置约化密度矩阵，裁剪：
                              ┌--╨--┐
                              │     △
                              │   ┌-╨-┐
                    ╭╮  ╭╮    │   |   △         ║
                Ws  |---│--...|---┼---|         ▽ V3
                ψ   └---┴--...┴---┴---┘         |             |  |
                ψ†  ┌---┬--...┬---┬---┐  -->    ◇      记录： -⨞--⨞--⨞-
                Ws† |---|--...|---|---|         |             V3 V2 V1
                    ╰╯  ╰╯    |   |   ▽         △
                              |   └-╥-┘         ║
                              |     ▽
                              └--╥--┘
            
            ... 依次循环

            4). 最后一个：
                    |     |
                    |     △
                    |   ┌-╨  ...                            |  |       |  |
                    |   |                             记录： ◻--⨞- ... -⨞--⨞-
                Ws  |---|--- ...         |
                ψ   └---┴--- ...   ->    ◻--
        """
        n = len(Ws_mpo.data)
        
        if n == 1:
            self.data[0] = Ws_mpo.data[0] @ self.data[0]
            return None
        
        dtype = self.dtype
        Lenvs = self._dm_get_Lenvs(Ws_mpo.data, self.data, n, dtype)
        
        V = tc.tensor(1., dtype=dtype, device=self.device).reshape(1,1)
        R = tc.tensor(1., dtype=dtype, device=self.device).reshape(1,1,1)
        R = self._dm_get_R(Ws_mpo.data[n-1], self.data[n-1], R, V)
        nm = tc.norm(V)
        linkdim = 1
        
        trunc_err_sum = TruncationError(0.0, 1.0)
        for j in range(n - 1, 0, -1):
            # 拿到密度矩阵
            rho = tf._dm_get_rho(Lenvs[j-1], R)
            
            # 本征分解
            prod_dim = self.data[j].shape[0] * Ws_mpo.data[j].shape[0]
            chi_max = trunc_para[0]
            iDc = min(chi_max, prod_dim) if chi_max is not None else prod_dim
            
            S, V = tc.linalg.eigh(rho)
            # print(rho.shape)
            S, V = S.flip(0), V.flip(1)
            tc.clamp_(S, min=0)
            tc.sqrt_(S)
            good, trunc_err = truncate(S, iDc, trunc_para[1], trunc_para[2])
            S = S[good]
            V = V[:, good]
            trunc_err_sum += trunc_err
            
            # 替换
            linkdim2 = len(S)
            self.data[j] = V.T.reshape(linkdim2, *self.data[j].shape[1:-1], linkdim)
            linkdim = linkdim2

            # 前进一步
            R = self._dm_get_R(Ws_mpo.data[j-1], self.data[j-1], R, V)

            if normalize:
                nm0 = tc.linalg.norm(R)
                R = R / nm0
                nm *= nm0

            if updateS:
                self.Ss[j] = S
            
        self.data[0] = R.reshape(1, *self.data[0].shape[1:-1], linkdim)
        self.llim = self.rlim = 0
        self.lognm = self.lognm + Ws_mpo.lognm
        self.lognm += tc.log(nm)
        return trunc_err_sum
    
    def apply_mpo(self:T, *args, **kwargs) -> T:
        newmpo = self.copy()
        newmpo.apply_mpo_(*args, **kwargs)
        return newmpo
    
    def _dm_get_Lenvs(self, W, ψ, n, dtype):
        Lenvs = []
        Lenv = tc.tensor(1., dtype=dtype, device=self.device).reshape(1,1,1,1)
        for j in range(n - 1):
            Lenv = self._dm_left2right(Lenv, W[j], ψ[j])
            # print(Lenv.shape)
            Lenv = Lenv/tc.norm(Lenv)
            Lenvs.append(Lenv)
        return Lenvs

    def _dm_get_R(self, W, ψ, R, V):
        return {MPS: tf._dm_get_R_mps, MPO: tf._dm_get_R_mpo}[type(self)](W, ψ, R, V)

    def _dm_left2right(self, Lenv, W, ψ):
        return {MPS: tf._dm_left2right_mps, MPO: tf._dm_left2right_mpo}[type(self)](Lenv, W, ψ)

    def _resume_canonical(self, W, W2):
        return {MPS: tf._resume_canonical_mps, MPO: tf._resume_canonical_mpo}[type(self)](W, W2)
    
    def apply_mpo_naive_(
                self,
                Ws_mpo: 'MPO',
            ):
        """局部直接收缩
        
        bond 将指数增加
        """
        for i in range(len(self.data)):
            self.data[i] = self._apply_mpo_step(Ws_mpo.data[i], self.data[i])

    def _apply_mpo_step(self, W, ψ):
        return {MPS: tf._apply_on_mps_step, MPO: tf._apply_on_mpo_step}[type(self)](W, ψ)

    def apply_submpo_(self, 
                      Ws_mpo: 'MPO',
                      start_pos: int,
                      trunc_para: tuple[int, float, float] = (None, None, None),
                      updateS = True,
                      normalize = False):
        assert start_pos + Ws_mpo.L <= self.L, "超出范围"
        self.move_llim_(start_pos)
        self.move_rlim_(start_pos + Ws_mpo.L - 1)
        subtt = self._partition(start_pos, start_pos + Ws_mpo.L - 1)
        subtt.apply_mpo_naive_(Ws_mpo)
        self._put_back_(subtt, start_pos)
    
    def _partition(self, startpos, endpos):
        newtt, newSs = [], []
        for i in range(startpos, endpos+1):
            newtt.append(self.data[i].clone())
            newSs.append(self.Ss[i].clone() if self.Ss[i] is not None else None)
        
        subL = endpos - startpos + 1
        if self.llim < startpos:
            warnings.warn("llim is out of range")
            newllim = -1
        elif self.llim > endpos:
            warnings.warn("llim is out of range")
            newllim = subL
        else:
            newllim = self.llim - startpos
        
        if self.rlim < startpos:
            warnings.warn("rlim is out of range")
            newrlim = -1
        elif self.rlim > endpos:
            warnings.warn("rlim is out of range")
            newrlim = subL
        else:
            newrlim = self.rlim - startpos
        
        return type(self)(Ws=newtt, Ss=newSs, llim=newllim, rlim=newrlim)
    
    def _put_back_(self, subtt, startpos):
        for i, W in enumerate(subtt.data):
            self.data[startpos+i] = W
        for i, S in enumerate(subtt.Ss):
            self.Ss[startpos+i] = S
        self.llim = subtt.llim + startpos
        self.rlim = subtt.rlim + startpos
        self.lognm += subtt.lognm


class MPS(TensorTrain):
    def __init__(
        self,
        Ws: list[tc.Tensor],
        Ss: list[tc.Tensor] = None,
        llim: int = None,
        rlim: int = None,
        lognm: float = None,
        L: int = None,
    ):
        super().__init__(Ws, Ss, llim, rlim, lognm, L=L)

    def to_quimb(self) -> 'MatrixProductState':
        import quimb.tensor as qtn
        if self.data[0].shape[0] == 1:
            res = []
            a,b,d = self.data[0].shape
            res.append(self.data[0].cpu().numpy().reshape(b,d).transpose([1,0]))
            for i in range(1,len(self.data)-1):
                res.append(self.data[i].cpu().numpy().swapaxes(1,2))
            a,b,d = self.data[-1].shape
            res.append(self.data[-1].cpu().numpy().reshape(a,b))
            return np.exp(self.lognm).item() * qtn.MatrixProductState(res)
        # 周期 MPO
        res = [self.data[i].cpu().numpy().swapaxes(1,2) for i in range(len(self.data))]
        return np.exp(self.lognm).item() * qtn.MatrixProductState(res)

    def to_itensor(self) -> None:
        from ...basicfun import save_hdf5
        assert self.data[0].shape[0] == 1, "只能处理 OBC"
        data_dict = {}
        data_dict["iscomplex"] = 1 if self.dtype.is_complex else 0
        tsr = data_dict["tensors"] = {}
        for i in range(len(self.data)):
            if i == 0:
                a,b,d = self.data[i].shape
                tsr[f"W{i+1}"] = self.data[i].cpu().numpy().reshape(b,d)
            elif i == len(self.data)-1:
                a,b,d = self.data[i].shape
                tsr[f"W{i+1}"] = self.data[i].cpu().numpy().reshape(a,b)
            else:
                tsr[f"W{i+1}"] = self.data[i].cpu().numpy()
        data_dict["lognm"] = self.lognm.cpu().numpy()
        data_dict["L"] = self.L
        data_dict["linkdim"] = [self.data[i].shape[0] for i in range(1, len(self.data))]
        data_dict["code"] = """    sites, psi = 
    jldopen("mpsdata.h5", "r") do file
        L, linkdimlist = file["L"], file["linkdim"]
        sites = siteinds("S=1/2",L)
        v = Vector{ITensor}(undef, L)
        l = [Index(linkdimlist[ii], "Link,l=$ii") for ii in 1:(L - 1)]
        iscomplex = file["iscomplex"]
        for ii in eachindex(sites)
            s = sites[ii]
            if ii == 1
                tmp = iscomplex==1 ? reshape([Complex(x.r, x.i) for x in file["tensors/W1"]], size(file["tensors/W1"])) : file["tensors/W1"]
                v[ii] = ITensor(tmp, l[ii], s)
            elseif ii == L
                tmp = iscomplex==1 ? reshape([Complex(x.r, x.i) for x in file["tensors/W$L"]], size(file["tensors/W$L"])) : file["tensors/W$L"]
                v[ii] = ITensor(tmp, s, dag(l[ii - 1]))
            else
                tmp = iscomplex==1 ? reshape([Complex(x.r, x.i) for x in file["tensors/W$ii"]], size(file["tensors/W$ii"])) : file["tensors/W$ii"]
                v[ii] = ITensor(tmp, l[ii], s, dag(l[ii - 1]))
            end
        end   
        sites, exp(file["lognm"]) * MPS(v)
    end"""  # 在 julia 中运行这段还没就可以还原 mpo 了
        save_hdf5("mpsdata.h5", '/', data_dict)
    
    @classmethod
    def from_quimb(cls, mps: 'MatrixProductState', device='cpu') -> 'MPS':
        device = 'cpu'
        siteinds = mps.outer_inds()

        inds1, inds2 = mps[0].inds
        if inds1 in siteinds:
            siteind, linkind1 = inds1, inds2
        else:
            siteind, linkind1 = inds2, inds1
        tmp = mps[0].to_dense([siteind], [linkind1])

        res = [tc.tensor(tmp, device=device).reshape(1, *tmp.shape)]
        for j in range(1, mps.L-1):
            inds1 = mps[j].inds
            for i in inds1:
                if i in siteinds:
                    siteind = i
                elif i != linkind1:
                    linkind2 = i
            tmp = mps[j].to_dense([linkind1], [siteind], [linkind2])
            res.append(tc.tensor(tmp, device=device))
            linkind1 = linkind2

        inds1, inds2 = mps[-1].inds
        if inds1 in siteinds:
            siteind, linkind1 = inds1, inds2
        else:
            siteind, linkind1 = inds2, inds1
        tmp = mps[-1].to_dense([linkind1], [siteind])
        res.append(tc.tensor(tmp, device=device).reshape(*tmp.shape, 1))
        return MPS(res)
        
        
    @classmethod
    def from_random(cls, L:int, bond_dim:Union[list[int], int], phys_dim=2, dtype=tc.complex128, device=None) -> 'MPS':
        if isinstance(bond_dim, int):
            linkdims_ = [1] + [bond_dim] * (L - 1) + [1]
        else:
            assert len(bond_dim) == L + 1
            linkdims_ = bond_dim
        ψ1 = [tc.randn(linkdims_[i],phys_dim,linkdims_[i+1], dtype=dtype, device=device) for i in range(L)]
        return cls(ψ1)
    
    @classmethod
    def from_product_state(cls, state: list[str], dtype=tc.float64, device=None) -> 'MPS':
        Ws = [tc.zeros(1, 2, 1, dtype=dtype, device=device) for i in range(len(state))]
        for i, s in enumerate(state):
            if s == "up":
                Ws[i][0, 0, 0].add_(1.)
            elif s == "down":
                Ws[i][0, 1, 0].add_(1.)
            else:
                raise ValueError(f"state {s} is not defined")
        return cls(Ws)
    
    @classmethod
    def from_ghz_state(cls, L, dtype=tc.float64, device=None) -> 'MPS':
        if L == 1:
            return MPS([tc.tensor([[[1./np.sqrt(2)], [1./np.sqrt(2)]]], device=device)])
        tsr1 = tc.zeros(1, 2, 2, dtype=dtype, device=device)
        tsr1[0,1,1].add_(1.)
        tsr1[0,0,0].add_(1.)

        tsr2 = tc.zeros(2, 2, 2, dtype=dtype, device=device)
        tsr2[1,1,1].add_(1.)
        tsr2[0,0,0].add_(1.)

        tsr3 = tc.zeros(2, 2, 1, dtype=dtype, device=device)
        tsr3[1,1,0].add_(1.)
        tsr3[0,0,0].add_(1.)

        Ws = [tsr1] + [tsr2 for _ in range(L-2)] + [tsr3]
        return cls(Ws, lognm=-tc.log(tc.tensor(2., device=device))/2)

    @classmethod
    def from_w_state(cls, L, which='up', dtype=tc.float64, device=None) -> 'MPS':
        if L == 1:
            return MPS([tc.tensor([[[1./np.sqrt(2)], [1./np.sqrt(2)]]], device=device)])
        i = 0 if which == 'up' else 1
        tsr1 = tc.zeros(1, 2, 2, dtype=dtype, device=device)
        tsr1[0,1-i,0].add_(1.)
        tsr1[0,i,1].add_(1.)

        tsr2 = tc.zeros(2, 2, 2, dtype=dtype, device=device)
        tsr2[1,1-i,1].add_(1.)
        tsr2[0,i,1].add_(1.)
        tsr2[0,1-i,0].add_(1.)

        tsr3 = tc.zeros(2, 2, 1, dtype=dtype, device=device)
        tsr3[1,1-i,0].add_(1.)
        tsr3[0,i,0].add_(1.)

        Ws = [tsr1] + [tsr2 for _ in range(L-2)] + [tsr3]
        return cls(Ws, lognm=-tc.log(tc.tensor(L, device=device))/2)
    
    def _get_str(self,full=False):
        out1 = self.__class__.__name__ +";  " + str(self.data[0].dtype) + ";  " + f"norm: {self.norm():.3e}" + ";  " + f"maxbonddim: {self.maxbonddim()}" + ";  " + f"device: {self.device.type}"  + ";\n"
        L = len(self.data)
        if L < 15:
            full = True
        out2 = "physdim: "
        out3 = "         --"
        out4 = "bonddim: "
        out5 = "site:     "
        llim = self.llim if self.llim is not None else -1
        rlim = self.rlim if self.rlim is not None else -1
        ldis = 0
        rdis = llim if llim > -1 else rlim if rlim > -1 else L
        tag = False
        for i in range(L):
            if full or ldis < 2 or rdis <= 2:
                a,b,c = self.data[i].shape
                ldis += 1
                rdis -= 1
                if i < llim:
                    out3 += "--▷---"
                    ldis = 1 if rdis <= 0 else ldis
                    rdis = (llim-i-1) if rdis <= 0 else rdis
                elif i > rlim:
                    out3 += "--◁---"
                    ldis = 1 if rdis <= 0 else ldis
                    rdis = (L-i-1) if rdis <= 0 else rdis
                else:
                    out3 += "--◻---"
                    ldis = 1 if rdis <= 0 else ldis
                    rdis = (rlim-i) if rdis <= 0 else rdis
                out2 += f"{b:>4}| "
                out4 += f"{a:^5} " if tag else f"{a:^4}  "
                out5 += f"  {i:^4}"
                tag = False
            elif ldis == 2:
                a,b,c = self.data[i].shape
                ldis += 1
                rdis -= 1
                if i < llim:
                    out3 += " ... -"
                elif i > rlim:
                    out3 += " ... -"
                else:
                    out3 += " ... -"
                out2 += f"   ..."
                out4 = out4[:-1] + f"{a:^4}..."
                out5 += f"  ... "
                tag = True
            else:
                ldis += 1
                rdis -= 1
                
        out4 += f" {c}"
        out2 += "\n"
        out3 += "-\n"
        out4 += "\n"
        out = out1 + out2 + out3 + out4 + out5
        return out
    
    def show(self, full=False):
        print(self._get_str(full=full))
    
    def __repr__(self) -> str:
        return self._get_str()
    
    def measure(self, operator:Union[tc.Tensor, str, list[str], list[tc.Tensor]], pos:Union[int, list[int, int], None] = None, pauli=False, logscale=False) -> tc.Tensor:
        """
        局域算符的观测值：
        
        .. code-block:: text
        
            -----▷------▷------◻------⨞------⨞------⨞----- ψ.conj()
                 |      |      |      |      |      |
                 |      |      ◻      |      |      |
                 |      |      |      |      |      |
            -----▷------▷------◻------⨞------⨞------⨞----- ψ
                               ↑
                              pos
        
        移动正交中心到 pos 位置，然后将 operator 作用在 pos 位置上
        
        如果不是局域的测量，使用单体门作用后 inner 的方法计算
        
        pauli 只当 operator 是 SpinOper 时生效，表示 operator 是 Pauli 矩阵

        #todo 使用局部 MPO 的方法来计算非最近邻的观测值
        
        Examples
        --------
        >>> 𝜓.measure('z', 0)
        >>> 𝜓.measure('xx', 0)
        >>> 𝜓.measure('xix', 0)
        >>> 𝜓.measure('xx+yy', 0)
        >>> 𝜓.measure('xx+yy', 0)
        >>> 𝜓.measure(qt.generate.pauli_matrix('xx'), 0)
        >>> 𝜓.measure(np.random.randn(4,4), 0)
        >>> 𝜓.measure(op.xx(0,1) + op.yy(0,1))
        >>> 𝜓.measure(op.heisenberg_operator(L))
        >>> 𝜓.measure(qtc.MPO.from_random(L=2, bond_dim=2, dtype=tc.float64), 0)
        """
        # -------- 单体门观测 --------
        if isinstance(operator, list):
            assert len(operator) == len(pos), f'长度需要一致, operator = {operator}, pos = {pos}'
            assert len(pos) == len(set(pos)), f'位置必须唯一, pos = {pos}'
            argpos = np.argsort(pos)
            newpos, newlocalmat = [], []
            for i in argpos:
                p = pos[i]
                newpos.append(p)
                o = operator[i]
                if isinstance(o, str):
                    assert len(o) == 1, f'str 形式只支持单体门， o = {o}'
                    local_mat = tc.tensor(pauli_matrix(o), device=self.device)
                elif isinstance(o, np.ndarray):
                    local_mat = tc.tensor(o, device=self.device)
                else:
                    local_mat = o
                assert isinstance(local_mat, tc.Tensor), f'operator 必须是 Tensor 或 str, type = {type(local_mat)}'
                assert local_mat.shape[0] == self.data[pos[i]].shape[1], 'list 形式只支持单体门'
                newlocalmat.append(local_mat)
            
            firstpos, lastpos = newpos[0], newpos[-1]
            if self.is_canonical_form():
                firstdata = self.Ss[firstpos].reshape(-1, 1, 1) * self.data[firstpos]
            else:
                self.move_llim_(firstpos)
                self.move_rlim_(newpos[-1])
                firstdata = self.data[firstpos]
            
            firstdata, mat = promote_dtype(firstdata, newlocalmat[0])
            Lenv = tf._ProjMPS_contract_left_env(firstdata, 
                        tf._local_apply(firstdata, mat), 
                        tc.eye(firstdata.shape[0], dtype=self.dtype, device=self.device))
            
            lognm = tc.tensor(0., dtype=self.dtype, device=self.device)
            ct = 1
            for i in range(firstpos+1, lastpos+1):
                if i in newpos:
                    Lenv, data, mat = promote_dtype(Lenv, self.data[i], newlocalmat[ct])
                    Lenv = tf._ProjMPS_contract_left_env(data,
                        tf._local_apply(data, mat), Lenv)
                    ct += 1
                else:
                    Lenv, data = promote_dtype(Lenv, self.data[i])
                    Lenv = tf._ProjMPS_contract_left_env(data, data, Lenv)
                Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)
            
            if logscale:
                return tc.log(Lenv.trace()) + self.lognm * 2
            return Lenv.trace() * tc.exp(self.lognm)**2
            
        if isinstance(operator, str):
            nop, npos = [], []
            for i, o in enumerate(operator):
                if o == 'I' or o == 'i':
                    continue
                if o not in ['x', 'y', 'z', 'X', 'Y', 'Z', 
                             'I', 'i', 'p', 'P', 'm', 'M']:
                    break
                nop.append(o)
                npos.append(i + pos)
            else:
                return self.measure(nop, npos, logscale=logscale)
            operator = pauli_matrix(operator)

        # -------- 局域门 --------
        if isinstance(operator, np.ndarray):
            operator = tc.tensor(operator, device=self.device)
            
        if isinstance(operator, tc.Tensor):
            minpos = pos
            dim = 1
            for maxpos in range(pos, self.L):
                dim *= self.data[maxpos].shape[1]
                if dim == operator.shape[0]:
                    break
            else:
                raise ValueError("operator shape is not match")
            
            if self.is_canonical_form():
                contracted_tsr = self.Ss[minpos].reshape(-1, 1, 1) * self.data[minpos]
            else:
                self.orthogonalize_(minpos)
                contracted_tsr = self.data[minpos]
            for i in range(minpos+1, maxpos+1):
                contracted_tsr = tf._full_contract_right_mps(contracted_tsr, self.data[i])
            res = contracted_tsr.conj().reshape(-1) @ tf._local_apply(contracted_tsr, operator).reshape(-1)
            if logscale:
                return self.lognm * 2 + tc.log(res)
            return tc.exp(self.lognm*2) * res
        
        # -------- 部分 MPO --------
        from ...generate.operas import SpinOper
        if isinstance(operator, SpinOper):
            assert pos is None, "pos must be None when operator is SpinOper"
            # 如果 operator 只包含一项
            if (len(operator.data) == 1 and 
                list(operator.data.values())[0][0].shape[0] == 1):
                nop, npos = [], []
                for i, j in zip(list(operator.data.keys())[0], 
                                list(operator.data.values())[0][0][0]):
                    if i == 'I':
                        continue
                    nop.append(i.upper() if pauli else i.lower())
                    npos.append(j)
                return self.measure(nop, npos, logscale=logscale)
            
            #!! 利用两体门，非最近邻的可以通过 swap 变换到最近邻，但是否更有效率？
            # assert hasattr(operator, "local"), "operator must have local method"
            # try:
            #     res = 0.
            #     for i in range(self.L - 1):
            #         mat, hasoper = operator.expandxy(pauli).local(i, L=self.L)
            #         if not hasoper:
            #             continue
            #         mat = totc(np.real_if_close(mat), device=self.device)
            #         res += self.measure(mat, [i,i+1])
            #     return res
            # except TypeError as e:
            #     raise "might contain unsupported gate"
            
            # 利用 MPO 方法
            pos, operator = operator._minimal_shift()
            operator = operator.to_mpo(pauli=pauli, backend='torch', device=self.device)
        
        if isinstance(operator, MPO):
            if self.is_canonical_form():
                firstdata = self.Ss[pos].reshape(-1, 1, 1) * self.data[pos]
            else:
                self.move_llim_(pos)
                self.move_rlim_(pos + operator.L - 1)
                firstdata = self.data[pos]
            Lenv = tf._mele_init_left_env(operator.data[0], firstdata.conj(), firstdata)
            lognm = tc.tensor(0., dtype=self.dtype, device=self.device)
            for i in range(1, operator.L):
                Lenv = tf._mele_contract_left_env(operator.data[i], self.data[pos+i].conj(), self.data[pos+i], Lenv)
                Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)
            a, b, c, d = Lenv.shape
            assert a==1 and c == 1, "should be 1"
            trLenv = Lenv.reshape(b,d).trace()
            if logscale:
                return tc.log(trLenv) + self.lognm * 2 + operator.lognm
            return trLenv * tc.exp(self.lognm)**2 * tc.exp(operator.lognm)

        raise ValueError(f"operator type {type(operator)} is not supported")
 
    
    def _apply_1b_gate(self, pos, gate_1b):
        gate_1b = self._convert_gate(gate_1b, 1)
        return tf._local_apply(self.data[pos], gate_1b)

    def _apply_2b_gate(self, pos, gate_2b):
        gate_2b = self._convert_gate(gate_2b, 2)
        next_pos = pos + 1 if self.L != tc.inf else (pos + 1) % len(self.data)
        W1, W2 = self.data[pos], self.data[next_pos]
        return tf._apply_2b_gate_mps(W1, W2, gate_2b)


class MPO(TensorTrain):
    def __init__(
        self,
        Ws: list[tc.Tensor],
        Ss: list[tc.Tensor] = None,
        llim: int = None,
        rlim: int = None,
        lognm: float = None,
        L: int = None,
    ):
        super().__init__(Ws, Ss, llim, rlim, lognm, L=L)

    def to_quimb(self) -> 'MatrixProductOperator':
        import quimb.tensor as qtn
        if self.data[0].shape[0] == 1:
            res = []
            a,b,c,d = self.data[0].shape
            res.append(self.data[0].cpu().numpy().reshape(b,c,d).transpose([2,0,1]))
            for i in range(1,len(self.data)-1):
                res.append(self.data[i].cpu().numpy().transpose([0,3,1,2]))
            a,b,c,d = self.data[-1].shape
            res.append(self.data[-1].cpu().numpy().reshape(a,b,c))
            return np.exp(self.lognm).item() * qtn.MatrixProductOperator(res)
        # 周期 MPO
        res = [self.data[i].cpu().numpy().transpose([0,3,1,2]) for i in range(len(self.data))]
        return np.exp(self.lognm).item() * qtn.MatrixProductOperator(res)

    def to_itensor(self) -> None:
        from ...basicfun import save_hdf5
        assert self.data[0].shape[0] == 1, "只能处理 OBC"
        data_dict = {}
        data_dict["iscomplex"] = 1 if self.dtype.is_complex else 0
        tsr = data_dict["tensors"] = {}
        for i in range(len(self.data)):
            if i == 0:
                a,b,c,d = self.data[i].shape
                tsr[f"W{i+1}"] = self.data[i].cpu().numpy().reshape(b,c,d)
            elif i == len(self.data)-1:
                a,b,c,d = self.data[i].shape
                tsr[f"W{i+1}"] = self.data[i].cpu().numpy().reshape(a,b,c)
            else:
                tsr[f"W{i+1}"] = self.data[i].cpu().numpy()
        data_dict["lognm"] = self.lognm.cpu().numpy()
        data_dict["L"] = self.L
        data_dict["linkdim"] = [self.data[i].shape[0] for i in range(1, len(self.data))]
        data_dict["code"] = """    sites, Hs = 
    jldopen("mpodata.h5", "r") do file
        L, linkdimlist = file["L"], file["linkdim"]
        sites = siteinds("S=1/2",L)
        v = Vector{ITensor}(undef, L)
        l = [Index(linkdimlist[ii], "Link,l=$ii") for ii in 1:(L - 1)]
        iscomplex = file["iscomplex"]
        for ii in eachindex(sites)
            s = sites[ii]
            if ii == 1
                tmp = iscomplex==1 ? reshape([Complex(x.r, x.i) for x in file["tensors/W1"]], size(file["tensors/W1"])) : file["tensors/W1"]
                v[ii] = ITensor(tmp, l[ii], dag(s), s')
            elseif ii == L
                tmp = iscomplex==1 ? reshape([Complex(x.r, x.i) for x in file["tensors/W$L"]], size(file["tensors/W$L"])) : file["tensors/W$L"]
                v[ii] = ITensor(tmp, dag(s), s', dag(l[ii - 1]))
            else
                tmp = iscomplex==1 ? reshape([Complex(x.r, x.i) for x in file["tensors/W$ii"]], size(file["tensors/W$ii"])) : file["tensors/W$ii"]
                v[ii] = ITensor(tmp, l[ii], dag(s), s', dag(l[ii - 1]))
            end
        end   
        sites, exp(file["lognm"]) * MPO(v)
    end"""  # 在 julia 中运行这段还没就可以还原 mpo 了
        save_hdf5("mpodata.h5", '/', data_dict)

    @classmethod
    def from_quimb(cls, mpo: 'MatrixProductOperator', device='cpu', upper='k', lower='b') -> 'MPO':
        linkinds = [i for i in mpo[0].inds if (
            not i.startswith(lower) and not i.startswith(upper)
            )][0]
        tmp = mpo[0].to_dense([f'{upper}0'], [f'{lower}0'], [linkinds])
        res = [tc.tensor(tmp, device=device).reshape(1, *tmp.shape)]

        for j in range(1, mpo.L-1):
            linkinds_ = [i for i in mpo[j].inds if (
                not i.startswith(lower) and not i.startswith(upper)
                and i != linkinds
                )][0]
            tmp = mpo[j].to_dense([linkinds], [f'{upper}{j}'], [f'{lower}{j}'], [linkinds_])
            res.append(tc.tensor(tmp, device=device))
            linkinds = linkinds_

        tmp = mpo[-1].to_dense([linkinds], [f'{upper}{mpo.L-1}'], [f'{lower}{mpo.L-1}'])
        res.append(tc.tensor(tmp, device=device).reshape(*tmp.shape, 1))
        return MPO(res)

    @classmethod
    def from_random(cls, L:int, bond_dim:Union[list[int], int], phys_dim:int=2, dtype=tc.complex128, device=None) -> 'MPO':
        if isinstance(bond_dim, int):
            linkdims_ = [1] + [bond_dim] * (L - 1) + [1]
        else:
            assert len(bond_dim) == L + 1
            linkdims_ = bond_dim
        Ws = [tc.randn(linkdims_[i], phys_dim, phys_dim, linkdims_[i+1], dtype=dtype, device=device) for i in range(L)]
        return cls(Ws)

    @classmethod
    def from_heisenberg(cls, L, j=1, h=0, cyclic=False, pauli=True, device=None) -> 'MPO':
        from ...generate.operas import heisenberg_operator
        ham = heisenberg_operator(L, j=j, h=h, cyclic=cyclic)
        return ham.to_mpo(pauli=pauli, backend='torch', device=device)

    @classmethod
    def from_eye(cls, L, phys_dim=2, dtype=tc.float64, device=None) -> 'MPO':
        eyempo = [None] * L
        if isinstance(phys_dim, int):
            phys_dim = [phys_dim] * L
        for i in range(L):
            dim = phys_dim[i]
            eyempo[i] = tc.eye(dim, dtype=dtype, device=device).reshape(1, dim, dim, 1)
        return cls(eyempo)

    @classmethod
    def from_oper(cls, ham, L, pauli=True, device=None) -> 'MPO':
        return ham.to_mpo(L, pauli=pauli, backend='torch', device=device)

    def _get_str(self, full=False):
        out1 = self.__class__.__name__ +";  " + str(self.data[0].dtype) + ";  " + f"norm: {self.norm():.3e}" + ";  " + f"maxbonddim: {self.maxbonddim()}" + ";  " + f"device: {self.device.type}"  + ";\n"
        L = len(self.data)
        if L < 15:
            full = True
        out2 = "physdim: "
        out6 = "physdim: "
        out3 = "         --"
        out4 = "bonddim: "
        out5 = "site:     "
        llim = self.llim if self.llim is not None else -1
        rlim = self.rlim if self.rlim is not None else -1
        ldis = 0
        rdis = llim if llim > -1 else rlim if rlim > -1 else L
        tag = False
        for i in range(L):
            if full or ldis < 2 or rdis <= 2:
                a,b,d,c = self.data[i].shape
                ldis += 1
                rdis -= 1
                if i < llim:
                    out3 += "--▷---"
                    ldis = 1 if rdis <= 0 else ldis
                    rdis = (llim-i-1) if rdis <= 0 else rdis
                elif i > rlim:
                    out3 += "--◁---"
                    ldis = 1 if rdis <= 0 else ldis
                    rdis = (L-i-1) if rdis <= 0 else rdis
                else:
                    out3 += "--◻---"
                    ldis = 1 if rdis <= 0 else ldis
                    rdis = (rlim-i-1) if rdis <= 0 else rdis
                out2 += f"{b:>4}| "
                out6 += f"{d:>4}| "
                out4 += f"{a:^5} " if tag else f"{a:^4}  "
                out5 += f"  {i:^4}"
                tag = False
            elif ldis == 2:
                a,b,d,c = self.data[i].shape
                ldis += 1
                rdis -= 1
                if i < llim:
                    out3 += " ... -"
                elif i > rlim:
                    out3 += " ... -"
                else:
                    out3 += " ... -"
                out2 += f"   ..."
                out6 += f"   ..."
                out4 = out4[:-1] + f"{a:^4}..."
                out5 += f"  ... "
                tag = True
            else:
                ldis += 1
                rdis -= 1
                
        out4 += f" {c}"
        out2 += "\n"
        out6 += "\n"
        out3 += "-\n"
        out4 += "\n"
        out = out1 + out2 + out3 + out6 + out4 + out5
        return out

    def show(self, full=False):
        print(self._get_str(full=full))

    def __repr__(self) -> str:
        return self._get_str()

    def _apply_1b_gate(self, pos, gate_1b):
        if isinstance(gate_1b, (tc.Tensor, np.ndarray)):
            gate_1b = self._convert_gate(gate_1b, 1)
            return tf._local_apply(self.data[pos], gate_1b)
        
        gate, top_or_bottom = gate_1b
        if top_or_bottom == "top":
            gate = self._convert_gate(gate, 1)
            return tf._local_apply(self.data[pos], gate)
        elif top_or_bottom == "bottom":
            gate = self._convert_gate(gate, 1)
            return tf._local_apply(self.data[pos].swapaxes(1,2), gate).swapaxes(1,2)
        elif top_or_bottom == "topbottom":
            try:
                gate0, gate1 = gate
                gate0 = self._convert_gate(gate0, 1)
                gate1 = self._convert_gate(gate1, 1)
                return tf._local_apply2(self.data[pos], gate0, gate1)
            except TypeError:
                gate = self._convert_gate(gate, 1)
                return tf._local_apply2(self.data[pos], gate, gate)

    def _apply_2b_gate(self, pos, gate_2b):
        next_pos = pos + 1 if self.L != tc.inf else (pos + 1) % len(self.data)
        
        if isinstance(gate_2b, (tc.Tensor, np.ndarray)):
            gate_2b = self._convert_gate(gate_2b, 2)
            return tf._apply_2b_gate_mpo_from_top(self.data[pos], self.data[next_pos], gate_2b)
        
        gate, top_or_bottom = gate_2b
        if top_or_bottom == "top":
            gate = self._convert_gate(gate, 2)
            return tf._apply_2b_gate_mpo_from_top(self.data[pos], self.data[next_pos], gate)
        elif top_or_bottom == "bottom":
            gate = self._convert_gate(gate, 2)
            return tf._apply_2b_gate_mpo_from_bottom(self.data[pos], self.data[next_pos], gate)
        elif top_or_bottom == "topbottom":
            try:
                gate0, gate1 = gate
                gate0 = self._convert_gate(gate0, 2)
                gate1 = self._convert_gate(gate1, 2)
                return tf._apply_2b_gate_mpo_from_topbottom(self.data[pos], self.data[next_pos], gate0, gate1)
            except TypeError:
                gate = self._convert_gate(gate, 2)
                return tf._apply_2b_gate_mpo_from_topbottom(self.data[pos], self.data[next_pos], gate, gate)

    def mele(self, y, x, logscale=False):
        """
        Compute ⟨y|A|x⟩ = ⟨y|Ax⟩
        """
        Lenv = tf._mele_init_left_env(self.data[0], y.data[0].conj(), x.data[0])
        # Lenv = tc.tensor(1., dtype=self.dtype, device=self.device).reshape(1,1,1,1)
        lognm = tc.tensor(0., dtype=self.dtype, device=self.device)
        for i in range(1, self.L):
            Lenv = tf._mele_contract_left_env(self.data[i], y.data[i].conj(), x.data[i], Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)
        a, *_ = Lenv.shape
        if logscale:
            return tc.log(Lenv.reshape(a,a).trace()) + self.lognm + x.lognm + y.lognm
        return Lenv.reshape(a,a).trace() * tc.exp(self.lognm) * tc.exp(x.lognm) * tc.exp(y.lognm)

    def diag_inner(self, mps):
        return tf.diagonal_inner(self.data, mps.data)
    
    def exp(self, x, order=2, trunc_para:tuple[int, float, float]=(None,None,None), dtype=tc.complex128):
        """
        - trunc_para 是一个包含三个数的元组，分别表示:
            - chi_max: int, 截断的最大值
            - svd_min: float, SVD 的最小值
            - trunc_cut: float, 截断的阈值
        """
        l = len(self.data)
        local_dims = [i.shape[1] for i in self.data]
        res = MPO(tf.mpo_eye(l, local_dims, dtype))
        for i in range(1,order+1):
            tmp = self.copy()
            for _ in range(i-1):
                tmp.apply_mpo_(self, trunc_para=trunc_para)
            res += x**i/math_lib.factorial(i) * tmp
            res.canonicalize_(trunc_para)
        return res
    
    def trace(self):
        data0 = self.data[0]
        Lenv = tc.tensor(1., dtype=data0.dtype, device=data0.device).reshape(data0.shape[0], data0.shape[0])
        for tsr in self.data:
            Lenv = Lenv @ tf._up_bottom_tr(tsr)
        return tc.trace(Lenv) * tc.exp(self.lognm)

    def dmrg(self, psi0=None, **kwargs) -> tuple[float, MPS]:
        r"""DMRG 方法求解 MPO 的基态
        
        Returns
        -------
        energy : float
            基态能量
        psi : MPS
            基态
        
        todo: eig pertube mixer 
        todo: dmrg 求第一激发态
        
        Examples
        --------
        >>> import quante as qt
        >>> L = 4
        >>> ham = qt.generate.operas.heisenberg_operator(L=L)
        >>> mpo = ham.to_mpo(L=L)
        >>> eng, psi = mpo.dmrg()
        >>> psi.lognm *= 0

        激发态 DMRG
        >>> eng, psi = mpo.dmrg(Ms=[psi])
        >>> eng
        
        对比
        >>> ham.gdenergy(k=2)
        """
        from .proj_algrithms import DMRG
        return DMRG(self, psi0=psi0, **kwargs).run2()

    def tdvp(self, init: MPS, final_time: Number, time_step: Number, **kwargs) -> Generator[tuple[Union[float, complex], MPS], None, None]:
        r"""利用 tdvp 方法求解时间演化

        计算 `exp( time_step * H ) | init >`
        
        Parameters
        ----------
        init : MPS
            初始态
        final_time : Number
            总时间
        time_step : Number
            时间步长
               
        Returns
        -------
        phis : list[tuple[float, MPS]]
            时间演化的态
        
        Notes
        -----
        - `svd` 的速度会比 `eig` 快，但是 `eig` 会更稳定
        - `chi_max` 要从小增加，否则会出现 nan
        
        Examples
        --------
        >>> L = 100
        >>> ham = qt.generate.operas.heisenberg_operator(L, j=(1, 1, 1))
        >>> ham = ham.expandxy(pauli=False)
        >>> H = tn.MPO(qtc.totc(ham.automata(L, pauli=False)))
        >>> psi0 = tn.MPS.product_state(["up","down"]*(L//2), dtype=tc.complex128)
        >>> psis = H.tdvp(init=psi0, time_step=0.1j, final_time=1.0j)
        >>> Ss = []
        >>> for t, phi in psis:
        >>>     Ss.append(phi.entanglement_entropy(bonds=[L//2])[0].item())
        >>> print(Ss)
        [0.017425858509214333, 0.05540147407158314, 0.10518787487322075, 0.16171880185660767, 0.22146216557201934, 0.2819921651885448, 0.34180493542491497, 0.40016281239278695, 0.4569178458476886, 0.5123157152579267]
        
        References
        ----------
        https://arxiv.org/abs/1408.5056
        """
        from .proj_algrithms import TDVP
        return TDVP(mpo=self, psi0=init, time_step=time_step, final_time=final_time, **kwargs).run()
    
