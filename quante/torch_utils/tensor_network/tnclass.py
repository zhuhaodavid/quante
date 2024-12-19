# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-07-10 21:48:14
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-14 02:46:18
# @Description:
#   目的：为了方便使用 torch 编写（带梯度的）张量网络程序，将关于 MPS/MPO 的功能集中到一个类中
#   特点：
#     - 此文件只调用 ./tensor/tcfuncs.py，以及 numpy 和 torch 中的函数，不依赖不调用 ./tensor 中任何其他的文件
#     - 这个文件中的函数可以被 ./tensor/tcfuncs.py 之外的其他文件调用。
#     - 此文件中的所有函数都应保证梯度链。
#     - 所有改变类本身的方法都会加上下划线，如 `orthogonalize_`，`canonicalize_`，`apply_gate_2b_` 等

import torch as tc
from typing import Union, TypeVar, Optional

from ..linalg.decomp import eigh, qr, rq, svd, truncate

from . import tnfuncs as tf
from ..linalg.krylov import lanczos_ground_state, lanczos_evolve_state
from ..utils import clone
from ...generate.matrix import pauli_matrix
from ...linalg.svd_robust import TruncationError
from ...linalg.krylov import lanczos_arpack
from ...linalg.evolve import expm_multiply

import math as math_lib # type: ignore
import copy
import numpy as np
import time
from numbers import Number

__all__ = [
    "MPS",
    "MPO",
    "ProjMPO",
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
        
        self.length = len(Ws) if L is None else L
        assert self.length == len(Ws) or self.length == tc.inf
        
        if self.length == tc.inf:
            self.llim, self.rlim = None, None
        else:
            self.llim = llim if llim is not None else 0
            self.rlim = rlim if rlim is not None else len(Ws) - 1
            
        self.Ss = Ss if Ss is not None else [None] * (len(Ws) + 1)
        self.dtype = Ws[0].dtype
        self.device = Ws[0].device
        self.lognm = lognm if lognm is not None else tc.tensor(0.0, dtype=tc.float64,device=self.device)
    
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

    def inner(self, anotherMPS, logscale=False, conj=False):
        if logscale:
            return tf.tn_inner(self.data, anotherMPS.data, logscale=True, conj_at_1=conj) + self.lognm + anotherMPS.lognm
        else:
            return tf.tn_inner(self.data, anotherMPS.data, logscale=False, conj_at_1=conj) * tc.exp(self.lognm) * tc.exp(anotherMPS.lognm)

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
            return tf.tn_norm(self.data, lognorm=lognorm) * tc.exp(self.lognm)
    
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
            except:
                raise ValueError("failed to reshape gate")
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
        if normalize:
            nm = tc.norm(phi)
            phi = phi / nm
            self.lognm += tc.log(nm)
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
                if normalize:
                    nm = tc.norm(A)
                    A = A / nm
                    self.lognm += tc.log(nm)
                self.data[pos] = U
                self.data[pos + 1] = A.reshape(-1,*Wshape[halfdim:])
                self.llim = self.rlim = pos + 1
            elif direction == "left":
                B, U = rq(W.reshape(-1,*Wshape[halfdim:]))
                if normalize:
                    nm = tc.norm(B)
                    B = B / nm
                    self.lognm += tc.log(nm)
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
            
            if normalize:
                nm = tc.norm(S)
                S = S / nm
                self.lognm += tc.log(nm)
                
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
                if normalize:
                    nm = tc.norm(self.data[pos + 1])
                    self.data[pos + 1] = self.data[pos + 1] / nm
                    self.lognm += tc.log(nm)
            elif direction == "left":
                self.llim = self.rlim = pos
                if normalize:
                    nm = tc.norm(self.data[pos])
                    self.data[pos] = self.data[pos] / nm
                    self.lognm += tc.log(nm)
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
        next_pos = pos + 1 if self.length != tc.inf else (pos + 1) % len(self.data)
        theta = self.Ss[pos].reshape(-1, *[1]*(W.ndim-1)) * W
        if svd_alg == 'eig':
            
            if self.length == tc.inf:
                raise ValueError("正则形式下不能使用 eig 方法，因为证明中用到了本征分解的正确性，如果有裁剪，会破坏左正交的正交性质，并且在无穷长链中，这个破坏会逐步积累。")
            
            W1, S, W2, err, direction = eigh(theta, trunc_para=trunc_para)
            if direction == 'right':
                _, W2 = rq(W2)
            # W1, S, W2, err = svd(theta, trunc_para=trunc_para)
            W1 = self._resume_canonical(W, W2)
            # print(pos, tc.dist(W1.reshape(W1.shape[0],-1) @ W1.reshape(W1.shape[0],-1).H, tc.eye(W1.shape[0])))
            # print((W1.reshape(W1.shape[0],-1) @ W1.reshape(W1.shape[0],-1).H).numpy())
            if normalize:
                nm = tc.norm(W1)/W1.shape[0]**0.5
                W1 = W1 / nm
                self.lognm += tc.log(nm)
            self.data[pos] = W1
            self.Ss[next_pos], self.data[next_pos] = S/tc.norm(S) if normalize else S, W2
            
        elif svd_alg == 'svd':
            W1, S, W2, err = svd(theta, trunc_para=trunc_para)
            if normalize:
                nm = tc.norm(S)
                S = S / nm
                self.lognm += tc.log(nm)
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
                dtype = tc.complex128,
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
    
    def _dm_get_Lenvs(self, W, ψ, n, dtype):
        Lenvs = []
        Lenv = tc.tensor(1., dtype=dtype, device=self.device).reshape(1,1,1,1)
        for j in range(n - 1):
            Lenv = self._dm_left2right(Lenv, W[j], ψ[j])
            Lenv = Lenv/tc.norm(Lenv)
            Lenvs.append(Lenv)
        return Lenvs
    
    def apply_mpo_naive_(
                self,
                Ws_mpo: 'MPO',
            ):
        """局部直接收缩
        
        bond 将指数增加
        """
        for i in range(len(self.data)):
            self.data[i] = self._apply_mpo_step(Ws_mpo.data[i], self.data[i])


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
    
    @classmethod
    def random(cls, N:int, linkdims:Union[list[int], int], dtype=tc.complex128, device=None):
        if isinstance(linkdims, int):
            linkdims_ = [1] + [linkdims] * (N - 1) + [1]
        else:
            assert len(linkdims) == N + 1
            linkdims_ = linkdims
        ψ1 = [tc.randn(linkdims_[i],2,linkdims_[i+1], dtype=dtype, device=device) for i in range(N)]
        return cls(ψ1)
    
    @classmethod
    def product_state(cls, state: list[str], dtype=tc.float64, device=None):
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
    def ghz_state(cls, L, dtype=tc.float64, device=None):
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
    def w_state(cls, L, which='up', dtype=tc.float64, device=None):
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
    
    def measure(self, operator:Union[tc.Tensor, str], pos:Union[int, list[int]]):
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
        
        Examples
        --------
        >>> vec.measure("z", i)
        >>> vec.measure("zz", [i,i+1])
        """
        if isinstance(operator, str):
            operator = tc.tensor(pauli_matrix(operator), device=self.device)
        
        try:
            minpos, maxpos = pos
        except TypeError:
            minpos = maxpos = pos
        dim = 1
        for i in range(minpos, maxpos+1):
            dim *= self.data[i].shape[1]
        assert operator.shape[0] == operator.shape[1] == dim, "operator shape is not match"
        
        if self.is_canonical_form():
            contracted_tsr = self.Ss[minpos].reshape(-1, *([1]*(self.data[0].ndim-1))) * self.data[minpos]
        else:
            self.orthogonalize_(minpos)
            contracted_tsr = self.data[minpos]
            
        for i in range(minpos+1, maxpos+1):
            contracted_tsr = tf._full_contract_right_mps(contracted_tsr, self.data[i])
        
        dtype = tc.complex128 if contracted_tsr.dtype.is_complex or operator.dtype.is_complex else tc.float64
        contracted_tsr = contracted_tsr.to(dtype=dtype,device=self.device)
        operator = operator.to(dtype=dtype,device=self.device)
        
        res = contracted_tsr.conj().reshape(-1) @ tf._local_apply(contracted_tsr, operator).reshape(-1)
        return tc.exp(self.lognm*2) * res
    
    def _apply_1b_gate(self, pos, gate_1b):
        gate_1b = self._convert_gate(gate_1b, 1)
        return tf._local_apply(self.data[pos], gate_1b)

    def _apply_2b_gate(self, pos, gate_2b):
        gate_2b = self._convert_gate(gate_2b, 2)
        next_pos = pos + 1 if self.length != tc.inf else (pos + 1) % len(self.data)
        W1, W2 = self.data[pos], self.data[next_pos]
        return tf._apply_2b_gate_mps(W1, W2, gate_2b)
    
    def _dm_left2right(self, Lenv, B, A):
        return tf._dm_left2right_mps(Lenv, B, A)
        
    def _dm_get_R(self, W, ψ, R, V):
        return tf._dm_get_R_mps(W, ψ, R, V)

    def _apply_mpo_step(self, W, ψ):
        return tf._apply_on_mps_step(W, ψ)

    def _resume_canonical(self, W, W2):
        return tf._resume_canonical_mps(W, W2)

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
    
    @classmethod
    def random(cls, N:int, linkdims:Union[list[int], int], phydim:int=2, dtype=tc.complex128, device=None):
        if isinstance(linkdims, int):
            linkdims_ = [1] + [linkdims] * (N - 1) + [1]
        else:
            assert len(linkdims) == N + 1
            linkdims_ = linkdims
        Ws = [tc.randn(linkdims_[i], phydim, phydim, linkdims_[i+1], dtype=dtype, device=device) for i in range(N)]
        return cls(Ws)
    
    @classmethod
    def heisenberg(cls, L, j=1, h=0, cyclic=False, pauli=True, device=None):
        from ...generate.operas import heisenberg_operator
        ham = heisenberg_operator(L, j=j, h=h, cyclic=cyclic)
        npmpo = ham.automata(L=L, pauli=pauli)
        return cls([tc.tensor(i,device=device) for i in npmpo])
    
    @classmethod
    def eye(cls, L, local_dims=2, dtype=tc.float64, device=None):
        eyempo = [None] * L
        if isinstance(local_dims, int):
            local_dims = [local_dims] * L
        for i in range(L):
            dim = local_dims[i]
            eyempo[i] = tc.eye(dim, dtype=dtype, device=device).reshape(1, dim, dim, 1)
        return cls(eyempo)
    
    def from_oper(cls, ham, L, pauli=True, device=None):
        npmpo = ham.automata(L=L, pauli=pauli)
        return cls([tc.tensor(i,device=device) for i in npmpo])
    
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
        next_pos = pos + 1 if self.length != tc.inf else (pos + 1) % len(self.data)
        
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

    
    def diag_inner(self, mps):
        return tf.diagonal_inner(self.data, mps.data)
    
    def _dm_left2right(self, Lenv, B, A):
        return tf._dm_left2right_mpo(Lenv, B, A)
        
    def _dm_get_R(self, W, ψ, R, V):
        return tf._dm_get_R_mpo(W, ψ, R, V)
    
    def _apply_mpo_step(self, W, ψ):
        return tf._apply_on_mpo_step(W, ψ)
    
    def _resume_canonical(self, W, W2):
        return tf._resume_canonical_mpo(W, W2)
    
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

    def dmrg(self, 
            psi0:MPS, # 初始态
            nsweep:int, # 扫多少次
            chi_max:list[int],  # MPS 中最大的 bond 维度
            cutoff:list[float],  # MPS 中奇异谱最小值
            backend:str='default',  # 计算有效哈密顿量基态的方法，"default", "larpack", "lanczos"
            svd_alg:str='svd',  # update MPS 中张量的方法
            nsite=2,  # 几格点 dmrg, 1 或 2
            outputlevel=1, # 输出等级，0 表示不输出中间信息
            ):
        r"""DMRG 方法求解 MPO 的基态
        
        Parameters
        ----------
        psi0 : MPS
            初始态
        nsweep : int
            扫多少次
        chi_max : list[int]
            MPS 中最大的 bond 维度
        cutoff : list[float]
            MPS 中奇异谱最小值
        backend : str, optional
            计算有效哈密顿量基态的方法，"default", "larpack", "lanczos", by default 'default'
        svd_alg : str, optional
            update MPS 中张量的方法, by default'svd'
        nsite : int, optional
            几格点 dmrg, 1 或 2, by default 2
        outputlevel : int, optional
            输出等级，0 表示不输出中间信息, by default 1
        
        Returns
        -------
        energy : float
            基态能量
        psi : MPS
            基态
        
        Notes
        -----
        - `svd` 的速度会比 `eig` 快，但是 `eig` 会更稳定
        - `chi_max` 要从小增加，否则会出现 nan
        
        todo: eig pertube mixer 
        todo: dmrg 求第一激发态
        
        Examples
        --------
        >>> from quante.torch_utils import tn
        >>> L = 100
        >>> ham = qt.generate.operas.heisenberg_operator(L, j=(1, 1, 1))
        >>> ham = ham.expandxy(pauli=False)
        >>> H = tn.MPO(ham.automata(L, pauli=False, dtype=np.float64)).to(device='cpu')
        >>> psi0 = tn.MPS.product_state(["up","down"]*(L//2), dtype=tc.float64)
        >>> nsweeps = 10
        >>> chi_max = [100] * nsweeps
        >>> cutoff = [1E-10] * nsweeps
        >>> eng, vec = H.dmrg(psi0, nsweeps, chi_max, cutoff)
        >>> print(eng)
        tensor(-44.1277, dtype=torch.float64)
        """
        # 参数检查
        if len(chi_max) > nsweep:
            chi_max = chi_max[:nsweep]
        elif len(chi_max) < nsweep:
            chi_max.extend([chi_max[-1]] * (nsweep - len(chi_max)))
        
        if len(cutoff) > nsweep:
            cutoff = cutoff[:nsweep]
        elif len(cutoff) < nsweep:
            cutoff.extend([cutoff[-1]] * (nsweep - len(cutoff)))
            
        N = len(psi0.data)
        assert N == len(self.data), 'MPS 和 MPO 的长度应该相等'
        if N == 1:
            raise Exception("长度 1 的 MPS 暂不支持，可以用 `numpy.linalg.eigh`, `scipy.eigsh` 等方法求解")
        
        # 初始设置
        psi = psi0.copy()
        psi.orthogonalize_(0)
        
        assert psi.llim == psi.rlim == 0 or psi.llim == psi.rlim == -1
        
        oper = self.copy()
        if psi.dtype.is_complex and not self.dtype.is_complex:
            oper.to(dtype=tc.complex128,device=self.device)
        
        projH = ProjMPO(oper, nsite=nsite)
        projH.set_position_(psi, 0)
        
        energy = 0.0
        max_trunc_err = 1.e-14  # lanczos 误差
        
        # sweep
        for sw in range(nsweep):
            
            trunc_err_list = []  # 记录误差
            sw_time_start = time.time()  # 记录时间
            
            # --------------main-------------
            for position, direction in projH._sweep_schedule():
                # print(position, N)
                energy, trunc_err = projH.dmrg_sweep_(
                    psi=psi,  # 当前态
                    position=position,  # 优化 postion 位置的张量
                    direction=direction,  # 方向 'left' 或 'right'
                    svd_alg=svd_alg,   # update MPS 中张量的方法
                    chi_max=chi_max[sw], # MPS 中最大的 bond 维度
                    svd_min=cutoff[sw], # MPS 中奇异谱最小值
                    backend=backend,  # 计算有效哈密顿量基态的方法，"default", "larpack", "lanczos"
                    max_trunc_err=max_trunc_err  # lanczos 误差
                    )
                
                trunc_err_list.append(trunc_err)
            # -------------end main-----------
            
            max_trunc_err = max(trunc_err_list)  # 最大误差
            sw_time = time.time() - sw_time_start  # 记录每步的时间
            
            if outputlevel >= 1:
                print(f"After sweep {sw}: energy={(energy * tc.exp(self.lognm)).item()} maxchi={psi.maxbonddim()} maxtruncerr={max_trunc_err:.2e} time={sw_time:.3f}", flush=True)
        
        return energy * tc.exp(self.lognm), psi


    def tdvp(
            self,
            init: MPS,  # 初始态
            t: Number,  # 总时间
            time_step: Number,  # 时间步长
            chi_max: list[int],  # MPS 中最大的 bond 维度
            trunc_cut: list[float],  # MPS 中奇异谱最小值
            *,
            backend='default',  # 计算有效哈密顿量基态的方法，"default", "expm_multiply", "lanczos"
            svd_alg='svd',  #  update MPS 中张量的方法
            normalize=True,  # 是否归一化
            reverse_step=True, # 时间演化必须是 True，虚时演化 False 回到 DMRG
            time_start=0.0, # 起始时间
            nsite=2,  # nsite = 1 不改变 bond dimension, nsite = 2 可以改变 bond dimension
            order=2,  # 时间演化的阶数
            outputlevel=1,  # 输出等级，0 表示不输出中间信息
            ):
        r"""利用 tdvp 方法求解时间演化
        
        Parameters
        ----------
        init : MPS
            初始态
        t : Number
            总时间
        time_step : Number
            时间步长
        chi_max : list[int]
            MPS 中最大的 bond 维度
        trunc_cut : list[float]
            MPS 中奇异谱最小值
        backend : str, optional
            计算有效哈密顿量基态的方法，"default", "expm_multiply", "lanczos", by default 'default'
        svd_alg : str, optional
            update MPS 中张量的方法, by default'svd'
        normalize : bool, optional
            是否归一化, by default True
        reverse_step : bool, optional
            时间演化必须是 True，虚时演化 False 回到 DMRG, by default True
        time_start : float, optional
            起始时间, by default 0.0
        nsite : int, optional
            nsite = 1 不改变 bond dimension, nsite = 2 可以改变 bond dimension, by default 2
        order : int, optional
            时间演化的阶数, by default 2
        outputlevel : int, optional
            输出等级，0 表示不输出中间信息, by default 1
        
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
        >>> H = qtc.tn.MPO(ham.automata(L, pauli=False, dtype=np.float64)).to(device='cpu')
        >>> psi0 = qtc.tn.MPS.product_state(["up","down"]*(L//2), dtype=tc.complex128)
        >>> nsweeps = 8
        >>> chi_max = [20]
        >>> cutoff = [1E-10]
        >>> phis = H.tdvp(init=psi0, t=1.0j, time_step=0.1j, chi_max=chi_max, trunc_cut=cutoff, nsite=2, order=1, outputlevel=1)
        >>> Ss = []
        >>> for t, phi in phis:
        >>>     Ss.append(phi.entanglement_entropy(bonds=[L//2])[0].item())
        >>> print(Ss)
        [0.017425858509214333, 0.05540147407158314, 0.10518787487322075, 0.16171880185660767, 0.22146216557201934, 0.2819921651885448, 0.34180493542491497, 0.40016281239278695, 0.4569178458476886, 0.5123157152579267]
        
        References
        ----------
        https://arxiv.org/abs/1408.5056
        """
        # 参数检查
        nsweeps = int(np.real(t/time_step))
        if np.abs(nsweeps*time_step - t) > 1e-10:
            raise ValueError(f"t / time_step = {t} / {time_step} = {t/time_step} 必须是整数")
        
        N = len(init.data)
        assert N == len(self.data), 'MPS 和 MPO 的长度应该相等'
        if N == 1:
            raise Exception("长度 1 的 MPS 暂不支持，可以用 `numpy.linalg.eigh`, `scipy.eigsh` 等方法求解")
        
        if len(chi_max) > nsweeps:
            chi_max = chi_max[:nsweeps]
        elif len(chi_max) < nsweeps:
            chi_max.extend([chi_max[-1]] * (nsweeps - len(chi_max)))
        
        if len(trunc_cut) > nsweeps:
            trunc_cut = trunc_cut[:nsweeps]
        elif len(trunc_cut) < nsweeps:
            trunc_cut.extend([trunc_cut[-1]] * (nsweeps - len(trunc_cut)))
        
        # 初态设置
        state = init.copy()
        current_time = time_start
        
        if np.iscomplex(time_step):
            if not state.dtype.is_complex:
                state.to(dtype = tc.complex128,device=self.device)
        
        # 算符设置
        oper = self.copy()
        reduced_operator = ProjMPO(oper, nsite=nsite)
        
        
        for sweep in range(nsweeps):
            
            sweep_elapsed_time = time.time()
            
            # --------------main-------------
            current_time, max_trunc_err = reduced_operator.tdvp_sweep_(
                state,  # 当前态
                order,  # 时间演化的阶数
                current_time,  # 当前时间
                time_step,  # 时间步长
                reverse_step,  # 时间演化必须是 True，虚时演化 False 回到 DMRG
                backend=backend,  # 计算有效哈密顿量基态的方法，"default", "expm_multiply", "lanczos"
                svd_alg=svd_alg,  # update MPS 中张量的方法，"svd", "eig"
                normalize=normalize,  # 是否归一化
                maxdim=chi_max[sweep],  # MPS 中最大的 bond 维度
                cutoff=trunc_cut[sweep],  # MPS 中奇异谱最小值
            )
            # -------------end main-----------
            
            sweep_elapsed_time = time.time() - sweep_elapsed_time
            
            if current_time.real == 0:
                if outputlevel >= 1:
                    print(f"--> time: {round(- current_time.imag,3)}: maxchi={state.maxbonddim()} maxtruncerr={max_trunc_err:.2e} elapsed_time={sweep_elapsed_time:.3f}", flush=True)
                    
                yield - current_time.imag, state
                
            else:
                if outputlevel >= 1:
                    print(f"--> time: {current_time/1j:.3f}: maxchi={state.maxbonddim()} maxtruncerr={max_trunc_err:.2e} elapsed_time={sweep_elapsed_time:.3f}", flush=True)
                yield current_time/1j, state
    
    def _tdvp_sub_time_steps(self, order:int):
        if order == 1:
            return tc.tensor([1, 0], dtype=tc.float64)
        elif order == 2:
            return tc.tensor([1/2, 1/2], dtype=tc.float64)
        elif order == 4:
            s = 1 / (2 - 2**(1/3))
            return tc.tensor([s/2, s/2, (1 - 2*s)/2, (1 - 2*s)/2, s/2, s/2], dtype=tc.float64)


class ProjMPO:
    def __init__(self, H, nsite=2) -> None:
        """
        ProjMPO 计算并存储 MPO 在由 MPS 定义的基中投影，保留 MPO 的某些站点索引未投影。
        可以通过调用 `set_position_` 方法来移动未投影的格点。

        ProjMPO `PH` 表示的网络图示（`PH.set_position_(psi, 3)`）：

        .. code-block:: text
        
            o--o--o-      -o--o--o--o--o--o <psi|
            |  |  |  |  |  |  |  |  |  |  |
            o--o--o--o--o--o--o--o--o--o--o H
            |  |  |  |  |  |  |  |  |  |  |
            o--o--o-      -o--o--o--o--o--o |psi>
                  ↑        ↑
               lpos=2    rpos=5
        """
        self.lpos:int = -1
        self.rpos:int = len(H.data)
        self.nsite:int = nsite
        self.L = len(H.data)
        self.mpo:MPO = H
        self.LR:list = [None] * len(H.data)
        self.dtype = H.dtype
        self.device = H.device
    
    def copy(self):
        new = ProjMPO(self.mpo, nsite=self.nsite)
        new.lpos = self.lpos
        new.rpos = self.rpos
        new.LR = [None if i is None else i.clone() for i in self.LR]
        new.dtype = self.dtype
        return new
    
    def _sweep_schedule(self):
        for position in range(self.L - self.nsite):
            yield (position, "right")
        for position in range(self.L - self.nsite, -1, -1):
            yield (position, "left")         
    
    def get_dim(self):
        dimnum = 1
        
        if self.lpos >= 0:
            Lenv = self.LR[self.lpos]
            dimnum *= Lenv.shape[0]
        
        for i in range(self.lpos + 1, self.rpos):
            dimnum *= self.mpo.data[i].shape[1]
        
        if self.rpos < len(self.mpo.data):
            Renv = self.LR[self.rpos]
            dimnum *= Renv.shape[0]
        
        return dimnum

    def makeL_(self, psi:MPS, k:int):
        if psi.llim <= k:
            print("警告：ProjMPO.makeL_(): psi.llim <= k")
            
        if k <= self.lpos:
            # 如果 k 比目前的 lpos 比小，那么就是从右往左移动，不需要重新计算
            self.lpos = k
        else:
            # 否则就是从左向右移动，需要利用 psi 得到新的 Lenv
            ll = max(self.lpos, -1)
            Lenv = self.lproj()
            while ll < k:
                Lenv = tf._contract_left_env(self.mpo.data[ll+1].to(dtype=self.dtype, device=self.device), psi.data[ll+1], Lenv)
                # 检查 Lenv 中是否有 inf, nan:
                if not tc.isfinite(Lenv).all():
                    raise ValueError(f"inf or nan in Lenv at {ll}")
                self.LR[ll + 1] = Lenv
                ll += 1
            self.lpos = k

    def lproj(self):
        if self.lpos <= -1:
            return tc.tensor([1.], dtype=self.dtype, device=self.device).reshape(1,1,1)
        return self.LR[self.lpos]

    def makeR_(self, psi:MPS, k:int):
        if psi.rlim >= k:
            print("警告：ProjMPO.makeR_(): psi.rlim >= k")
            
        if self.rpos <= k:
            # 如果 rpos 比目前的 k 比小，那么就是从左往右移动，不需要重新计算
            self.rpos = k
        else:
            # 否则就是从右向左移动，需要利用 psi 得到新的 Renv
            rl = min(self.rpos, len(self.mpo.data))
            Renv = self.rproj()
            while rl > k:
                Renv = tf._contract_right_env(self.mpo.data[rl - 1].to(dtype=self.dtype, device=self.device), psi.data[rl - 1], Renv)
                # 检查 Lenv 中是否有 inf, nan:
                if not tc.isfinite(Renv).all():
                    raise ValueError(f"inf or nan in Lenv at {rl}")
                self.LR[rl - 1] = Renv
                rl -= 1
            self.rpos = k

    def rproj(self):
        if self.rpos >= len(self.mpo.data):
            return tc.tensor([1.], dtype=self.dtype, device=self.device).reshape(1,1,1)
        return self.LR[self.rpos]
    
    def set_position_(self, psi:MPS, pos:int):
        self.dtype = psi.data[0].dtype
        self.makeL_(psi, pos - 1)
        self.makeR_(psi, pos + self.nsite)
    
    def solve_ground_state(self, v, *, method='default', lanczos_tol=1e-14):
        if method == 'default':
            # use ED for small matrix dimensions, but lanczos by default
            if self.get_dim() < 400:
                mat = self.to_matrix()
                E, theta = tc.linalg.eigh(mat)
                return E[0], theta[:, 0].reshape(*v.shape)
            else:
                return self.lanczos_ground(v, tol=lanczos_tol)
        if method == 'larpack':
            return self.larpack_ground(v, tol=lanczos_tol)
        elif method == 'lanczos':
            return self.lanczos_ground(v, tol=lanczos_tol)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def solve_evolve_state(self, v, delta, *, method='default', lanczos_tol=1e-14):
        if method == 'default':
            # use ED for small matrix dimensions, but lanczos by default
            if self.get_dim() < 400:
                mat = self.to_matrix()
                E, theta = tc.linalg.eigh(mat)
                expE = tc.exp(E * delta)
                theta = theta.to(dtype=expE.dtype,device=self.device)
                v = v.to(dtype=expE.dtype,device=self.device)
                exp_dH_v = theta @ (expE * (theta.H @ v.reshape(-1)))
                return exp_dH_v.reshape(*v.shape)
            else:
                return self.lanczos_evolve(v, delta, tol=lanczos_tol)
        if method == 'expm_multiply':
            return self.expm_multiply_evolve(v, delta, tol=lanczos_tol)
        elif method == 'lanczos':
            return self.lanczos_evolve(v, delta, tol=lanczos_tol)
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def prepare_solve(self):
        Lenv = self.lproj()
        Lenv = Lenv.reshape(Lenv.shape[0], -1).contiguous()
        
        if self.nsite == 2:
            H1 = self.mpo.data[self.lpos + 1]
            H2 = self.mpo.data[self.lpos + 2]
            H12 = tf._prepare_solve_ground_state(H1, H2).contiguous()
        elif self.nsite == 1:
            H12 = self.mpo.data[self.lpos + 1]
            d, e, *ijh = H12.shape
            H12 = H12.swapaxes(0,1).reshape(d*e, -1).contiguous()
        elif self.nsite == 0:
            H12 = None
        
        Renv = self.rproj()
        Renv = Renv.permute([2,1,0])
        Renv = Renv.reshape(Renv.shape[0], -1).contiguous()
        return Lenv, H12, Renv
        
    def to_matrix(self):
        Lenv, H12, Renv = self.prepare_solve()
        if H12 is None:
            return tf.make_matrix0(Lenv, Renv)
        out = tf.make_matrix(Lenv, H12.to(dtype=self.dtype,device=self.device), Renv)
        return out
    
    def larpack_ground(self, v, tol):
        s = v.shape
        Lenv, H12, Renv = self.prepare_solve()
        if self.nsite == 0:
            matvec = lambda v: tf._matrix_vector_product0(Lenv.numpy(), Renv.numpy(), v)
        else:
            matvec = lambda v: tf._matrix_vector_product(Lenv.numpy(), H12.numpy(), Renv.numpy(), v)
        val, vec = lanczos_arpack(matvec, v.numpy().reshape(-1), tol=tol)
        return tc.tensor(val, dtype=tc.float64, device=v.device), tc.tensor(vec, dtype=v.dtype, device=v.device).reshape(*s)
    
    
    def lanczos_ground(self, v, tol=1e-14):
        Lenv, H12, Renv = self.prepare_solve()
        if self.nsite == 0:
            matmul = lambda inipsi: tf._matrix_vector_product0(Lenv, Renv, inipsi)
        else:
            matmul = lambda inipsi: tf._matrix_vector_product(Lenv, H12, Renv, inipsi)
        val, vec = lanczos_ground_state(matmul, v.reshape(-1), tol=tol)
        return val, vec.reshape(*v.shape)
    
    def expm_multiply_evolve(self, v, delta, tol):
        Lenv, H12, Renv = self.prepare_solve()
        if self.nsite == 0:
            matmul = lambda inipsi: delta * tf._matrix_vector_product0(Lenv.numpy(), Renv.numpy(), inipsi)
            rmatmul = lambda inipsi: np.conj(delta) * tf._matrix_vector_product0(Lenv.numpy(), Renv.numpy(), inipsi)
            trmatul = tf._trace_matrix_vector_product0(Lenv, Renv).item() * delta
            # mat = self.to_matrix()
            # assert np.isclose(mat.trace().item(), trmatul)
            
        else:
            H12 = H12.to(dtype=self.dtype,device=self.device)
            matmul = lambda inipsi: delta * tf._matrix_vector_product(Lenv.numpy(), H12.numpy(), Renv.numpy(), inipsi)
            rmatmul = lambda inipsi: np.conj(delta)* tf._matrix_vector_product(Lenv.numpy(), H12.numpy(), Renv.numpy(), inipsi)
            trmatul = tf._trace_matrix_vector_product(Lenv, H12, Renv).item() * delta
            # mat = self.to_matrix()
            # assert np.isclose(mat.trace().item(), trmatul)
        
        res = expm_multiply(matmul, v.numpy().reshape(-1), traceA=trmatul, herm=rmatmul)
        return tc.tensor(res, dtype=v.dtype, device=v.device).reshape(*v.shape)
    
    def lanczos_evolve(self, v, delta, tol=1e-14):
        Lenv, H12, Renv = self.prepare_solve()
        if self.nsite == 0:
            matmul = lambda inipsi: tf._matrix_vector_product0(Lenv, Renv, inipsi)
        else:
            H12 = H12.to(dtype=self.dtype,device=self.device)
            matmul = lambda inipsi: tf._matrix_vector_product(Lenv, H12, Renv, inipsi)
        vec = lanczos_evolve_state(matmul, v.reshape(-1), delta, tol=tol)
        return vec.reshape(*v.shape)

    def dmrg_sweep_(self, 
                    psi:MPS,  # 当前态
                    position:int,   # 优化 postion 位置的张量
                    direction:str,   # 方向 'left' 或 'right'
                    svd_alg:str,    # update MPS 中张量的方法
                    chi_max:int,  # MPS 中最大的 bond 维度
                    svd_min:float,  # MPS 中奇异谱最小值
                    backend:str,   # 计算有效哈密顿量基态的方法，"default", "larpack", "lanczos"
                    max_trunc_err:float  # 最大的误差
                    ) -> tuple[float, float]:
        """
        优化 position 位置的张量:
        参数如下：
        .. code-block:: python
            psi:MPS,  # 当前态
            position:int,   # 优化 postion 位置的张量
            direction:str,   # 方向 'left' 或 'right'
            svd_alg:str,    # update MPS 中张量的方法
            chi_max:int,  # MPS 中最大的 bond 维度
            svd_min:float,  # MPS 中奇异谱最小值
            backend:str,   # 计算有效哈密顿量基态的方法，"default", "larpack", "lanczos"
            max_trunc_err:float  # 最大的误差
        """
        # prepare_update_local
        self.set_position_(psi, position)
        if self.nsite == 2:
            phi = tf._full_contract_right_mps2(psi.data[position], psi.data[position + 1])
        elif self.nsite == 1:
            phi = psi.data[position]
        phi = phi/tc.norm(phi)
        
        # solve for the ground state of the effective Hamiltonian
        energy, phi = self.solve_ground_state(phi, method=backend, lanczos_tol=max(svd_min, 0.05*max_trunc_err))
        # todo `lanczos_tol` 有没有更好的选择？
        
        # update the MPS
        if self.nsite == 2:
            err_step = psi.update_two_site_(position, phi, direction=direction, svd_alg=svd_alg, trunc_para=(chi_max, svd_min, None), normalize=True)
            return energy, err_step.eps
        elif self.nsite == 1:
            psi.update_single_site_(position, phi)
            return energy, 0.0
      
    def tdvp_sweep_(
            self,
            state, # 当前态
            order, # 时间演化的阶数
            current_time,  # 当前时间
            time_step, # 时间步长
            reverse_step, # 时间演化必须是 True，虚时演化 False 回到 DMRG
            maxdim,  # MPS 中最大的 bond 维度
            cutoff,  # MPS 中奇异谱最小值
            normalize,  # 是否归一化
            backend,  # 计算有效哈密顿量基态的方法，"default", "expm_multiply", "lanczos"
            svd_alg, # update MPS 中张量的方法
        ):
        # todo order = 1, 4 的实现
        order_orderings = ['forward', 'backward']
        order_sub_time_steps = self.sub_time_steps(order)
        order_sub_time_steps = [i * time_step for i in order_sub_time_steps]
        maxtruncerr = 0.0
        
        # subtime iteration
        for subtime, sub_time_step in enumerate(order_sub_time_steps):
            direction = order_orderings[subtime % 2]
            
            N = len(state.data)
            nsite = self.nsite
            
            if direction == 'forward':
                if state.rlim != state.llim or state.llim != 0:
                    state.orthogonalize_(0)
                assert state.rlim == state.llim == 0
                self.set_position_(state, 0)
            elif direction == 'backward':
                if state.rlim != state.llim or state.rlim != N - nsite:
                    state.orthogonalize_(N - nsite)
                assert state.rlim == state.llim == N - nsite
                self.set_position_(state, N - nsite)
            else:
                raise ValueError(f"direction = {direction} 不合法")
            
            # site iteration
            for b in self.sweep_bonds(direction, N):
                current_time, trunc_err = self.tdvp_sweep_local_(
                    reverse_step,  # bool 时间演化必须是 True，虚时演化 False 回到 DMRG
                    state,  # 当前态
                    b,  # 优化 b 位置的张量
                    current_time, # 当前时间
                    sub_time_step, # 时间步长
                    normalize, # 是否归一化
                    direction, # 方向 'forward' 或 'backward'
                    backend,  # 计算有效哈密顿量基态的方法，"default", "expm_multiply", "lanczos"
                    svd_alg,  # update MPS 中张量的方法
                    cutoff,  # MPS 中奇异谱最小值
                    maxdim,  # MPS 中最大的 bond 维度
                    maxtruncerr=maxtruncerr  # 最大的误差
                    )
                if maxtruncerr < trunc_err.eps:
                    maxtruncerr = trunc_err.eps
        return current_time, maxtruncerr

    def sub_time_steps(self, order):
        if order == 1:
            return [1.0, 0.0]
        elif order == 2:
            return [0.5, 0.5]
        elif order == 4:
            s = 1 / (2 - 2**(1/3))
            return [s/2, s/2, (1 - 2*s)/2, (1 - 2*s)/2, s/2, s/2]

    def sweep_bonds(self, direction, N):
        if direction == 'forward':
            return range(N - self.nsite + 1)
        elif direction == 'backward':
            return range(N - self.nsite, -1, -1)
        else:
            raise ValueError(f"direction = {direction} 不合法")
        
    def tdvp_sweep_local_(
            self,
            reverse_step, # bool 时间演化必须是 True，虚时演化 False 回到 DMRG
            state:MPS,  # 当前态
            b,  # 优化 b 位置的张量
            current_time, # 当前时间
            time_step, # 时间步长
            normalize, # 是否归一化
            direction, # 方向 'forward' 或 'backward'
            backend, # 计算有效哈密顿量基态的方法，"default", "expm_multiply", "lanczos"
            svd_alg,  # update MPS 中张量的方法
            cutoff,  # MPS 中奇异谱最小值
            maxdim,  # MPS 中最大的 bond 维度
            maxtruncerr # 最大的误差
    ):
        # todo 整理这段代码
        nsite = self.nsite
        if nsite == 2:
            
            self.set_position_(state, b)
            reduced_state = tf._full_contract_right_mps2(state.data[b], state.data[b + 1])
            
            reduced_state = self.solve_evolve_state(reduced_state, time_step, method=backend, lanczos_tol=max(cutoff, 0.05*maxtruncerr))
            
            current_time += time_step
            if normalize:
                nm = tc.norm(reduced_state)
                reduced_state = reduced_state / nm
                state.lognm += tc.log(nm)
            ortho = "right" if direction == "forward" else "left"
            
            trunc_err = state.update_two_site_(b, reduced_state, ortho, svd_alg=svd_alg, trunc_para=(maxdim, cutoff, None))
            
            maxtruncerr = max(maxtruncerr, trunc_err.eps)
            
            if not self.is_half_sweep_done(direction, b, len(state.data)) and reverse_step:
                # Do backwards evolution step
                b1 = b + 1 if direction == "forward" else b
                
                bond_reduced_state = state.data[b1]
                self.nsite = nsite - 1
                self.set_position_(state, b1)
                
                bond_reduced_state = self.solve_evolve_state(bond_reduced_state, -time_step, method=backend, lanczos_tol=max(cutoff, 0.05*maxtruncerr))
                
                current_time -= time_step
                if normalize:
                    nm = tc.norm(bond_reduced_state)
                    bond_reduced_state = bond_reduced_state / nm
                    state.lognm += tc.log(nm)
                
                state.update_single_site_(b1, bond_reduced_state)
                
                self.nsite = nsite
            return current_time, trunc_err
        
        # todo nsite = 1 以及 not reverse_step 的实现
        elif nsite == 1:
            self.set_position_(state, b)
            
            reduced_state = state.data[b].clone()
            
            reduced_state = self.solve_evolve_state(reduced_state, time_step, method=backend, lanczos_tol=max(cutoff, 0.05*maxtruncerr))
            
            current_time += time_step
            if normalize:
                nm = tc.norm(reduced_state)
                reduced_state = reduced_state / nm
                state.lognm += tc.log(nm)

            ortho = "right" if direction == "forward" else "left"
            
            state.update_single_site_(b, reduced_state)
            
            maxtruncerr = maxtruncerr
            
            
            if not self.is_half_sweep_done(direction, b, len(state.data)):
                # Do backwards evolution step
                if reverse_step:
                    if direction == "forward":
                        b1 = b + 1
                        U, S, V, _ = svd(reduced_state, lr_indx=[[0,1],[2]])
                        state.update_single_site_(b, U)
                        bond_reduced_state = S.reshape(-1,1) * V
                        state.llim += 1
                        
                    elif direction == "backward":
                        b1 = b
                        U, S, V, _ = svd(reduced_state, lr_indx=[[0],[1,2]])
                        state.update_single_site_(b, V)
                        bond_reduced_state = S.reshape(1,-1) * U
                        state.rlim -= 1
                        
                        
                    
                    self.nsite = nsite - 1
                    self.set_position_(state, b1)
                    
                    bond_reduced_state = self.solve_evolve_state(bond_reduced_state, -time_step, method=backend, lanczos_tol=max(cutoff, 0.05*maxtruncerr))
                    
                    current_time -= time_step
                    if normalize:
                        nm = tc.norm(bond_reduced_state)
                        bond_reduced_state = bond_reduced_state / nm
                        state.lognm += tc.log(nm)
                        
                    if direction == "forward":
                        nexttsr = state.data[b + 1]
                        nexttsrshape = nexttsr.shape
                        state.data[b + 1] = (bond_reduced_state @ nexttsr.reshape(nexttsrshape[0], -1)).reshape(-1, *nexttsrshape[1:])
                        state.rlim += 1
                        
                    elif direction == "backward":
                        nexttsr = state.data[b - 1]
                        nexttsrshape = nexttsr.shape
                        state.data[b - 1] = (nexttsr.reshape(-1, nexttsrshape[-1]) @ bond_reduced_state).reshape(*nexttsrshape[:-1],-1)
                        state.llim -= 1
                    
                    self.nsite = nsite
                else:
                    # only move ortho center
                    if direction == "forward":
                        state.orthogonalize_(b + 1)
                    elif direction == "backward":
                        state.orthogonalize_(b - 1)
                
                
            return current_time, TruncationError(0.0, 1.0)

    def is_half_sweep_done(self, direction, b, N):
        return (direction == "forward" and b == N - self.nsite) or \
               (direction == "backward" and b == 0)