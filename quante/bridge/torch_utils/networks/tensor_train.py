# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 15:43:04
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-28 16:50:58

import copy
import warnings
import numpy as np
import torch as tc
from typing import Union, TypeVar, Optional
T = TypeVar('T')

from . import tensor_operations as tf
from ..core_utils import clone
from ..linalg.decomp import eigh, qr, rq, svd, truncate, log_or_not_update, TruncationError


class TensorTrain:
    def __init__(self, Ws: list[tc.Tensor], Ss: Optional[list[tc.Tensor]] = None, llim: Optional[int] = None, rlim: Optional[int] = None, lognm: Optional[float] = None, L:Optional[int] = None, workdevice='cpu') -> None:
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
        self.workdevice = workdevice
        self.lognm = lognm if lognm is not None else tc.tensor(0.0, dtype=tc.float64,device=self.device)
    
    def __len__(self):
        return self.L
    
    def __getitem__(self, key):
        return self.data[key]
    
    @property
    def phys_dim(self):
        return [x.shape[-2] for x in self.data]
    
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
            lognmber = tc.log(abs(number)) + self.lognm
        except TypeError:  # when number is pure float form,
            import numpy as np
            lognmber = np.log(abs(number)) + self.lognm
        self.data[0] *= number/abs(number)
        return type(self)(self.data, lognm=lognmber)

    def maxbonddim(self):
        if len(self.data) == 1:
            return 1
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
        return self
        
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

    def inner(self, anotherTT, logscale=False, conj=True):
        """计算 <ψ|ϕ>: ψ.inner(ϕ)

        默认情况下，``ψ | ϕ`` 等价于 ``ψ.inner(ϕ, logscale=False, conj=True)``。

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
        assert isinstance(anotherTT, type(self)), f"wrong type: {type(anotherTT)}"
        if conj:
            conjdata = [i.conj() for i in self.data]
        else:
            conjdata = self.data
        if logscale:
            coef, lognm = tf.tn_inner(conjdata, anotherTT.data, logscale=True) 
            return tc.log(coef) + lognm + self.lognm + anotherTT.lognm
        else:
            return tf.tn_inner(conjdata, anotherTT.data, logscale=False) * tc.exp(self.lognm) * tc.exp(anotherTT.lognm)

    def __or__(self, anotherTT):
        """计算默认内积: ``self | anotherTT`` 等价于 ``self.inner(anotherTT)``。"""
        if self is anotherTT:
            return self.norm(lognorm=False)**2
        return self.inner(anotherTT, logscale=False, conj=True)

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
        

    def canonicalize_(self, trunc_para:tuple[int,float,float]=(None,None,None), qrnormalize=False, canonicalform=True):
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
        if canonicalform:
            self.llim = self.rlim = None
        else:
            self.llim = 0
            self.rlim = 0
        self.lognm = self.lognm + lognm
        return trunc_err


    def apply_gate_(
        self,
        pos:int,
        gate:Union[tc.Tensor, tuple[tc.Tensor, str]],
        *,
        direction: str = None,
        svd_alg: str = "svd", 
        eigdirection: str = None,
        pertube: tc.Tensor = None,
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
        if pertube is not None:
            assert svd_alg == "eig", "pertube 仅支持 eig 算法"
            assert eigdirection is not None, "eigdirection 必须给定"

        if gate_range == 1:
            # if not unitary_gate:
            #     self.move_llim_(pos)
            #     self.move_rlim_(pos)
            
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
            return self.update_two_site_(
                pos, W, direction=direction, svd_alg=svd_alg, 
                trunc_para=trunc_para, normalize=normalize, 
                updateS=updateS, pertube=pertube, eigdirection=eigdirection
            )
            # -------------------------


    def _convert_gate(self, gate, site_num):
        if gate.ndim == site_num == 2:
            try:
                gate = gate.reshape(2,2,2,2)
            except Exception as e:
                raise ValueError(f"gate shape error: {gate.shape}, site_num={site_num}")
        if isinstance(gate, np.ndarray):
            if np.iscomplexobj(gate):
                gate = tc.tensor(gate, dtype=tc.complex128, device=self.workdevice)
                if not self.dtype.is_complex:
                    self.to(dtype=tc.complex128, device=self.workdevice)
            else:
                if self.dtype.is_complex:
                    gate = tc.tensor(gate, dtype=tc.complex128, device=self.workdevice)
                else:
                    gate = tc.tensor(gate, dtype=tc.float64, device=self.workdevice)
        else:
            if gate.dtype.is_complex and not self.dtype.is_complex:
                self.to(dtype=tc.complex128, device=self.workdevice)
            elif not gate.dtype.is_complex and self.dtype.is_complex:
                gate = gate.to(dtype=tc.complex128, device=self.workdevice)
        return gate


    def update_single_site_(self, pos, phi, normalize=False, unitary_gate=False):
        # if not unitary_gate:
        #     self.move_llim_(pos)
        #     self.move_rlim_(pos)
        
        assert self.data[pos].ndim == phi.ndim, "维度不匹配"
        phi, self.lognm = log_or_not_update(phi, self.lognm, use_log=normalize)
        self.data[pos] = phi
        if not unitary_gate:
            if self.is_canonical_form():
                self.llim = 0
                self.rlim = pos
            else:
                if self.llim > pos:
                    self.llim = pos
                if self.rlim < pos:
                    self.rlim = pos
        return TruncationError(0.0, 1.0)
    
    def update_two_site_(self, pos, W, direction, *,
                   svd_alg="eig",
                   trunc_para:tuple[int,float,float]=(None,None,None),
                   normalize=False,
                   eigdirection=None,
                   pertube=None,
                   updateS=True,
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
            return self.update_two_site_cannonical_(
                pos, W, svd_alg=svd_alg, trunc_para=trunc_para, normalize=normalize,
            )
        
        # -------------- 使用 qr ------------
        if svd_alg == "qr":
            assert trunc_para == (None, None, None) and pertube is None, 'qr method do not need trunc_para, normalize and pertube'
            Wshape = W.shape
            halfdim = W.ndim // 2
            direction = "right" if direction is None else direction
            if direction == "right":
                U, A = qr(W.reshape(*Wshape[:halfdim],-1))
                A, self.lognm = log_or_not_update(A, self.lognm, use_log=normalize)
                self.data[pos] = U.to(self.device)
                self.data[pos + 1] = A.reshape(-1,*Wshape[halfdim:]).to(self.device)
                self.llim = self.rlim = pos + 1
            elif direction == "left":
                B, U = rq(W.reshape(-1,*Wshape[halfdim:]))
                B, self.lognm = log_or_not_update(B, self.lognm, use_log=normalize)
                self.data[pos] = B.reshape(*Wshape[:halfdim],-1).to(self.device)
                self.data[pos + 1] = U.to(self.device)
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
                self.data[pos] = W1.to(self.device)
                self.data[pos + 1] = (S.reshape(-1, *([1]*(W2.ndim-1))) * W2).to(self.device)
                self.llim = self.rlim = pos + 1
            elif direction == "left":
                self.data[pos] = (W1 * S).to(self.device)
                self.data[pos + 1] = W2.to(self.device)
                self.llim = self.rlim = pos
            elif direction == "center":
                sqrtS = tc.sqrt(S)
                self.data[pos] = (W1 * sqrtS).to(self.device)
                self.data[pos + 1] = (sqrtS.reshape(-1, *([1]*(W2.ndim-1))) * W2).to(self.device)
                self.llim, self.rlim = pos, pos + 1
            else:
                raise ValueError(f"not defined direction (left or right): {direction}")
            
            if updateS:
                self.Ss[pos + 1] = S
                
            return trunc_err
        
        # -------------- 使用 eig ------------
        elif svd_alg == "eig":
            # todo 目前 eig 使用的是自动判断方向，是否有更好的做法？
            W1, S, W2, trunc_err, eigdirection = eigh(W, eigdirection=eigdirection, trunc_para=trunc_para, pertube=pertube, pos=pos, drt=direction)
            # save_hdf5("log.h5", f"{(pos, direction)}/7eigoutput", {f"W1": W1.reshape(-1), "S": S.reshape(-1), "W2": W2.reshape(-1)})
            
            # 如果 eig 选择的方向与需要的方向不一样，通过 qr 调整回来
            if direction is None:
                direction = eigdirection
            elif direction == "right" and eigdirection == "left":
                W1, R = qr(W1)
                W2 = (R @ W2.reshape(W2.shape[0],-1)).reshape(*W2.shape)
            elif direction == "left" and eigdirection == "right":
                L, W2 = rq(W2)
                W1 = (W1.reshape(-1, W1.shape[-1]) @ L).reshape(*W1.shape)
            
            self.data[pos] = W1.to(device=self.device)
            self.data[pos + 1] = W2.to(device=self.device)
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
                self.Ss[pos + 1] = (S if not normalize else S / tc.norm(S)).to(device=self.device)
                
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
            W1 = type(self)._resume_canonical(W, W2)
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
                Ws_mpo,
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
            self.data[0] = tf._apply_on_mps_step(Ws_mpo.data[0], self.data[0])
            # self.data[0] = Ws_mpo.data[0] @ self.data[0]
            return None
        
        dtype = self.dtype
        Lenvs = self._dm_get_Lenvs(Ws_mpo.data, self.data, n, dtype)
        
        V = tc.tensor(1., dtype=dtype, device=self.device).reshape(1,1)
        R = tc.tensor(1., dtype=dtype, device=self.device).reshape(1,1,1)
        R = type(self)._dm_get_R(Ws_mpo.data[n-1], self.data[n-1], R, V)
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
            if self.data[j].ndim == 3:
                nshape = (Ws_mpo.data[j].shape[1], )
            else:
                nshape = (Ws_mpo.data[j].shape[1], self.data[j].shape[2])
            self.data[j] = V.T.reshape(linkdim2, *nshape, linkdim)
            linkdim = linkdim2

            # 前进一步
            R = type(self)._dm_get_R(Ws_mpo.data[j-1], self.data[j-1], R, V)

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
            Lenv = type(self)._dm_left2right(Lenv, W[j], ψ[j])
            # print(Lenv.shape)
            Lenv = Lenv/tc.norm(Lenv)
            Lenvs.append(Lenv)
        return Lenvs

    
    def apply_mpo_naive_(
                self,
                Ws_mpo,
            ):
        """局部直接收缩
        
        bond 将指数增加
        """
        for i in range(len(self.data)):
            self.data[i] = type(self)._apply_mpo_step(Ws_mpo.data[i], self.data[i])


    def apply_mpo_zip_up(
        self,
        Ws_mpo,
        trunc_para: tuple[int, float, float] = (None, None, None),
        *,
        direction: str = "right",
        updateS: bool = True,
        normalize: bool = False,
    ) -> TruncationError:
        """Apply an MPO with the zip-up algorithm.

        The local MPO-MPS/MPO contraction is compressed immediately by SVD,
        so only one matrix ``carry`` is propagated through the chain instead
        of first building the full product-bond tensor train.
        """
        n = len(Ws_mpo.data)
        assert n == len(self.data), f"MPO 和 TensorTrain 的长度应该相等: {n} != {len(self.data)}"
        if n == 1:
            self.apply_mpo_naive_(Ws_mpo)
            self.lognm = self.lognm + Ws_mpo.lognm
            self.llim = self.rlim = 0
            return TruncationError(0.0, 1.0)

        trunc_err_sum = TruncationError(0.0, 1.0)

        if direction == "right":
            carry = None
            for j in range(n - 1):
                theta = type(self)._apply_mpo_step(Ws_mpo.data[j], self.data[j])
                if carry is not None:
                    raw_left = theta.shape[0]
                    theta = (carry @ theta.reshape(raw_left, -1)).reshape(
                        carry.shape[0], *theta.shape[1:]
                    )

                Wj, S, Vh, trunc_err = svd(
                    theta,
                    lr_indx=[list(range(theta.ndim - 1)), [theta.ndim - 1]],
                    trunc_para=trunc_para,
                )
                S, self.lognm = log_or_not_update(S, self.lognm, use_log=normalize)
                self.data[j] = Wj
                carry = S.reshape(-1, 1) * Vh.reshape(len(S), -1)
                trunc_err_sum += trunc_err
                if updateS:
                    self.Ss[j + 1] = S

            theta = type(self)._apply_mpo_step(Ws_mpo.data[-1], self.data[-1])
            raw_left = theta.shape[0]
            theta = (carry @ theta.reshape(raw_left, -1)).reshape(
                carry.shape[0], *theta.shape[1:]
            )
            theta, self.lognm = log_or_not_update(theta, self.lognm, use_log=normalize)
            self.data[-1] = theta
            self.llim = self.rlim = n - 1

        elif direction == "left":
            carry = None
            for j in range(n - 1, 0, -1):
                theta = type(self)._apply_mpo_step(Ws_mpo.data[j], self.data[j])
                if carry is not None:
                    raw_right = theta.shape[-1]
                    theta = (theta.reshape(-1, raw_right) @ carry).reshape(
                        *theta.shape[:-1], carry.shape[-1]
                    )

                U, S, Wj, trunc_err = svd(
                    theta,
                    lr_indx=[[0], list(range(1, theta.ndim))],
                    trunc_para=trunc_para,
                )
                S, self.lognm = log_or_not_update(S, self.lognm, use_log=normalize)
                self.data[j] = Wj
                carry = U.reshape(-1, len(S)) * S.reshape(1, -1)
                trunc_err_sum += trunc_err
                if updateS:
                    self.Ss[j] = S

            theta = type(self)._apply_mpo_step(Ws_mpo.data[0], self.data[0])
            raw_right = theta.shape[-1]
            theta = (theta.reshape(-1, raw_right) @ carry).reshape(
                *theta.shape[:-1], carry.shape[-1]
            )
            theta, self.lognm = log_or_not_update(theta, self.lognm, use_log=normalize)
            self.data[0] = theta
            self.llim = self.rlim = 0

        else:
            raise ValueError(f"not defined direction (left or right): {direction}")

        self.lognm = self.lognm + Ws_mpo.lognm
        return trunc_err_sum


    def apply_submpo_(self, Ws_mpo, start_pos: int):
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

    def swapsite_(self, i, j, direction='right', move_center=False, **kwargs):
        assert 0 <= i < self.L-1 and 0 < j < self.L, "超出范围"
        assert i < j
        
        if i != j - 1:
            for a in range(i, j-1):
                self.swapsite_(a, a+1, direction='right', **kwargs)
            for a in range(j-1, i-1, -1):
                self.swapsite_(a, a+1, direction='left', **kwargs)
            return
        
        if move_center:
            self.move_llim_(i)
            self.move_rlim_(i+1)

        contracted_tsr = tf._full_contract_two(self.data[i], self.data[i+1])
        if contracted_tsr.ndim == 4:
            contracted_tsr = contracted_tsr.permute(0, 2, 1, 3)
        else:
            contracted_tsr = contracted_tsr.permute(0, 3, 4, 1, 2, 5)
        
        self.update_two_site_(i, contracted_tsr, direction=direction, **kwargs)
