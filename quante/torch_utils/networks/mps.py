# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 15:43:40
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-20 19:25:00

import numpy as np
import torch as tc
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:  # 类型检查时，导入 torch
    from quimb.tensor.tensor_1d import MatrixProductState

from .tensor_train import TensorTrain
from . import tensor_operations as tf
from ..utils import promote_dtype
from ..linalg.decomp import log_or_not_update, tt_decompose
from ...generate.matrix import pauli_matrix

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

    _dm_get_R = tf._dm_get_R_mps
    _dm_left2right = tf._dm_left2right_mps
    _resume_canonical = tf._resume_canonical_mps
    _apply_mpo_step = tf._apply_on_mps_step

    def to_vector(self) -> tc.Tensor:
        return tf._full_contract_mps(self.data) * tc.exp(self.lognm)
    
    to_matrix = to_vector

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

    def to_itensor(self, filename) -> None:
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
        data_dict["llim"] = self.llim
        data_dict["rlim"] = self.rlim + 2
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
        save_hdf5(filename, '/', data_dict)
    
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
    
    @classmethod
    def from_vector(cls, vec: tc.Tensor, phys_dim=2, trunc_para=(None, None, None)) -> 'MPS':
        tt, Ss, lognm = tt_decompose(vec, phys_dim, trunc_para=trunc_para)
        return MPS(Ws=tt, Ss=Ss, llim=0, rlim=0, lognm=lognm)
    
    from_matrix = from_vector

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
        
        from .mpo import MPO
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

