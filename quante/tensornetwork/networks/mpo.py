# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-22 22:35:13
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-23 18:23:11


import math as math_lib
from numbers import Number
from typing import Generator, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from .mps import MPS
    from quimb.tensor.tensor_1d import MatrixProductOperator

from ..core.tensor_train import TensorTrain
from ..core import tensor_operations as tf
from ..core.tensor_utils import log_or_not_update, tt_decompose


class MPO(TensorTrain):
    def __init__(
        self,
        Ws: list[np.ndarray],
        Ss: list[np.ndarray] = None,
        llim: int = None,
        rlim: int = None,
        lognm: float = None,
        L: int = None,
    ):
        super().__init__(Ws, Ss, llim, rlim, lognm, L=L)

    _dm_get_R = tf._dm_get_R_mpo
    _dm_left2right = tf._dm_left2right_mpo
    _resume_canonical = tf._resume_canonical_mpo
    _apply_mpo_step = tf._apply_on_mpo_step

    def to_matrix(self):
        return tf._full_contract_mpo(self.data) * np.exp(self.lognm)
    
    def to_torch(self, device='cpu'):
        from ...bridge.torch_utils.networks import MPO as tcMPO  # type: ignore
        from ...bridge.torch_utils.core_utils import totc  # type: ignore
        import torch as tc # type: ignore
        Ws = totc(self.data)
        Ss = totc(self.Ss)
        lognm = tc.tensor(self.lognm, dtype=tc.float64, device=device)
        return tcMPO(Ws, Ss, self.llim, self.rlim, lognm, self.L, workdevice=device)

    def to_quimb(self) -> "MatrixProductOperator":
        import quimb.tensor as qtn

        if self.data[0].shape[0] == 1:
            res = []
            a, b, c, d = self.data[0].shape
            res.append(self.data[0].reshape(b, c, d).transpose([2, 0, 1]))
            for i in range(1, len(self.data) - 1):
                res.append(self.data[i].transpose([0, 3, 1, 2]))
            a, b, c, d = self.data[-1].shape
            res.append(self.data[-1].reshape(a, b, c))
            return np.exp(self.lognm).item() * qtn.MatrixProductOperator(res)
        # 周期 MPO
        res = [self.data[i].transpose([0, 3, 1, 2]) for i in range(len(self.data))]
        return np.exp(self.lognm).item() * qtn.MatrixProductOperator(res)

    def to_itensor(self, filename) -> None:
        from ...basicfun import save_hdf5

        assert self.data[0].shape[0] == 1, "只能处理 OBC"
        data_dict = {}
        data_dict["iscomplex"] = 1 if np.iscomplexobj(self.data[0]) else 0
        tsr = data_dict["tensors"] = {}
        for i in range(len(self.data)):
            if i == 0:
                a, b, c, d = self.data[i].shape
                tsr[f"W{i+1}"] = self.data[i].reshape(b, c, d)
            elif i == len(self.data) - 1:
                a, b, c, d = self.data[i].shape
                tsr[f"W{i+1}"] = self.data[i].reshape(a, b, c)
            else:
                tsr[f"W{i+1}"] = self.data[i]
        data_dict["lognm"] = np.asarray(self.lognm)
        data_dict["llim"] = self.llim
        data_dict["rlim"] = self.rlim + 2
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
        save_hdf5(filename, "/", data_dict)

    @classmethod
    def from_quimb(cls, mpo: "MatrixProductOperator", upper="k", lower="b") -> "MPO":
        linkinds = [
            i for i in mpo[0].inds
            if not i.startswith(lower) and not i.startswith(upper)
        ][0]
        tmp = mpo[0].to_dense([f"{upper}0"], [f"{lower}0"], [linkinds])
        res = [np.asarray(tmp).reshape(1, *tmp.shape)]

        for j in range(1, mpo.L - 1):
            linkinds_ = [
                i for i in mpo[j].inds
                if not i.startswith(lower) and not i.startswith(upper) and i != linkinds
            ][0]
            tmp = mpo[j].to_dense([linkinds], [f"{upper}{j}"], [f"{lower}{j}"], [linkinds_])
            res.append(np.asarray(tmp))
            linkinds = linkinds_

        tmp = mpo[-1].to_dense([linkinds], [f"{upper}{mpo.L-1}"], [f"{lower}{mpo.L-1}"])
        res.append(np.asarray(tmp).reshape(*tmp.shape, 1))
        return MPO(res)

    @classmethod
    def from_random(cls, L: int, bond_dim: Union[list[int], int], phys_dim: int = 2, dtype=np.complex128) -> "MPO":
        if isinstance(bond_dim, int):
            linkdims_ = [1] + [bond_dim] * (L - 1) + [1]
        else:
            assert len(bond_dim) == L + 1
            linkdims_ = bond_dim
        shape_list = [(linkdims_[i], phys_dim, phys_dim, linkdims_[i + 1]) for i in range(L)]
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            Ws = [(np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype) for shape in shape_list]
        else:
            Ws = [np.random.randn(*shape).astype(dtype) for shape in shape_list]
        return cls(Ws)

    @classmethod
    def from_heisenberg(cls, L, j=1, h=0, cyclic=False, pauli=True) -> "MPO":
        from ...generate.operas import heisenberg_operator

        ham = heisenberg_operator(L, j=j, h=h, cyclic=cyclic)
        return cls.from_oper(ham, L=L, pauli=pauli)

    @classmethod
    def from_eye(cls, L, phys_dim=2, dtype=np.float64) -> "MPO":
        eyempo = [None] * L
        if isinstance(phys_dim, int):
            phys_dim = [phys_dim] * L
        for i in range(L):
            dim = phys_dim[i]
            eyempo[i] = np.eye(dim, dtype=dtype).reshape(1, dim, dim, 1)
        return cls(eyempo)

    @classmethod
    def from_oper(cls, ham, L=None, pauli=True) -> "MPO":
        L = ham.L if L is None else L
        if hasattr(ham, "automata"):
            return cls([np.asarray(w) for w in ham.automata(L, pauli=pauli)])
        return cls.from_matrix(ham.to_matrix(pauli=pauli))

    @classmethod
    def from_matrix(cls, vec: np.ndarray, phys_dim=2, trunc_para=(None, None, None)) -> "MPO":
        """Generate a Matrix Product Operator (MPO) from a given matrix.

        Parameters
        ----------
        vec : np.ndarray
            The input tensor to be converted.
        phys_dim : int, optional
            The physical dimension of the tensor, by default 2.
        trunc_para : tuple, optional
            The truncation parameters, by default (None, None, None).
            - chi_max: int, the maximum bond dimension
            - svd_min: float, the minimum singular value
            - trunc_cut: float, the truncation threshold

        Returns
        -------
        MPO
            Matrix product operator.
        """
        tt, Ss, lognm = tt_decompose(vec, phys_dim, trunc_para=trunc_para)
        return MPO(Ws=tt, Ss=Ss, llim=0, rlim=0, lognm=lognm)

    def _get_str(self, full=False, l=4):
        out1 = (
            self.__class__.__name__
            + ";  "
            + str(self.data[0].dtype)
            + ";  "
            + f"norm: {self.norm():.3e}"
            + ";  "
            + f"maxbonddim: {self.maxbonddim()}"
            + ";\n"
        )
        L = len(self.data)
        if L < 15:
            full = True

        siteindx = self._get_str_index(full, l)
        tsrstr = self._get_tsr_str(siteindx)
        bonddim, site, phydims = self._get_full_str(tsrstr, siteindx)
        out2 = "physdim: " + phydims[0] + "\n"
        out3 = "         " + tsrstr + "\n"
        out4 = "physdim: " + phydims[1] + "\n"
        out5 = "bonddim: " + bonddim + "\n"
        out6 = "site:    " + site + "\n"
        return out1 + out2 + out3 + out4 + out5 + out6
        # out1 = (
        #     self.__class__.__name__
        #     + ";  "
        #     + str(self.data[0].dtype)
        #     + ";  "
        #     + f"norm: {self.norm():.3e}"
        #     + ";  "
        #     + f"maxbonddim: {self.maxbonddim()}"
        #     + ";\n"
        # )
        # L = len(self.data)
        # if L < 15:
        #     full = True
        # out2 = "physdim: "
        # out6 = "physdim: "
        # out3 = "         --"
        # out4 = "bonddim: "
        # out5 = "site:     "
        # llim = self.llim if self.llim is not None else -1
        # rlim = self.rlim if self.rlim is not None else -1
        # ldis = 0
        # rdis = llim if llim > -1 else rlim if rlim > -1 else L
        # tag = False
        # for i in range(L):
        #     if full or ldis < 2 or rdis <= 2:
        #         a, b, d, c = self.data[i].shape
        #         ldis += 1
        #         rdis -= 1
        #         if i < llim:
        #             out3 += "--|>--"
        #             ldis = 1 if rdis <= 0 else ldis
        #             rdis = (llim - i - 1) if rdis <= 0 else rdis
        #         elif i > rlim:
        #             out3 += "-<|---"
        #             ldis = 1 if rdis <= 0 else ldis
        #             rdis = (L - i - 1) if rdis <= 0 else rdis
        #         else:
        #             out3 += "--O---"
        #             ldis = 1 if rdis <= 0 else ldis
        #             rdis = (rlim - i - 1) if rdis <= 0 else rdis
        #         out2 += f"{b:>4}| "
        #         out6 += f"{d:>4}| "
        #         out4 += f"{a:^5} " if tag else f"{a:^4}  "
        #         out5 += f"  {i:^4}"
        #         tag = False
        #     elif ldis == 2:
        #         a, b, d, c = self.data[i].shape
        #         ldis += 1
        #         rdis -= 1
        #         out3 += " ... -"
        #         out2 += f"   ..."
        #         out6 += f"   ..."
        #         out4 = out4[:-1] + f"{a:^4}..."
        #         out5 += f"  ... "
        #         tag = True
        #     else:
        #         ldis += 1
        #         rdis -= 1

        # out4 += f" {c}"
        # out2 += "\n"
        # out6 += "\n"
        # out3 += "-\n"
        # out4 += "\n"
        # out = out1 + out2 + out3 + out6 + out4 + out5
        # return out

    def show(self, full=False):
        print(self._get_str(full=full))

    def __repr__(self) -> str:
        return self._get_str()

    def _apply_1b_gate(self, pos, gate_1b):
        if isinstance(gate_1b, np.ndarray):
            gate = gate_1b
            top_or_bottom = "top"
        else:
            gate, top_or_bottom = gate_1b

        if top_or_bottom == "top":
            gate = self._convert_gate(gate, 1)
            return tf._local_apply(self.data[pos], gate)
        elif top_or_bottom == "bottom":
            gate = self._convert_gate(gate, 1)
            return tf._local_apply(self.data[pos].swapaxes(1, 2), gate).swapaxes(1, 2)
        elif top_or_bottom == "topbottom":
            try:
                gate0, gate1 = gate
                gate0 = self._convert_gate(gate0, 1)
                gate1 = self._convert_gate(gate1, 1)
                return tf._local_apply2(self.data[pos], gate0, gate1)
            except TypeError:
                gate = self._convert_gate(gate, 1)
                return tf._local_apply2(self.data[pos], gate, gate)
        raise ValueError(f"Unknown gate direction: {top_or_bottom}")

    def _apply_2b_gate(self, pos, gate_2b):
        next_pos = pos + 1 if self.L != np.inf else (pos + 1) % len(self.data)

        if isinstance(gate_2b, np.ndarray):
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
        raise ValueError(f"Unknown gate direction: {top_or_bottom}")

    def mele(self, y, x, logscale=False):
        """
        Compute <y|A|x> = <y|Ax>
        """
        Lenv = tf._mele_init_left_env(self.data[0], y.data[0].conj(), x.data[0])
        lognm = np.array(0.0, dtype=self.dtype)
        for i in range(1, self.L):
            Lenv = tf._mele_contract_left_env(self.data[i], y.data[i].conj(), x.data[i], Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)
        a, *_ = Lenv.shape
        if logscale:
            return np.log(np.trace(Lenv.reshape(a, a))) + self.lognm + x.lognm + y.lognm
        return np.trace(Lenv.reshape(a, a)) * np.exp(self.lognm) * np.exp(x.lognm) * np.exp(y.lognm)

    def diag_inner(self, mps):
        return tf.diagonal_inner(self.data, mps.data)

    def exp(self, x, order=2, trunc_para: tuple[int, float, float] = (None, None, None), dtype=np.complex128):
        """
        - trunc_para 是一个包含三个数的元组，分别表示:
            - chi_max: int, 截断的最大值
            - svd_min: float, SVD 的最小值
            - trunc_cut: float, 截断的阈值
        """
        l = len(self.data)
        local_dims = [i.shape[1] for i in self.data]
        res = MPO(tf.mpo_eye(l, local_dims, dtype))
        for i in range(1, order + 1):
            tmp = self.copy()
            for _ in range(i - 1):
                tmp.apply_mpo_(self, trunc_para=trunc_para)
            res += x**i / math_lib.factorial(i) * tmp
            res.canonicalize_(trunc_para)
        return res

    def trace(self):
        data0 = self.data[0]
        Lenv = np.array(1.0, dtype=data0.dtype).reshape(data0.shape[0], data0.shape[0])
        for tsr in self.data:
            Lenv = Lenv @ tf._up_bottom_tr(tsr)
        return np.trace(Lenv) * np.exp(self.lognm)

    def dmrg(self, psi=None, **kwargs) -> tuple[float, "MPS"]:
        r"""DMRG 方法求解 MPO 的基态

        Returns
        -------
        energy : float
            基态能量
        psi : MPS
            基态

        todo: eig pertube mixer
        todo: dmrg 求第一激发态
        """
        from ..algorithms.dmrg import DMRG

        return DMRG(self, psi=psi, **kwargs).run2()

    def tdvp(self, init: "MPS", final_time: Number, time_step: Number, **kwargs) -> Generator[tuple[Union[float, complex], "MPS"], None, None]:
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

        References
        ----------
        https://arxiv.org/abs/1408.5056
        """
        from ..algorithms.tdvp import TDVP

        return TDVP(mpo=self, psi0=init, time_step=time_step, final_time=final_time, **kwargs).run()


class SumMPO:
    def __init__(self, Hs: list[MPO]) -> None:
        assert all(isinstance(H, MPO) for H in Hs), "Hs must be a list of MPOs"
        self.Hs = Hs

    def copy(self):
        return SumMPO([H.copy() for H in self.Hs])

    def __len__(self):
        return len(self.Hs[0])

    @property
    def shape(self):
        return self.Hs[0].shape

    @property
    def dtype(self):
        return self.Hs[0].dtype

    @property
    def phys_dim(self):
        return self.Hs[0].phys_dim

    def __iter__(self):
        return iter(self.Hs)

    @property
    def lognm(self):
        return np.array(0.0, dtype=np.float64)

    def to_matrix(self):
        return sum(H.to_matrix() for H in self.Hs)
