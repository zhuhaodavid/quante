# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-22 22:29:52
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-23 22:07:09


from typing import TYPE_CHECKING, Union

import numpy as np
import re

if TYPE_CHECKING:
    from quimb.tensor.tensor_1d import MatrixProductState

from ..core.tensor_train import TensorTrain
from ..core import tensor_operations as tf
from ..core.tensor_utils import log_or_not_update, promote_dtype, tt_decompose
from ...generate.matrix import pauli_matrix


class MPS(TensorTrain):
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

    _dm_get_R = tf._dm_get_R_mps
    _dm_left2right = tf._dm_left2right_mps
    _resume_canonical = tf._resume_canonical_mps
    _apply_mpo_step = tf._apply_on_mps_step

    def to_vector(self) -> np.ndarray:
        return tf._full_contract_mps(self.data) * np.exp(self.lognm)

    to_matrix = to_vector

    def __or__(self, another):
        """Bra notation sugar.

        ``psi1 | psi2`` computes ``<psi1|psi2>``.
        ``psi1 | H`` returns a delayed ``BraMPS`` so that
        ``psi1 | H | psi2`` computes ``<psi1|H|psi2>``.
        """
        if isinstance(another, MPS):
            if another is self:
                return self.norm() ** 2
            return self.inner(another)

        from .mpo import MPO
        if isinstance(another, MPO):
            assert len(self) == len(another), (
                f"length mismatch: mps={len(self)}, mpo={len(another)}"
            )
            return BraMPS(self, another)

        if isinstance(another, tuple):
            return BraMPS(self, another)

        from ...generate.operas import SpinOper
        if isinstance(another, SpinOper):
            return BraMPS(self, another)
   
        return NotImplemented

    def to_quimb(self) -> "MatrixProductState":
        import quimb.tensor as qtn

        if self.data[0].shape[0] == 1:
            res = []
            a, b, d = self.data[0].shape
            res.append(self.data[0].reshape(b, d).transpose([1, 0]))
            for i in range(1, len(self.data) - 1):
                res.append(self.data[i].swapaxes(1, 2))
            a, b, d = self.data[-1].shape
            res.append(self.data[-1].reshape(a, b))
            return np.exp(self.lognm).item() * qtn.MatrixProductState(res)
        # 周期 MPS
        res = [self.data[i].swapaxes(1, 2) for i in range(len(self.data))]
        return np.exp(self.lognm).item() * qtn.MatrixProductState(res)

    def to_itensor(self, filename) -> None:
        from ...basicfun import save_hdf5

        assert self.data[0].shape[0] == 1, "只能处理 OBC"
        data_dict = {}
        data_dict["iscomplex"] = 1 if np.iscomplexobj(self.data[0]) else 0
        tsr = data_dict["tensors"] = {}
        for i in range(len(self.data)):
            if i == 0:
                a, b, d = self.data[i].shape
                tsr[f"W{i+1}"] = self.data[i].reshape(b, d)
            elif i == len(self.data) - 1:
                a, b, d = self.data[i].shape
                tsr[f"W{i+1}"] = self.data[i].reshape(a, b)
            else:
                tsr[f"W{i+1}"] = self.data[i]
        data_dict["lognm"] = np.asarray(self.lognm)
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
    end"""  # 在 julia 中运行这段还没就可以还原 mps 了
        save_hdf5(filename, "/", data_dict)

    @classmethod
    def from_quimb(cls, mps: "MatrixProductState") -> "MPS":
        siteinds = mps.outer_inds()

        inds1, inds2 = mps[0].inds
        if inds1 in siteinds:
            siteind, linkind1 = inds1, inds2
        else:
            siteind, linkind1 = inds2, inds1
        tmp = mps[0].to_dense([siteind], [linkind1])

        res = [np.asarray(tmp).reshape(1, *tmp.shape)]
        for j in range(1, mps.L - 1):
            inds1 = mps[j].inds
            for i in inds1:
                if i in siteinds:
                    siteind = i
                elif i != linkind1:
                    linkind2 = i
            tmp = mps[j].to_dense([linkind1], [siteind], [linkind2])
            res.append(np.asarray(tmp))
            linkind1 = linkind2

        inds1, inds2 = mps[-1].inds
        if inds1 in siteinds:
            siteind, linkind1 = inds1, inds2
        else:
            siteind, linkind1 = inds2, inds1
        tmp = mps[-1].to_dense([linkind1], [siteind])
        res.append(np.asarray(tmp).reshape(*tmp.shape, 1))
        return MPS(res)

    def to_tenpy(self):
        import tenpy.linalg.np_conserved as npc
        import tenpy.networks as tpn

        norm = self.norm()
        Bflat = [i.swapaxes(0, 1) for i in self.data]
        bc = "finite" if Bflat[0].shape[0] == 1 else "infinite"
        dtype = np.complex128 if np.iscomplexobj(self.data[0]) else np.float64

        # The following code is adapted from tenpy.networks.MPS.from_Bflat.
        sites = [tpn.site.SpinHalfSite(conserve=None, sort_charge=True)] * self.L
        L = len(sites)
        Bflat = list(Bflat)
        if len(Bflat) != L:
            raise ValueError("Length of Bflat does not match number of sites.")
        ci = sites[0].leg.chinfo
        legL = npc.LegCharge.from_qflat(ci, [ci.make_valid(None)] * Bflat[0].shape[1])
        legL = legL.bunch()[1]
        SVs = [np.ones(B.shape[1]) / np.sqrt(B.shape[1]) for B in Bflat]
        SVs.append(np.ones(Bflat[-1].shape[2]) / np.sqrt(Bflat[-1].shape[2]))
        Bs = []
        for i, site in enumerate(sites):
            B = np.array(Bflat[i], dtype)
            # calculate the LegCharge of the right leg
            legs = [site.leg, legL, None]  # other legs are known
            legs = npc.detect_legcharge(B, ci, legs, None, qconj=-1)
            B = npc.Array.from_ndarray(B, legs, dtype)
            B.iset_leg_labels(["p", "vL", "vR"])
            Bs.append(B)
            legL = legs[-1].conj()  # prepare for next `i`
        if bc == "infinite":
            # for an iMPS, the last leg has to match the first one.
            chdiff = Bs[-1].get_leg("vR").charges[0] - Bs[0].get_leg("vL").charges[0]
            Bs[-1] = Bs[-1].gauge_total_charge("vR", ci.make_valid(chdiff))
        return tpn.MPS(sites, Bs, SVs, form="B", bc=bc, norm=norm)

    @classmethod
    def from_random(cls, L: int, bond_dim: Union[list[int], int], phys_dim=2, dtype=np.complex128) -> "MPS":
        if isinstance(bond_dim, int):
            linkdims_ = [1] + [bond_dim] * (L - 1) + [1]
        else:
            assert len(bond_dim) == L + 1
            linkdims_ = bond_dim
        if isinstance(phys_dim, int):
            phys_dim_ = [phys_dim] * L
        else:
            assert len(phys_dim) == L
            phys_dim_ = phys_dim
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            psi1 = [
                (np.random.randn(linkdims_[i], phys_dim_[i], linkdims_[i + 1])
                 + 1j * np.random.randn(linkdims_[i], phys_dim_[i], linkdims_[i + 1])).astype(dtype)
                for i in range(L)
            ]
        else:
            psi1 = [
                np.random.randn(linkdims_[i], phys_dim_[i], linkdims_[i + 1]).astype(dtype)
                for i in range(L)
            ]
        return cls(psi1)

    @classmethod
    def from_product_state(cls, state: list[str], dtype=np.float64) -> "MPS":
        Ws = [np.zeros((1, 2, 1), dtype=dtype) for i in range(len(state))]
        for i, s in enumerate(state):
            if s == "up":
                Ws[i][0, 0, 0] += 1.0
            elif s == "down":
                Ws[i][0, 1, 0] += 1.0
            else:
                raise ValueError(f"state {s} is not defined")
        return cls(Ws)

    @classmethod
    def from_product_state2(cls, state: list[np.ndarray], dtype=np.float64) -> "MPS":
        Ws = [
            np.asarray(i.reshape(1, -1, 1), dtype=dtype)
            for i in state
        ]
        return cls(Ws)

    @classmethod
    def from_ghz_state(cls, L, dtype=np.float64) -> "MPS":
        if L == 1:
            return MPS([np.array([[[1.0 / np.sqrt(2)], [1.0 / np.sqrt(2)]]], dtype=dtype)])
        tsr1 = np.zeros((1, 2, 2), dtype=dtype)
        tsr1[0, 1, 1] += 1.0
        tsr1[0, 0, 0] += 1.0

        tsr2 = np.zeros((2, 2, 2), dtype=dtype)
        tsr2[1, 1, 1] += 1.0
        tsr2[0, 0, 0] += 1.0

        tsr3 = np.zeros((2, 2, 1), dtype=dtype)
        tsr3[1, 1, 0] += 1.0
        tsr3[0, 0, 0] += 1.0

        Ws = [tsr1] + [tsr2 for _ in range(L - 2)] + [tsr3]
        return cls(Ws, lognm=-np.log(2.0) / 2)

    @classmethod
    def from_w_state(cls, L, which="up", dtype=np.float64) -> "MPS":
        if L == 1:
            return MPS([np.array([[[1.0 / np.sqrt(2)], [1.0 / np.sqrt(2)]]], dtype=dtype)])
        i = 0 if which == "up" else 1
        tsr1 = np.zeros((1, 2, 2), dtype=dtype)
        tsr1[0, 1 - i, 0] += 1.0
        tsr1[0, i, 1] += 1.0

        tsr2 = np.zeros((2, 2, 2), dtype=dtype)
        tsr2[1, 1 - i, 1] += 1.0
        tsr2[0, i, 1] += 1.0
        tsr2[0, 1 - i, 0] += 1.0

        tsr3 = np.zeros((2, 2, 1), dtype=dtype)
        tsr3[1, 1 - i, 0] += 1.0
        tsr3[0, i, 0] += 1.0

        Ws = [tsr1] + [tsr2 for _ in range(L - 2)] + [tsr3]
        return cls(Ws, lognm=-np.log(float(L)) / 2)

    @classmethod
    def from_vector(cls, vec: np.ndarray, phys_dim=2, trunc_para=(None, None, None)) -> "MPS":
        tt, Ss, lognm = tt_decompose(vec, phys_dim, trunc_para=trunc_para)
        return MPS(Ws=tt, Ss=Ss, llim=0, rlim=0, lognm=lognm)

    from_matrix = from_vector



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
        out4 = "bonddim: " + bonddim + "\n"
        out5 = "site:    " + site + "\n"
        return out1 + out2 + out3 + out4 + out5
        

    def show(self, full=False, l=4):
        print(self._get_str(full=full, l=l))

    def __repr__(self) -> str:
        return self._get_str()
    
    def measure_single(self, operator:list[np.ndarray], pos:list[int], logscale=False):
        """
        局域算符的观测值：

        .. code-block:: text

            -----▷------▷------◻------⨞------⨞------⨞----- ψ1.conj()
                 |      |      |      |      |      |
                 |      |      ◻      |      ◻      |
                 |      |      |      |      |      |
            -----▷------▷------◻------⨞------⨞------⨞----- ψ2
                               ↑             ↑
                              pos           pos

        移动正交中心到 pos 位置，然后将 operator 作用在 pos 位置上。

        如果不是局域的测量，使用单体门作用后 inner 的方法计算。

        pauli 只当 operator 是 SpinOper 时生效，表示 operator 是 Pauli 矩阵。

        #todo 使用局部 MPO 的方法来计算非最近邻的观测值
        """
        assert len(operator) == len(pos), f"长度需要一致, operator = {operator}, pos = {pos}"
        assert len(pos) == len(set(pos)), f"位置必须唯一, pos = {pos}"
        for p,o in zip(pos, operator):
            assert isinstance(o, np.ndarray)
            assert isinstance(p, int)
        assert (np.diff(pos) > 0).all(), "pos 需要从小到大"

        # 做个正则化
        firstpos, lastpos = pos[0], pos[-1]
        if self.is_canonical_form():
            firstdata = self.Ss[firstpos].reshape(-1, 1, 1) * self.data[firstpos]
        else:
            self.move_llim_(firstpos)
            self.move_rlim_(pos[-1])
            firstdata = self.data[firstpos]

        # 收缩左环境
        Lenv = tf._ProjMPS_contract_left_env(
            firstdata,
            tf._local_apply(firstdata, operator[0]),
            np.eye(firstdata.shape[0], dtype=firstdata.dtype),
        )
        lognm = 0.0
        ct = 1
        for i in range(firstpos + 1, lastpos + 1):
            if i in pos:
                data, mat = self.data[i], operator[ct]
                Lenv = tf._ProjMPS_contract_left_env(data, tf._local_apply(data, mat), Lenv)
                ct += 1
            else:
                data = self.data[i]
                Lenv = tf._ProjMPS_contract_left_env(data, data, Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)

        if logscale:
            return np.log(np.trace(Lenv)) + self.lognm * 2
        return np.trace(Lenv) * np.exp(self.lognm) ** 2

    def measure_local(self, operator:np.ndarray, pos:list[int], logscale=False):
        """
        局域算符的观测值：

        .. code-block:: text

            -----▷------▷------◻------⨞------⨞------⨞----- ψ.conj()
                 |      |      |      |      |      |
                 |      |     -----------------     |
                 |      |    |     operator    |    |
                 |      |     -----------------     |
                 |      |      |      |      |      |
            -----▷------▷------◻------⨞------⨞------⨞----- ψ
                               ↑      ↑      ↑
                              pos    pos    pos
        """
        assert (np.diff(pos) == 1).all(), "必须是连续的格点"
        dim = 1
        for p in pos:
            dim *= self.data[p].shape[1]
            if dim == operator.shape[0]:
                break
        else:
            raise ValueError("operator shape is not match")
        minpos, maxpos = pos[0], pos[-1]

        if self.is_canonical_form():
            contracted_tsr = self.Ss[minpos].reshape(-1, 1, 1) * self.data[minpos]
        else:
            self.orthogonalize_(minpos)
            contracted_tsr = self.data[minpos]
        for i in range(minpos + 1, maxpos + 1):
            contracted_tsr = tf._full_contract_right_mps(contracted_tsr, self.data[i])
        res = contracted_tsr.conj().reshape(-1) @ tf._local_apply(contracted_tsr, operator).reshape(-1)
        if logscale:
            return self.lognm * 2 + np.log(res)
        return np.exp(self.lognm * 2) * res

    def measure_mpo(self, operator:np.ndarray, pos:int, logscale=False):
        """
        mpo 的观测值：

        .. code-block:: text

            -----▷------▷------◻------⨞------⨞------⨞----- ψ.conj()
                 |      |      |      |      |      |
                 |      |      ◻------◻------◻      |
                 |      |      |      |      |      |
            -----▷------▷------◻------⨞------⨞------⨞----- ψ
                               ↑             ↑
                              pos           pos
        """
        if self.is_canonical_form():
            firstdata = self.Ss[pos].reshape(-1, 1, 1) * self.data[pos]
        else:
            self.move_llim_(pos)
            self.move_rlim_(pos + operator.L - 1)
            firstdata = self.data[pos]
        
        Lenv = tf._mele_init_left_env2(operator.data[0], firstdata.conj(), firstdata)
        lognm = 0.0
        for i in range(1, operator.L):
            Lenv = tf._mele_contract_left_env(operator.data[i], self.data[pos + i].conj(), self.data[pos + i], Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)
        a, b, c, d = Lenv.shape
        trLenv = np.trace(Lenv.reshape(b, d))
        if logscale:
            return np.log(trLenv) + self.lognm * 2 + operator.lognm
        return trLenv * np.exp(self.lognm) ** 2 * np.exp(operator.lognm)


    def measure(self, operator: Union[str, list[np.ndarray], np.ndarray], pos: list[int] = None, pauli=False, logscale=False) -> np.ndarray:
        """计算观测值

        operator 可以是
        - string
        - list of array
        - SpinOper
        - MPO
        """
        # -------- 单体门观测 --------
        if isinstance(operator, str):
            assert pos is not None
            if len(operator) == 1 and isinstance(pos, int):
                pos = [pos]
            operator = [pauli_matrix(o) for o in operator]
        if isinstance(operator, list):
            assert pos is not None
            return self.measure_single(operator, pos, logscale)

        # -------- 局域门 --------
        if isinstance(operator, np.ndarray):
            assert pos is not None
            return self.measure_local(operator, pos, logscale)

        # -------- Oper --------
        from ...generate.operas import SpinOper
        if isinstance(operator, SpinOper):
            pos, operator = operator._minimal_shift()
            return self.measure_mpo(operator.to_mpo(pauli=pauli), pos, logscale)

        # -------- MPO --------
        from .mpo import MPO
        if isinstance(operator, MPO):
            if pos is None:
                pos = 0
            return self.measure_mpo(operator, pos, logscale)

        raise ValueError(f"operator type {type(operator)} is not supported")


    def _apply_1b_gate(self, pos, gate_1b):
        gate_1b = self._convert_gate(gate_1b, 1)
        return tf._local_apply(self.data[pos], gate_1b)

    def _apply_2b_gate(self, pos, gate_2b):
        gate_2b = self._convert_gate(gate_2b, 2)
        next_pos = pos + 1 if self.L != np.inf else (pos + 1) % len(self.data)
        W1, W2 = self.data[pos], self.data[next_pos]
        return tf._apply_2b_gate_mps(W1, W2, gate_2b)

    def density_matrix(self, pos):
        ldim = self.data[0].shape[0]
        Lenv = np.eye(ldim**2, dtype=self.dtype).reshape(ldim, ldim, ldim, ldim)

        i = 0
        while i < pos:
            Lenv = tf._inner_step(Lenv, self.data[i].conj(), self.data[i])
            i += 1

        rdim = self.data[-1].shape[-1]
        Renv = np.eye(rdim**2, dtype=self.dtype).reshape(rdim, rdim, rdim, rdim)
        i = self.L - 1
        while i > pos:
            Renv = tf._inner_step(
                Renv, self.data[i].conj().swapaxes(0, -1), self.data[i].swapaxes(0, -1)
            )
            i -= 1

        W = self.data[pos]
        return np.einsum("abcd,cge,dhf,abef->gh", Lenv, W.conj(), W, Renv)

class BraMPS:
    """Delayed expression for ``bra | mpo``.

    It stores the left MPS and the MPO without contracting them. The matrix
    element is evaluated when another MPS is attached on the right:
    ``bra | mpo | ket``.
    """

    def __init__(self, bra: "MPS", operator) -> None:
        self.bra = bra
        self.operator = operator
    
    def __or__(self, ket):
        assert isinstance(ket, MPS)
        from .mpo import MPO
        if self.bra is ket:
            if isinstance(self.operator, tuple):
                return self.bra.measure(*self.operator)
            else:
                return self.bra.measure(self.operator)
        if isinstance(self.operator, MPO):
            return self.operator.mele(self.bra, ket)
        if isinstance(self.operator, tuple):
            operator, pos = self.operator
            # -------- 单体门 --------
            if isinstance(operator, str):
                assert pos is not None
                operator = [pauli_matrix(o) for o in operator]
            if isinstance(operator, list):
                assert pos is not None
                return self.inner_single(operator, pos, ket)
            # -------- 局域门 --------
            if isinstance(operator, np.ndarray):
                assert pos is not None
                return self.inner_local(operator, pos, ket)
            # -------- MPO --------
            from .mpo import MPO
            if isinstance(operator, MPO):
                assert pos is not None
                return self.inner_mpo(operator, pos, ket)
        # -------- Oper --------
        from ...generate.operas import SpinOper
        if isinstance(self.operator, SpinOper):
            pos, operator = self.operator._minimal_shift()
            return self.inner_mpo(operator.to_mpo(pauli=False), pos, ket)
        return NotImplemented
    

    def inner_single(self, operator:list[np.ndarray], pos:list[int], ket:"MPS", logscale=False):
        """
        局域算符的观测值：

        .. code-block:: text

            -----▷------▷------◻------⨞------⨞------⨞----- bra.conj()
                 |      |      |      |      |      |
                 |      |      ◻      |      ◻      |
                 |      |      |      |      |      |
            -----▷------▷------◻------⨞------⨞------⨞----- ket
                               ↑             ↑
                              pos           pos

        移动正交中心到 pos 位置，然后将 operator 作用在 pos 位置上。

        如果不是局域的测量，使用单体门作用后 inner 的方法计算。

        pauli 只当 operator 是 SpinOper 时生效，表示 operator 是 Pauli 矩阵。

        #todo 使用局部 MPO 的方法来计算非最近邻的观测值
        """
        assert len(operator) == len(pos), f"长度需要一致, operator = {operator}, pos = {pos}"
        assert len(pos) == len(set(pos)), f"位置必须唯一, pos = {pos}"
        for p,o in zip(pos, operator):
            assert isinstance(o, np.ndarray)
            assert isinstance(p, int)
        assert (np.diff(pos) > 0).all(), "pos 需要从小到大"

        # 收缩左环境
        bra0 = self.bra.data[0]
        Lenv = np.eye(bra0.shape[0], dtype=bra0.dtype).reshape(1,1)
        lognm = 0.0
        ct = 0
        for i in range(self.bra.L):
            if i in pos:
                bra_i, mat, ket_i = self.bra.data[i], operator[ct], ket.data[i]
                Lenv = tf._ProjMPS_contract_left_env(bra_i.conj(), tf._local_apply(ket_i, mat), Lenv)
                ct += 1
            else:
                bra_i, ket_i = self.bra.data[i], ket.data[i]
                Lenv = tf._ProjMPS_contract_left_env(bra_i.conj(), ket_i, Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)

        if logscale:
            return np.log(np.trace(Lenv)) + self.bra.lognm + ket.lognm
        return np.trace(Lenv) * np.exp(self.bra.lognm + ket.lognm)

    def inner_local(self, operator:np.ndarray, pos:list[int], ket:"MPS", logscale=False):
        """
        局域算符的观测值：

        .. code-block:: text

            -----▷------▷------◻------⨞------⨞------⨞----- ψ.conj()
                 |      |      |      |      |      |
                 |      |     -----------------     |
                 |      |    |     operator    |    |
                 |      |     -----------------     |
                 |      |      |      |      |      |
            -----▷------▷------◻------⨞------⨞------⨞----- ψ
                               ↑      ↑      ↑
                              pos    pos    pos
        """
        assert (np.diff(pos) == 1).all(), "必须是连续的格点"
        dim = 1
        for p in pos:
            dim *= self.bra.data[p].shape[1]
            if dim == operator.shape[0]:
                break
        else:
            raise ValueError("operator shape is not match")
        minpos, maxpos = pos[0], pos[-1]

        # 收缩左环境
        bra0 = self.bra.data[0]
        Lenv = np.eye(bra0.shape[0], dtype=bra0.dtype).reshape(1,1)
        lognm = 0.0
        for i in range(minpos):
            bra_i, ket_i = self.bra.data[i], ket.data[i]
            Lenv = tf._ProjMPS_contract_left_env(bra_i.conj(), ket_i, Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)

        bra_contracted_tsr = self.bra.data[minpos]
        for i in range(minpos + 1, maxpos + 1):
            bra_contracted_tsr = tf._full_contract_right_mps(bra_contracted_tsr, self.bra.data[i])
        
        ket_contracted_tsr = ket.data[minpos]
        for i in range(minpos + 1, maxpos + 1):
            ket_contracted_tsr = tf._full_contract_right_mps(ket_contracted_tsr, ket.data[i])

        ket_contracted_tsr = tf._local_apply(ket_contracted_tsr, operator)
        Lenv = tf._ProjMPS_contract_left_env(bra_contracted_tsr.conj(), ket_contracted_tsr, Lenv)
        Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)

        for i in range(maxpos+1, self.bra.L):
            bra_i, ket_i = self.bra.data[i], ket.data[i]
            Lenv = tf._ProjMPS_contract_left_env(bra_i.conj(), ket_i, Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)

        if logscale:
            return np.log(np.trace(Lenv)) + self.bra.lognm + ket.lognm
        return np.trace(Lenv) * np.exp(self.bra.lognm + ket.lognm)

    def inner_mpo(self, operator:np.ndarray, pos:int, ket:"MPS", logscale=False):
        """
        mpo 的观测值：

        .. code-block:: text

            -----▷------▷------◻------⨞------⨞------⨞----- ψ.conj()
                 |      |      |      |      |      |
                 |      |      ◻------◻------◻      |
                 |      |      |      |      |      |
            -----▷------▷------◻------⨞------⨞------⨞----- ψ
                               ↑             ↑
                              pos           pos
        """
        bra0, ket0 = self.bra.data[0], ket.data[0]
        
        Lenv = np.eye(bra0.shape[0], dtype=bra0.dtype).reshape(1,1)
        lognm = 0.0
        for i in range(pos):
            bra_i, ket_i = self.bra.data[i], ket.data[i]
            Lenv = tf._ProjMPS_contract_left_env(bra_i.conj(), ket_i, Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)

        a, b = Lenv.shape
        Lenv = Lenv.reshape(1,a,1,b)
        for i in range(operator.L):
            Lenv = tf._mele_contract_left_env(operator.data[i], self.bra.data[pos + i].conj(), ket[pos + i], Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)
        
        a, b, c, d = Lenv.shape
        assert a == c == 1
        Lenv = Lenv.reshape(b, d)
        for i in range(pos+operator.L, ket.L):
            bra_i, ket_i = self.bra.data[i], ket.data[i]
            Lenv = tf._ProjMPS_contract_left_env(bra_i.conj(), ket_i, Lenv)
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=logscale)

        trLenv = np.trace(Lenv)
        if logscale:
            return np.log(trLenv) + self.bra.lognm + ket.lognm + operator.lognm
        return trLenv * np.exp(self.bra.lognm + ket.lognm + operator.lognm) ** 2 


    # def _get_str(self, full=True, l=4):
    #     out = (
    #         "BraMPS;  "
    #         + f"sites: {len(self.bra)}; \n"
    #     )
    #     siteindx = self.bra._get_str_index(full, l)
    #     tsrstr = self.bra._get_tsr_str(siteindx)
    #     bonddim, site, bra_phydims = self.bra._get_full_str(tsrstr, siteindx)
    #     out += "site:     " + site + "\n"
    #     out += "bra:      " + tsrstr + "\n"

    #     bonddim = ['|' if bra_phydims[0][i] == '|' else c for i, c in enumerate(bonddim)]
    #     out += f"bra bond: {''.join(bonddim)}\n"

    #     from .mpo import MPO
    #     for o in self.operators:
    #         print(isinstance(o, MPO))
    #         if isinstance(o, MPO):
    #             mpo_str = o._get_tsr_str(siteindx)
    #             out += "mpo:      " + mpo_str + "\n"
    #             bonddim, site, mpo_phydims = o._get_full_str(tsrstr, siteindx)
    #             bonddim = ['|' if mpo_phydims[0][i] == '|' else c for i, c in enumerate(bonddim)]
    #             out += "mpo bond: " + "".join(bonddim) + "\n"
    #         else:
    #             return NotImplemented
    #     return out
    
    # def show(self, full=False, l=4):
    #     print(self._get_str(full=full, l=l))

    # def __repr__(self) -> str:
    #     return self._get_str()

    def _get_str(self, full=False, l=4):
        from .mpo import MPO

        if not isinstance(self.operator, MPO):
            return f"<bra|{type(self.operator).__name__}; sites: {len(self.bra)};"

        L = len(self.bra)
        if L < 15:
            full = True

        siteindx = sorted(
            set(self.bra._get_str_index(full, l))
            | set(self.operator._get_str_index(full, l))
        )

        bra_tsrstr = self.bra._get_tsr_str(siteindx)
        bra_bonddim, site, bra_phydims = self.bra._get_full_str(
            bra_tsrstr, siteindx
        )

        mpo_tsrstr = self.operator._get_tsr_str(siteindx)
        mpo_bonddim, _, mpo_phydims = self.operator._get_full_str(
            mpo_tsrstr, siteindx
        )

        out = (
            "<bra|mpo"
            + ";  "
            + str(np.result_type(self.bra.dtype, self.operator.dtype))
            + ";  "
            + f"sites: {L}"
            + ";\n"
        )
        out += "bra bond: " + bra_bonddim + "\n"
        out += "bra:      " + bra_tsrstr + "  bra.conj()\n"
        out += "bra phys: " + bra_phydims[0] + "\n"
        out += "mpo phys: " + mpo_phydims[0] + "\n"
        out += "mpo:      " + mpo_tsrstr + "  mpo\n"
        out += "ket phys: " + mpo_phydims[1] + "  open ket legs\n"
        out += "mpo bond: " + mpo_bonddim + "\n"
        out += "site:     " + site + "\n"
        return out
    
    def show(self, full=False, l=4):
        print(self._get_str(full=full, l=l))

    def __repr__(self) -> str:
        return self._get_str()
