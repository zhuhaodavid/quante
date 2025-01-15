# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-15 14:50:09
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-18 03:31:28

import time
import torch as tc
import numpy as np
from typing import TypeVar, Generator, Union
from functools import reduce
T = TypeVar('T')

from . import MPS, MPO
from . import tnfuncs as tf
from ..linalg import lanczos_ground_state, lanczos_evolve_state, svd, expm_multiply
from ...linalg.krylov import lanczos_arpack, tenpy_arnoldi
# from ...linalg.evolve import expm_multiply
from ...linalg.svd_robust import TruncationError

__all__ = ['DMRG', 'TDVP']


def solve_ground_state(oper:'ProjMPO', v, *, method='default', lanczos_tol=1e-14):
    """计算 ProjMPO 的 ground state """
    if method == 'default':
        # use ED for small matrix dimensions, but lanczos by default
        if oper.shape < 400:
            mat = oper.to_matrix()
            E, theta = tc.linalg.eigh(mat)
            return E[0], theta[:, 0].reshape(*v.shape)
        else:
            s = v.shape
            matmul = oper.get_matmul_func('torch')
            val, vec = lanczos_ground_state(matmul, v.reshape(-1), tol=lanczos_tol)
            return val, vec.reshape(*s)

    s = v.shape
    if method == 'lanczos':  # 自己实现的 lanczos
        matmul = oper.get_matmul_func('torch')
        val, vec = lanczos_ground_state(matmul, v.reshape(-1), tol=lanczos_tol)
        return val, vec.reshape(*s)

    matmul = oper.get_matmul_func('numpy')
    if method == 'larpack': # scipy sparse eigs
        val, vec = lanczos_arpack(matmul, v.numpy().reshape(-1), tol=lanczos_tol)
        return tc.tensor(val, device=v.device), tc.tensor(vec, dtype=v.dtype, device=v.device).reshape(*s)

    elif method == 'arnoldi':  # tenpy arnoldi 用来处理非厄密矩阵时可以考虑 #todo 自己实现
        val, vec = tenpy_arnoldi(matmul, v.numpy().reshape(-1))
        return tc.tensor(val, device=v.device), tc.tensor(vec, dtype=v.dtype, device=v.device).reshape(*s)

    else:
        raise ValueError(f"Unknown method: {method}")

def solve_evolve_state(oper:'ProjMPO', v, delta, *, method='default', lanczos_tol=1e-14):
    """计算 ProjMPO 的 evolve state """
    if method == 'default':
        # use ED for small matrix dimensions, but lanczos by default
        if oper.shape < 400:
            mat = oper.to_matrix()
            E, theta = tc.linalg.eigh(mat)
            expE = tc.exp(E * delta)
            theta = theta.to(dtype=expE.dtype,device=oper.device)
            v = v.to(dtype=expE.dtype,device=oper.device)
            exp_dH_v = theta @ (expE * (theta.H @ v.reshape(-1)))
            return exp_dH_v.reshape(*v.shape)
        else:
            matmul = oper.get_matmul_func('torch')
            vec = lanczos_evolve_state(matmul, v.reshape(-1), delta, tol=lanczos_tol)
            return vec.reshape(*v.shape)
    if method == 'lanczos':  # 使用 lanczos 进行演化
        matmul = oper.get_matmul_func('torch')
        vec = lanczos_evolve_state(matmul, v.reshape(-1), delta, tol=lanczos_tol)
        return vec.reshape(*v.shape)
    elif method == 'expm_multiply':  # 使用 expm_multiply 进行演化
        Lenv, H12, Renv = oper.prepare_solve()
        if oper.nsite == 0:
            matmul = lambda inipsi: delta * tf._matrix_vector_product0(Lenv, Renv, inipsi)
            rmatmul = lambda inipsi: np.conj(delta) * tf._matrix_vector_product0(Lenv, Renv, inipsi)
            trmatul = tf._trace_matrix_vector_product0(Lenv, Renv).item() * delta
        else:
            H12 = H12.to(dtype=oper.dtype,device=oper.device)
            matmul = lambda inipsi: delta * tf._matrix_vector_product(Lenv, H12, Renv, inipsi)
            rmatmul = lambda inipsi: np.conj(delta)* tf._matrix_vector_product(Lenv, H12, Renv, inipsi)
            trmatul = tf._trace_matrix_vector_product(Lenv, H12, Renv).item() * delta
        res = expm_multiply(matmul, v, traceA=trmatul, herm=rmatmul)
        return res.reshape(*v.shape)
    else:
        raise ValueError(f"Unknown method: {method}")


class ProjOper:
    def __init__(self, mid, nsite=2) -> None:
        self.mid = mid
        self.lpos = -1
        self.rpos = self.L = len(mid)
        self.dtype = mid.dtype
        self.device = mid.device
        self.LR = [None] * self.L
        self.nsite = nsite

    def copy(self:T) -> T:
        new = self(self.mid, nsite=self.nsite)
        new.lpos = self.lpos
        new.rpos = self.rpos
        new.LR = [None if i is None else i.clone() for i in self.LR]
        new.dtype = self.dtype
        return new

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
                Lenv = self._contract_left_env(self.mid.data[ll+1].to(dtype=self.dtype, device=self.device), psi.data[ll+1], Lenv)
                # 检查 Lenv 中是否有 inf, nan:
                if not tc.isfinite(Lenv).all():
                    raise ValueError(f"inf or nan in Lenv at {ll}")
                self.LR[ll + 1] = Lenv
                ll += 1
            self.lpos = k

    def makeR_(self, psi:MPS, k:int):
        if psi.rlim >= k:
            print("警告：ProjMPO.makeR_(): psi.rlim >= k")
            
        if self.rpos <= k:
            # 如果 rpos 比目前的 k 比小，那么就是从左往右移动，不需要重新计算
            self.rpos = k
        else:
            # 否则就是从右向左移动，需要利用 psi 得到新的 Renv
            rl = min(self.rpos, len(self.mid.data))
            Renv = self.rproj()
            while rl > k:
                Renv = self._contract_right_env(self.mid.data[rl - 1].to(dtype=self.dtype, device=self.device), psi.data[rl - 1], Renv)
                # 检查 Lenv 中是否有 inf, nan:
                if not tc.isfinite(Renv).all():
                    raise ValueError(f"inf or nan in Lenv at {rl}")
                self.LR[rl - 1] = Renv
                rl -= 1
            self.rpos = k

    def lproj(self):
        if self.lpos <= -1:
            ndim = 3 if self.mid[0].ndim == 4 else 2
            return tc.tensor([1.], dtype=self.dtype, device=self.device).reshape(*[1]*ndim)
        return self.LR[self.lpos]
    
    def rproj(self):
        if self.rpos >= len(self.mid.data):
            ndim = 3 if self.mid[0].ndim == 4 else 2
            return tc.tensor([1.], dtype=self.dtype, device=self.device).reshape(*[1]*ndim)
        return self.LR[self.rpos]
     
    def set_position_(self, psi:MPS, pos:int):
        self.dtype = psi.data[0].dtype
        self.makeL_(psi, pos - 1)
        self.makeR_(psi, pos + self.nsite)

    @property
    def site_range(self):
        return range(self.lpos + 1, self.rpos)
    
    def set_nsite(self, nsite):
        self.nsite = nsite
    
    def __len__(self):
        return self.L


class ProjMPO(ProjOper):
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
        super().__init__(mid=H, nsite=nsite)

    _contract_left_env = staticmethod(tf._ProjMPO_contract_left_env)
    _contract_right_env = staticmethod(tf._ProjMPO_contract_right_env)

    def __matmul__(self, v:tc.Tensor) -> tc.Tensor:
        Lenv, H12, Renv = self.prepare_solve()
        if self.nsite == 0:
            return tf._matrix_vector_product0(Lenv, Renv, v)
        else:
            return tf._matrix_vector_product(Lenv, H12, Renv, v).reshape(*v.shape)

    def get_matmul_func(self, backend='torch'):
        Lenv, H12, Renv = self.prepare_solve()
        if backend == 'numpy':
            Lenv = Lenv.numpy()
            H12 = H12.numpy() if H12 is not None else None
            Renv = Renv.numpy()
        else:
            H12 = H12.to(dtype=self.dtype, device=self.device) if H12 is not None else None
        if self.nsite == 0:
            return lambda v: tf._matrix_vector_product0(Lenv, Renv, v)
        else:
            return lambda v: tf._matrix_vector_product(Lenv, H12, Renv, v)

    def prepare_solve(self):
        Lenv = self.lproj()
        Lenv = Lenv.reshape(Lenv.shape[0], -1).contiguous()
        
        if self.nsite == 2:
            H1 = self.mid.data[self.lpos + 1]
            H2 = self.mid.data[self.lpos + 2]
            H12 = tf._prepare_solve_ground_state(H1, H2).contiguous()
        elif self.nsite == 1:
            H12 = self.mid.data[self.lpos + 1]
            d, e, *_ = H12.shape
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

    @property
    def shape(self):
        dimnum = 1
        
        if self.lpos >= 0:
            Lenv = self.LR[self.lpos]
            dimnum *= Lenv.shape[0]
        
        for i in range(self.lpos + 1, self.rpos):
            dimnum *= self.mid.data[i].shape[1]
        
        if self.rpos < len(self.mid.data):
            Renv = self.LR[self.rpos]
            dimnum *= Renv.shape[0]
        
        return dimnum


class ProjMPS(ProjOper):
    def __init__(self, M, nsite=2) -> None:
        """
        ProjMPS 计算并存储 MPS 在由另一个 MPS 定义的基中投影，保留 MPS 的某些站点索引未投影。
        可以通过调用 `set_position_` 方法来移动未投影的格点。

        ProjMPS `PH` 表示的网络图示（`PH.set_position_(psi, 3)`）：

        .. code-block:: text
        
            o--o--o--o--o--o--o--o--o--o--o M
            |  |  |  |  |  |  |  |  |  |  |
            o--o--o-      -o--o--o--o--o--o |psi>
                  ↑        ↑
               lpos=2    rpos=5
        """
        assert nsite == 2, "Only two-site ProjMPS currently supported"
        super().__init__(mid=M, nsite=nsite)
       
    _contract_left_env = staticmethod(tf._ProjMPS_contract_left_env)
    _contract_right_env = staticmethod(tf._ProjMPS_contract_right_env)

    def prepare_solve(self):
        Lenv = self.lproj().T.contiguous()
        
        if self.nsite == 2:
            M1 = self.mid.data[self.lpos + 1]
            M2 = self.mid.data[self.lpos + 2]
            M12 = tf._projMPS_prepare_solve_ground_state(M1, M2).contiguous()
        elif self.nsite == 1:
            M12 = self.mid.data[self.lpos + 1]
            d, *_ = M12.shape
            M12 = M12.reshape(d, -1)
        elif self.nsite == 0:
            M12 = None
        
        Renv = self.rproj()
        return Lenv, M12, Renv
        
    def __matmul__(self, v:tc.Tensor) -> tc.Tensor:
        Lenv, M12, Renv = self.prepare_solve()
        if self.nsite > 0:
            pm = tf._projMPS_make_vec(Lenv, M12.conj(), Renv)
        else:
            pm = Lenv @ Renv
        scalar = pm.reshape(-1) @ v.reshape(-1)
        return scalar * pm.reshape(*v.shape).conj()
    
    def to_matrix(self):
        Lenv, M12, Renv = self.prepare_solve()
        if self.nsite > 0:
            pm = tf._projMPS_make_vec(Lenv, M12.conj(), Renv)
        else:
            pm = Lenv @ Renv
        return pm.reshape(-1, 1).conj() @ pm.reshape(1, -1)

    def get_matmul_func(self, backend='torch'):
        Lenv, M12, Renv = self.prepare_solve()
        if backend == 'numpy':
            Lenv = Lenv.numpy()
            M12 = M12.numpy() if M12 is not None else None
            Renv = Renv.numpy()
        if self.nsite > 0:
            pm = tf._projMPS_make_vec(Lenv, M12.conj(), Renv)
        else:
            pm = Lenv @ Renv
        return lambda v: (pm.reshape(-1) @ v.reshape(-1)) * pm.reshape(*v.shape).conj()


class SumMPO:
    def __init__(self, Hs:list[MPO]) -> None:
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
    
    def __iter__(self):
        return iter(self.Hs)
    
    @property
    def lognm(self):
        return tc.tensor(0.0, dtype=tc.float64)


class ProjSumMPO:
    def __init__(self, Hs:list[MPO]) -> None:
        """
        ProjMPO 计算并存储 MPO 在由 MPS 定义的基中投影，保留 MPO 的某些站点索引未投影。
        可以通过调用 `set_position_` 方法来移动未投影的格点。

        ProjMPO `PH` 表示的网络图示（`PH.set_position_(psi, 3)`）：

        .. code-block:: text
        
            o--o--o-      -o--o--o--o--o--o <psi|
            |  |  |  |  |  |  |  |  |  |  |
        Σ   o--o--o--o--o--o--o--o--o--o--o Hⱼ
            |  |  |  |  |  |  |  |  |  |  |
            o--o--o-      -o--o--o--o--o--o |psi>
                  ↑        ↑
               lpos=2    rpos=5
        """
        assert all(isinstance(H, MPO) for H in Hs), "Hs must be a list of MPOs"
        self.Hs = [ProjMPO(H, nsite=2) for H in Hs]

    def __matmul__(self, v:tc.Tensor) -> tc.Tensor:
        return reduce(lambda acc, H: H @ acc, self.Hs, v)

    def set_nsite(self, nsite):
        self.PH.set_nsite(nsite)
        for p in self.pm:
            p.set_nsite(nsite)

    def site_range(self):
        return self.PH.site_range()

    def set_position_(self, psi, pos):
        for H in self.Hs:
            H.set_position_(psi, pos)

    def get_matmul_func(self, backend='torch'):
        funcs = [H.get_matmul_func(backend) for H in self.Hs]
        # lognms = [tc.exp(H.mid.lognm).item() for H in self.Hs]
        # return lambda v: reduce(lambda x, pair: x + pair[0](v) * pair[1], zip(funcs, lognms), 0)
        def matmul(v):
            res = funcs[0](v) * tc.exp(self.Hs[0].mid.lognm)
            for i in range(1,len(funcs)):
                res += funcs[i](v)  * tc.exp(self.Hs[i].mid.lognm)
            return res
        return matmul

    def to_matrix(self):
        lognms = [tc.exp(H.mid.lognm).item() for H in self.Hs]
        return sum(coef*H.to_matrix() for H, coef in zip(self.Hs, lognms))

    @property
    def shape(self):
        return self.Hs[0].shape


class ProjMPOMPS:
    def __init__(self, H, mpsv):
        self.PH = ProjMPO(H)
        self.pm = [ProjMPS(m) for m in mpsv]

    def copy(self):
        return ProjMPOMPS(self.PH.copy(), [m.copy() for m in self.pm])

    @property
    def shape(self):
        return self.PH.shape

    def __len__(self):
        return len(self.PH)

    def set_nsite(self, nsite):
        self.PH.set_nsite(nsite)
        for p in self.pm:
            p.set_nsite(nsite)

    def site_range(self):
        return self.PH.site_range()

    def set_position_(self, psi, pos):
        self.PH.set_position_(psi, pos)
        for p in self.pm:
            p.set_position_(psi, pos)

    def get_matmul_func(self, backend='torch'):
        matmul1 = self.PH.get_matmul_func(backend) 
        matmul2 = [p.get_matmul_func(backend) for p in self.pm]
        def matmul(v):
            res = matmul1(v)
            for matmulp in matmul2:
                res += matmulp(v)
            return res
        return matmul

    def to_matrix(self):
        mat = self.PH.to_matrix()
        for p in self.pm:
            mat += p.to_matrix()
        return mat


class DMRG:
    def __init__(self, mpo, **kwargs):
        if isinstance(mpo, MPO):
            self.mpo = mpo
        elif isinstance(mpo, list) and all(isinstance(x, MPO) for x in mpo):
            self.mpo = SumMPO(mpo)
        else:
            raise ValueError(f"mpo 必须为 MPO 或列表, 目前为 {type(mpo).__name__}")

        self.Ms = kwargs.get('Ms', None)
        self.nsweep = kwargs.get('nsweep', 5)  # 扫描次数
        self.maxengnum = kwargs.get('maxengnum', 2)  # 判断收敛的时候使用的能量数目
        self.restol = kwargs.get('restol', 1.e-7)  # 收敛精度
        self.outputlevel = kwargs.get('outputlevel', 1)  # 输出级别

        # lanczos 参数
        self.backend = kwargs.get('backend', 'default')  # lanczos 后端
        self.max_trunc_err = kwargs.get('max_trunc_err', 1.e-14)  # lanczos 误差

        # mps 更新参数
        self.svd_alg = kwargs.get('svd_alg', 'eig')
        self.chi_max = kwargs.get('chi_max', [10, 20, 100, 100, 200])  # 奇异值分解的最大 bond dimension
        self.cutoff = kwargs.get('cutoff', [1E-10])
        self.svd_min = kwargs.get('svd_min', 1E-10)

        self.normalize = kwargs.get('normalize', True)  # 是否归一化

        self.N = len(self.mpo)
        assert self.N != 1, "MPS 长度不能为 1"

        # 初态
        self.psi = kwargs.get('psi0', None)
        if self.psi is None:
            dtype = kwargs.get('dtype', self.mpo.dtype)
            self.psi = MPS.from_random(self.N, bond_dim=2, dtype=dtype)
        self.psi.orthogonalize_(0)
        assert self.psi.llim == self.psi.rlim == 0 or self.psi.llim == self.psi.rlim == -1
        assert self.N == len(self.psi.data), 'MPS 和 MPO 的长度应该相等'

    def precheck(self):
        """检查参数的正确性"""
        if len(self.chi_max) > self.nsweep:
            self.chi_max = self.chi_max[:self.nsweep]
        elif len(self.chi_max) < self.nsweep:
            new_chi_max = [self.chi_max[-1]] * (self.nsweep - len(self.chi_max))
            self.chi_max.extend(new_chi_max)
        if len(self.cutoff) > self.nsweep:
            self.cutoff = self.cutoff[:self.nsweep]
        elif len(self.cutoff) < self.nsweep:
            new_cutoff = [self.cutoff[-1]] * (self.nsweep - len(self.cutoff))
            self.cutoff.extend(new_cutoff)

    def build_projH(self, nsite):
        """构建 projH"""
        oper = self.mpo.copy()
        if self.psi.dtype.is_complex and not self.mpo.dtype.is_complex:
            oper.to(dtype=tc.complex128,device=self.psi.device)
        if isinstance(self.mpo, MPO) and self.Ms is None:
            projH = ProjMPO(oper, nsite=nsite)
        elif isinstance(self.mpo, MPO) and isinstance(self.Ms, list) and all(isinstance(x, MPS) for x in self.Ms):
            projH = ProjMPOMPS(oper, self.Ms)
            assert nsite == 2, "nsite 必须为 2"
        elif isinstance(self.mpo, SumMPO) and self.Ms is None:
            projH = ProjSumMPO(oper)
            assert nsite == 2, "nsite 必须为 2"
        else:
            raise ValueError(
                f"""不支持的 mpo 和 Ms 类型:{
                    (type(self.mpo).__name__, type(self.Ms).__name__)
                    }, 允许的组合包括：
                    (MPO, NoneType), (MPO, list[MPS]), (SumMPO, NoneType)""")
        projH.set_position_(self.psi, 0)
        return projH

    def run1(self):
        nsite = 1
        self.precheck()
        projH = self.build_projH(nsite)
        energy_list = []
        for sw in range(self.nsweep):
            sw_time_start = time.time()
            for pos, drt in DMRG._sweep_schedule(self.N, nsite):
                # prepare_update_local
                self.psi.orthogonalize_(pos)
                projH.set_position_(self.psi, pos)
                phi = self.psi.data[pos]
                phi = phi/tc.norm(phi)
                # solve for the ground state of the effective Hamiltonian
                lanczos_tol=max(self.svd_min, 0.05*self.max_trunc_err) # todo `lanczos_tol` 有没有更好的选择？
                energy, phi = solve_ground_state(projH, phi, method=self.backend, lanczos_tol=lanczos_tol)
                # update the MPS
                self.psi.update_single_site_(pos, phi)
            sw_time = time.time() - sw_time_start  # 记录每步的时间
            # post process
            if self.outputlevel >= 1:
                print(f"After sweep {sw}: energy={(energy * tc.exp(self.mpo.lognm)).item()} maxchi={self.psi.maxbonddim()} time={sw_time:.3f}", flush=True)
            energy_list.append(energy)
            if len(energy_list) > self.maxengnum:
                energy_list.pop(0)
                if all(np.diff(energy_list) < self.restol):
                    break
        return energy * tc.exp(self.mpo.lognm), self.psi


    def run2(self):
        nsite = 2
        self.precheck()
        projH = self.build_projH(nsite)
        energy_list = []
        for sw in range(self.nsweep):
            sw_time_start = time.time()
            for pos, drt in DMRG._sweep_schedule(self.N, nsite):
                # prepare_update_local
                projH.set_position_(self.psi, pos)
                phi = tf._full_contract_right_mps2(self.psi.data[pos], self.psi.data[pos + 1])
                phi = phi/tc.norm(phi)
                # solve for the ground state of the effective Hamiltonian
                lanczos_tol=max(self.svd_min, 0.05*self.max_trunc_err) # todo `lanczos_tol` 有没有更好的选择？
                energy, phi = solve_ground_state(projH, phi, method=self.backend, lanczos_tol=lanczos_tol)
                # update the MPS
                trunc_para = (self.chi_max[sw], self.svd_min, self.cutoff[sw])
                self.psi.update_two_site_(pos, phi, direction=drt, svd_alg=self.svd_alg, trunc_para=trunc_para, normalize=self.normalize)
            sw_time = time.time() - sw_time_start  # 记录每步的时间
            # post process
            if self.outputlevel >= 1:
                print(f"After sweep {sw}: energy={(energy * tc.exp(self.mpo.lognm)).item()} maxchi={self.psi.maxbonddim()} time={sw_time:.3f}", flush=True)
            energy_list.append(energy)
            if len(energy_list) > self.maxengnum:
                energy_list.pop(0)
                if all(np.diff(energy_list) < self.restol):
                    break
        self.psi.lognm *= 0. # 归一
        return energy * tc.exp(self.mpo.lognm), self.psi

    @staticmethod
    def _sweep_schedule(L, nsite):
        for position in range(L - nsite):
            yield (position, "right")
        for position in range(L - nsite, -1, -1):
            yield (position, "left") 

class TDVP:
    def __init__(self, mpo, psi0, time_step, final_time, **kwargs):
        self.mpo = mpo
        self.init = psi0
        self.time_step = time_step
        self.final_time = final_time

        self.start_time = kwargs.get('start_time', 0.0)
        self.cur_time = 0.0
        self.cur_state = psi0.copy()
        self.sweepn = None

        self.chi_max = kwargs.get('chi_max', [20])
        self.trunc_cut = kwargs.get('trunc_cut', [1E-10])
        self.backend = kwargs.get('backend', 'default') # 计算有效哈密顿量基态的方法，"default", "expm_multiply", "lanczos"
        self.svd_alg = kwargs.get('svd_alg', 'svd')  #  update MPS 中张量的方法
        self.normalize = kwargs.get('normalize', True)  # 是否归一化
        self.reverse_step = kwargs.get('reverse_step', True) # 时间演化必须是 True，虚时演化 False 回到 DMRG
        self.nsite = kwargs.get('nsite', 2)  # nsite = 1 不改变 bond dimension, nsite = 2 可以改变 bond dimension
        self.order = kwargs.get('order', 2)  # 时间演化的阶数
        self.outputlevel = kwargs.get('outputlevel', 1)  # 输出等级，0 表示不输出中间信息

    def precheck(self, nsweeps):
        """检查参数的正确性"""
        # if np.abs(nsweeps*self.time_step - self.final_time) > 1e-10:
        #     raise ValueError(f"t / time_step = {self.final_time} / {self.time_step} = {self.final_time/self.time_step} 必须是整数")

        N = len(self.init.data)
        assert N == len(self.mpo.data), 'MPS 和 MPO 的长度应该相等'
        if N == 1:
            raise Exception("长度 1 的 MPS 暂不支持，可以用 `numpy.linalg.eigh`, `scipy.eigsh` 等方法求解")
        if len(self.chi_max) > nsweeps:
            self.chi_max = self.chi_max[:nsweeps]
        elif len(self.chi_max) < nsweeps:
            new_chi_max = [self.chi_max[-1]] * (nsweeps - len(self.chi_max))
            self.chi_max.extend(new_chi_max)
        if len(self.trunc_cut) > nsweeps:
            self.trunc_cut = self.trunc_cut[:nsweeps]
        elif len(self.trunc_cut) < nsweeps:
            new_cutoff = [self.trunc_cut[-1]] * (nsweeps - len(self.trunc_cut))
            self.trunc_cut.extend(new_cutoff)

    def run(self) -> Generator[tuple[Union[float, complex], MPS], None, None]:
        # 参数检查
        nsweeps = int(np.real(self.final_time/self.time_step))
        self.precheck(nsweeps)
        # 初态设置
        if np.iscomplex(self.time_step):
            if not self.cur_state.dtype.is_complex:
                self.cur_state.to(dtype=tc.complex128)
        # 算符设置
        reduced_operator = ProjMPO(self.mpo, nsite=self.nsite)
        for sweep in range(nsweeps):
            self.sweepn = sweep
            sweep_elapsed_time = time.time()
            # --------------main-------------
            self.sweep(reduced_operator)
            # -------------end main-----------
            sweep_elapsed_time = time.time() - sweep_elapsed_time
            if self.cur_time.real == 0:
                if self.outputlevel >= 1:
                    print(f"--> time: {round(- self.cur_time.imag,3)}: maxchi={self.cur_state.maxbonddim()} elapsed_time={sweep_elapsed_time:.3f}", flush=True)
                yield - self.cur_time.imag, self.cur_state
            else:
                if self.outputlevel >= 1:
                    print(f"--> time: {self.cur_time/1j:.3f}: maxchi={self.cur_state.maxbonddim()} elapsed_time={sweep_elapsed_time:.3f}", flush=True)
                yield self.cur_time/1j, self.cur_state
    

    def sweep(self, reduced_operator):
        # todo order = 1, 4 的实现
        order_orderings = ['forward', 'backward']
        order_sub_time_steps = self.sub_time_steps(self.order)
        order_sub_time_steps = [i * self.time_step for i in order_sub_time_steps]
        maxtruncerr = 0.0

        # subtime iteration
        for subtime, sub_time_step in enumerate(order_sub_time_steps):
            direction = order_orderings[subtime % 2]

            N = len(self.cur_state.data)
            nsite = reduced_operator.nsite

            if direction == 'forward':
                if self.cur_state.rlim != self.cur_state.llim or self.cur_state.llim != 0:
                    self.cur_state.orthogonalize_(0)
                assert self.cur_state.rlim == self.cur_state.llim == 0
                reduced_operator.set_position_(self.cur_state, 0)
            elif direction == 'backward':
                if self.cur_state.rlim != self.cur_state.llim or self.cur_state.rlim != N - nsite:
                    self.cur_state.orthogonalize_(N - nsite)
                assert self.cur_state.rlim == self.cur_state.llim == N - nsite
                reduced_operator.set_position_(self.cur_state, N - nsite)
            else:
                raise ValueError(f"direction = {direction} 不合法")

            # site iteration
            for b in self.sweep_bonds(reduced_operator.nsite, direction, N):
                trunc_err = self.tdvp_sweep_local_(
                    reduced_operator,  # 有效哈密顿量
                    b,  # 优化 b 位置的张量
                    sub_time_step, # 时间步长
                    direction, # 方向 'forward' 或 'backward'
                    maxtruncerr=maxtruncerr  # 最大的误差
                    )
                if maxtruncerr < trunc_err.eps:
                    maxtruncerr = trunc_err.eps
        return maxtruncerr


    def tdvp_sweep_local_(
            self,
            reduced_operator,  # 有效哈密顿量
            b,  # 优化 b 位置的张量
            time_step, # 时间步长
            direction, # 方向 'forward' 或 'backward'
            maxtruncerr # 最大的误差
    ):
        # todo 整理这段代码
        nsite = reduced_operator.nsite
        cutoff = self.trunc_cut[self.sweepn]
        if nsite == 2:

            reduced_operator.set_position_(self.cur_state, b)
            reduced_state = tf._full_contract_right_mps2(self.cur_state.data[b], self.cur_state.data[b + 1])

            reduced_state = solve_evolve_state(reduced_operator, reduced_state, time_step, method=self.backend, lanczos_tol=max(cutoff, 0.05*maxtruncerr))

            self.cur_time += time_step
            if self.normalize:
                nm = tc.norm(reduced_state)
                reduced_state = reduced_state / nm
                self.cur_state.lognm += tc.log(nm)
            ortho = "right" if direction == "forward" else "left"

            trunc_err = self.cur_state.update_two_site_(b, reduced_state, ortho, svd_alg=self.svd_alg, trunc_para=(self.chi_max[self.sweepn], cutoff, None))

            maxtruncerr = max(maxtruncerr, trunc_err.eps)

            if not self.is_half_sweep_done(reduced_operator.nsite, direction, b, len(self.cur_state.data)) and self.reverse_step:
                # Do backwards evolution step
                b1 = b + 1 if direction == "forward" else b

                bond_reduced_state = self.cur_state.data[b1]
                reduced_operator.nsite = nsite - 1
                reduced_operator.set_position_(self.cur_state, b1)

                bond_reduced_state = solve_evolve_state(reduced_operator, bond_reduced_state, -time_step, method=self.backend, lanczos_tol=max(cutoff, 0.05*maxtruncerr))

                self.cur_time -= time_step
                if self.normalize:
                    nm = tc.norm(bond_reduced_state)
                    bond_reduced_state = bond_reduced_state / nm
                    self.cur_state.lognm += tc.log(nm)

                self.cur_state.update_single_site_(b1, bond_reduced_state)

                reduced_operator.nsite = nsite
            return trunc_err

        # todo nsite = 1 以及 not reverse_step 的实现
        elif nsite == 1:
            reduced_operator.set_position_(self.cur_state, b)

            reduced_state = self.cur_state.data[b].clone()

            reduced_state = solve_evolve_state(reduced_operator, reduced_state, time_step, method=self.backend, lanczos_tol=max(cutoff, 0.05*maxtruncerr))

            self.cur_time += time_step
            if self.normalize:
                nm = tc.norm(reduced_state)
                reduced_state = reduced_state / nm
                self.cur_state.lognm += tc.log(nm)

            ortho = "right" if direction == "forward" else "left"

            self.cur_state.update_single_site_(b, reduced_state)

            maxtruncerr = maxtruncerr

            if not self.is_half_sweep_done(direction, b, len(self.cur_state.data)):
                # Do backwards evolution step
                if self.reverse_step:
                    if direction == "forward":
                        b1 = b + 1
                        U, S, V, _ = svd(reduced_state, lr_indx=[[0,1],[2]])
                        self.cur_state.update_single_site_(b, U)
                        bond_reduced_state = S.reshape(-1,1) * V
                        self.cur_state.llim += 1

                    elif direction == "backward":
                        b1 = b
                        U, S, V, _ = svd(reduced_state, lr_indx=[[0],[1,2]])
                        self.cur_state.update_single_site_(b, V)
                        bond_reduced_state = S.reshape(1,-1) * U
                        self.cur_state.rlim -= 1

                    reduced_operator.nsite = nsite - 1
                    reduced_operator.set_position_(self.cur_state, b1)

                    bond_reduced_state = solve_evolve_state(reduced_operator, bond_reduced_state, -time_step, method=self.backend, lanczos_tol=max(cutoff, 0.05*maxtruncerr))

                    self.cur_time -= time_step
                    if self.normalize:
                        nm = tc.norm(bond_reduced_state)
                        bond_reduced_state = bond_reduced_state / nm
                        self.cur_state.lognm += tc.log(nm)

                    if direction == "forward":
                        nexttsr = self.cur_state.data[b + 1]
                        nexttsrshape = nexttsr.shape
                        self.cur_state.data[b + 1] = (bond_reduced_state @ nexttsr.reshape(nexttsrshape[0], -1)).reshape(-1, *nexttsrshape[1:])
                        self.cur_state.rlim += 1

                    elif direction == "backward":
                        nexttsr = self.cur_state.data[b - 1]
                        nexttsrshape = nexttsr.shape
                        self.cur_state.data[b - 1] = (nexttsr.reshape(-1, nexttsrshape[-1]) @ bond_reduced_state).reshape(*nexttsrshape[:-1],-1)
                        self.cur_state.llim -= 1

                    reduced_operator.nsite = nsite
                else:
                    # only move ortho center
                    if direction == "forward":
                        self.cur_state.orthogonalize_(b + 1)
                    elif direction == "backward":
                        self.cur_state.orthogonalize_(b - 1)

            return TruncationError(0.0, 1.0)

    def sub_time_steps(self, order):
        if order == 1:
            return [1.0, 0.0]
        elif order == 2:
            return [0.5, 0.5]
        elif order == 4:
            s = 1 / (2 - 2**(1/3))
            return [s/2, s/2, (1 - 2*s)/2, (1 - 2*s)/2, s/2, s/2]

    def sweep_bonds(self, nsite, direction, N):
        if direction == 'forward':
            return range(N - nsite + 1)
        elif direction == 'backward':
            return range(N - nsite, -1, -1)
        else:
            raise ValueError(f"direction = {direction} 不合法")


    def is_half_sweep_done(self, nsite, direction, b, N):
        return (direction == "forward" and b == N - nsite) or \
               (direction == "backward" and b == 0)

