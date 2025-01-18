# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 15:45:13
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-18 16:14:56

import time  # type: ignore
import torch as tc
import numpy as np

from .mps import MPS
from .mpo import MPO, SumMPO
from .projtt import (ProjMPO, ProjMPOMPS, ProjSumMPO,
                     solve_ground_state)
from . import tensor_operations as tf


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
                phi = tf._full_contract_two(self.psi.data[pos], self.psi.data[pos + 1])
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

