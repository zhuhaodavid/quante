# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 15:45:13
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-23 01:43:54

import time  # type: ignore
from tqdm import tqdm
import warnings

from ...linalg.krylov.toy import lanczos_arpack, lanczos_ground_state, arnoldi_ground_state
from ...linalg.krylov.eigsolve.arnoldi import Arnoldi
from ...linalg.krylov.eigsolve.lanczos import Lanczos
from ..core.tensor_utils import argsort
import numpy as np

from ..networks.mps import MPS
from ..networks.mpo import MPO, SumMPO
from .projections import (ProjMPO, ProjMPOMPS, ProjSumMPO, ProjOper)
from ..core import tensor_operations as tf


def solve_ground_state(oper:'ProjMPO', v, *,
                       method='default', 
                       which='LM', 
                       isherm=True,
                       lanczos_tol=1e-14,
                       refer=None,
                       **eigs_kwargs):
    """计算 ProjMPO 的 ground state """
    if method == 'default':
        if isherm:
            # use ED for small matrix dimensions, but lanczos by default
            if oper.shape < 400:
                mat = oper.to_matrix()
                E, theta = np.linalg.eigh(mat)
                sort = argsort(E, which, refer=refer)
                return E[sort[0]], theta[:, sort[0]].reshape(*v.shape)
            else:
                method = 'lanczos'
        else:
            # use ED for small matrix dimensions, but lanczos by default
            if oper.shape < 400:
                mat = oper.to_matrix()
                E, theta = np.linalg.eig(mat)
                sort = argsort(E, which, refer=refer)
                # print(E[sort[:10]])
                return E[sort[0]], theta[:, sort[0]].reshape(*v.shape)
            else:
                method = 'arnoldi'

    s = v.shape
    matmul = oper.get_matmul_func()
    x0 = v.reshape(-1)

    if method == 'lanczos':  # 自己实现的 lanczos
        assert which in ['LM', 'LR', 'SR'], f'which: {which} not allowed, should be in [LM, LR, SR]'
        # val, vec, _ = Lanczos(tol=lanczos_tol, maxiter=1, krylovdim=3, verbosity=0, **eigs_kwargs).eigsolve(matmul, x0, 1, which, lau=None) 
        # return val[0], vec[0].reshape(*s)
        val, vec = lanczos_ground_state(matmul, v.reshape(-1), E_tol=lanczos_tol,**eigs_kwargs)
        return val, vec.reshape(*s)

    elif method == 'arnoldi':  # tenpy arnoldi 用来处理非厄密矩阵时可以考虑 #todo 自己实现
        assert which in ['LM', 'LR', 'SR', 'LI', 'SI']
        # val, vec, _ = Arnoldi(maxiter=1, krylovdim=3, verbosity=0, **eigs_kwargs).eigsolve(matmul, x0, 2, which, lau=None)
        # return val[0], vec[0].reshape(*s)
        val, vec = arnoldi_ground_state(matmul, v.reshape(-1),refer=refer, which=which, **eigs_kwargs)  # tol 并没有用
        return val[0], vec[0].reshape(*s)
    
    elif method == 'larpack': # scipy sparse eigs
        val, vec = lanczos_arpack(matmul, v.numpy().reshape(-1), tol=lanczos_tol,
                                  which=which, **eigs_kwargs)
        return val, vec
   
    else:
        raise ValueError(f"Unknown method: {method}")

class DMRG:
    def __init__(self, mpo, **kwargs):
        # 保存 mpo 数据
        if isinstance(mpo, MPO) or isinstance(mpo, ProjOper) or isinstance(mpo, ProjMPOMPS) or isinstance(mpo, ProjSumMPO):
            # 如果输入 mpo，那就保存这个 mpo
            self.mpo = mpo  
        elif (isinstance(mpo, list) and 
              all(isinstance(x, MPO) for x in mpo)):
            # 如果输入 list of mpo，那就打包成 SumMPO，
            # 这样可以方便的调用 len(self.mpo) 等方法
            self.mpo = SumMPO(mpo)  
        else:
            raise ValueError(f"mpo 必须为 MPO 或列表, 目前为 {type(mpo).__name__}")
        
        # 链长
        self.L = len(self.mpo)
        assert self.L != 1, "MPS 长度不能为 1"

        # 记录已使用的初态
        used_kwargs = set()
        
        # 初态
        self.psi = kwargs.get('psi', None)
        used_kwargs.add('psi')

        if self.psi is None:
            # 如果没有指定初态，那就随机出来，数据类型与 self.mpo 相同
            dtype = kwargs.get('dtype', self.mpo.dtype)
            used_kwargs.add('dtype')
            self.psi = MPS.from_random(self.L, bond_dim=2, phys_dim=self.mpo.phys_dim, dtype=dtype)
        self.psi.orthogonalize_(0)
        assert self.L == len(self.psi.data), 'MPS 和 MPO 的长度应该相等'

        # 要排除的子空间基矢
        self.Ms = kwargs.get('Ms', None)
        used_kwargs.add('Ms')
        self.weight = kwargs.get('weight', None if self.Ms is None else [1.0]*len(self.Ms))
        used_kwargs.add('weight')
        assert self.weight is None or isinstance(self.weight, list), "weight 必须为 list"

        # DMRG 参数
        self.nsweep = kwargs.get('nsweep', 5)  # 扫描次数
        used_kwargs.add('nsweep')
        self.restol = kwargs.get('restol', 1.e-7)  # 能量收敛精度
        used_kwargs.add('restol')
        self.outputlevel = kwargs.get('outputlevel', 2)  # 输出级别
        used_kwargs.add('outputlevel')
        self.which = kwargs.get('which', 'SR')  # 计算基态还是边界态
        used_kwargs.add('which')
        self.isherm = kwargs.get('isherm', True)  # 是否厄密
        used_kwargs.add('isherm')
        self.ifnorm = kwargs.get('normenv', False)  # 环境是否归一
        used_kwargs.add('normenv')

        # lanczos 参数
        self.backend = kwargs.get('backend', 'default')  # lanczos 后端
        used_kwargs.add('backend')
        # 'lanczos', 'arnoldi', 'larpack'
        self.max_trunc_err = kwargs.get('max_trunc_err', 1.e-14)  # lanczos 误差
        used_kwargs.add('max_trunc_err')
        
        self.eigs_kwargs = kwargs.get('eigs_kwargs', {})
        used_kwargs.add('eigs_kwargs')

        # mps 更新参数
        self.noise = kwargs.get('noise', None)  # 噪声
        used_kwargs.add('noise')
        self.svd_alg = kwargs.get('svd_alg', 'eig')
        used_kwargs.add('svd_alg')
        self.chi_max = kwargs.get('chi_max', [10, 20, 100, 100, 200])  # 奇异值分解的最大 bond dimension
        used_kwargs.add('chi_max')
        self.cutoff = kwargs.get('cutoff', [1E-10])
        used_kwargs.add('cutoff')
        self.svd_min = kwargs.get('svd_min', 1E-10)
        used_kwargs.add('svd_min')
        self.normalize = kwargs.get('normalize', True)  # 是否归一化
        used_kwargs.add('normalize')

        # 检查是否有未使用的参数
        unused_kwargs = set(kwargs.keys()) - used_kwargs
        if unused_kwargs:
            warnings.warn(f"未使用的参数: {unused_kwargs}")

        # 记录结果：
        self.energy = np.inf
        self.sw = 0

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

    def build_projH(self, nsite) -> ProjMPO | ProjMPOMPS | ProjSumMPO:
        """构建 projH"""
        if isinstance(self.mpo, ProjOper) or isinstance(self.mpo, ProjMPOMPS) or isinstance(self.mpo, ProjSumMPO):
            return self.mpo

        oper = self.mpo.copy()
        if isinstance(self.mpo, MPO) and self.Ms is None:
            projH = ProjMPO(oper, nsite=nsite, ifnorm=self.ifnorm)
        elif isinstance(self.mpo, MPO) and isinstance(self.Ms, list) and all(isinstance(x, MPS) for x in self.Ms):
            projH = ProjMPOMPS(oper, self.Ms, self.weight, ifnorm=self.ifnorm)
            assert nsite == 2, "nsite 必须为 2"
        elif isinstance(self.mpo, SumMPO) and self.Ms is None:
            projH = ProjSumMPO(oper, ifnorm=self.ifnorm)
            assert nsite == 2, "nsite 必须为 2"
        else:
            raise ValueError(
                f"""不支持的 mpo 和 Ms 类型:{
                    (type(self.mpo).__name__, type(self.Ms).__name__)
                    }, 允许的组合包括：
                    (MPO, NoneType), (MPO, list[MPS]), (SumMPO, NoneType)""")
        # projH.set_position_(self.psi, 0)  # 是否需要？
        return projH

    def run1(self):
        nsite = 1
        self.precheck()
        projH = self.build_projH(nsite)
        for sw in range(self.nsweep):
            sw_time_start = time.time()
            for pos, drt in DMRG._sweep_schedule(self.L, nsite):
                # prepare_update_local
                self.psi.orthogonalize_(pos)
                projH.set_position_(self.psi, pos)
                phi = self.psi.data[pos]
                phi = phi/np.norm(phi)
                # solve for the ground state of the effective Hamiltonian
                lanczos_tol=max(self.svd_min, 0.05*self.max_trunc_err) # todo `lanczos_tol` 有没有更好的选择？
                energy, phi = solve_ground_state(projH, phi, method=self.backend, lanczos_tol=lanczos_tol)
                # update the MPS
                self.psi.update_single_site_(pos, phi)
            energy *= np.exp(self.mpo.lognm + projH.lrlognm)
            sw_time = time.time() - sw_time_start  # 记录每步的时间
            self.logstate(sw, sw_time, energy)
            if self.checkdone(energy):
                break
        return energy * np.exp(self.mpo.lognm), self.psi
    

    def run2(self):
        self.convergent = False
        nsite = 2
        self.precheck()
        projH = self.build_projH(nsite)
        # save_hdf5("log.h5", f"init", {f"{i}": self.psi.data[i].reshape(-1) for i in range(self.psi.L)}, mode='w')
        energy = self.energy
        for sw in range(self.nsweep):
            
            # 仅当 self.outputlevel >= 1 时显示进度条
            if self.outputlevel >= 2:
                pbar = tqdm(total=2 * self.L - 2, desc=f"Sweep {sw+1}", dynamic_ncols=True, ascii=True)
            else:
                pbar = None

            for pos, drt in DMRG._sweep_schedule(self.L, nsite):
                projH.set_position_(self.psi, pos)
                
                phi = tf._full_contract_two(self.psi.data[pos], self.psi.data[pos + 1])
                phi /= np.linalg.norm(phi)
           
                lanczos_tol=max(self.svd_min, 0.05*self.max_trunc_err) # todo `lanczos_tol` 有没有更好的选择？
                energy, phi = solve_ground_state(projH, phi, 
                            method=self.backend,
                            lanczos_tol=lanczos_tol,
                            which=self.which,
                            isherm=self.isherm,
                            refer=energy,
                            **self.eigs_kwargs)
            
                drho = self.noise * projH.noiseterm(phi=phi, drt=drt) if self.noise is not None else None

                trunc_para = (self.chi_max[sw], self.svd_min, self.cutoff[sw])
                self.psi.update_two_site_(pos, phi, drt,
                                        svd_alg=self.svd_alg, trunc_para=trunc_para, normalize=self.normalize,
                                        eigdirection=drt, pertube=drho, updateS=True)
                
                if drt == 'left' and pos == self.L // 2:
                    real_energy = energy * np.exp(self.mpo.lognm + projH.lrlognm)

                if pbar is not None:
                    pbar.set_postfix({"pE": f"{energy:.4e}", "chi": self.psi.maxbonddim()})
                    # pE is the abbreviation of the pseudo energy
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            
            if self.checkdone(real_energy):
                if self.outputlevel >= 1:
                    print(f"Energy converged to {real_energy:.10f} after {sw + 1} sweeps.")
                self.convergent = True
                break
        if not self.convergent:
            print(f"Energy did not converge after {self.nsweep} sweeps.")
        self.sw = sw
        self.psi.lognm *= 0. # 归一
        return self.energy, self.psi

    @staticmethod
    def _sweep_schedule(L, nsite):
        for position in range(L - nsite + 1):
            yield (position, "right")
        for position in range(L - nsite, -1, -1):
            yield (position, "left") 
        # for position in range(L - nsite + 1):
        #     yield (position, "right")
        # for position in range(L - nsite, -1, -1):
        #     yield (position, "left") 
    
    def checkdone(self, energy):
        isdone = abs(self.energy - energy) < self.restol * abs(energy)
        self.energy = energy
        return isdone
