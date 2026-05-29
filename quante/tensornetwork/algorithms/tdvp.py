# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 15:45:38
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-29 12:18:03

import time  # type: ignore
import numpy as np
from typing import Generator, Union

from ..networks.mps import MPS
from .projections import ProjMPO
from ..core import tensor_operations as tf
from ...linalg.krylov.toy import lanczos_evolve_state
from ...linalg import svd, expm_multiply
from ...linalg.decomp.svd_robust import TruncationError
import scipy.linalg


def solve_evolve_state(oper:'ProjMPO', v, delta, *, method='default', lanczos_tol=1e-14):
    """计算 ProjMPO 的 evolve state """
    if method == 'default':
        # use ED for small matrix dimensions, but lanczos by default
        if oper.shape < 400:
            mat = oper.to_matrix()
            E, theta = scipy.linalg.eigh(mat)
            expE = np.exp(E * delta)
            exp_dH_v = theta @ (expE * (theta.conj().T @ v.reshape(-1)))
            return exp_dH_v.reshape(*v.shape)
        else:
            method = 'lanczos'
            # method = 'expm_multiply'
            # matmul = oper.get_matmul_func()
            # vec = lanczos_evolve_state(matmul, v.reshape(-1), delta, P_tol=lanczos_tol)
            # return vec.reshape(*v.shape)
    if method == 'lanczos':  # 使用 lanczos 进行演化
        matmul = oper.get_matmul_func()
        vec = lanczos_evolve_state(matmul, v.reshape(-1), delta, tol=lanczos_tol)
        return vec.reshape(*v.shape)
    elif method == 'expm_multiply':  # 使用 expm_multiply 进行演化
        Lenv, H12, Renv = oper.prepare_solve()
        if oper.nsite == 0:
            matmul = lambda inipsi: delta * tf._matrix_vector_product0(Lenv, Renv, inipsi)
            rmatmul = lambda inipsi: np.conj(delta) * tf._matrix_vector_product0(Lenv, Renv, inipsi)
            trmatul = tf._trace_matrix_vector_product0(Lenv, Renv) * delta
        else:
            matmul = lambda inipsi: delta * tf._matrix_vector_product(Lenv, H12, Renv, inipsi)
            rmatmul = lambda inipsi: np.conj(delta)* tf._matrix_vector_product(Lenv, H12, Renv, inipsi)
            trmatul = tf._trace_matrix_vector_product(Lenv, H12, Renv) * delta
        if (np.iscomplexobj(delta) or np.iscomplexobj(Lenv) or np.iscomplexobj(H12) \
            or np.iscomplexobj(Renv)) and np.isrealobj(v):
            v = v.astype(np.complex128)
        res = expm_multiply(matmul, v.reshape(-1), traceA=trmatul, herm=rmatmul)
        return res.reshape(*v.shape)
    else:
        raise ValueError(f"Unknown backend: {method}")

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
            reduced_state = tf._full_contract_two(self.cur_state.data[b], self.cur_state.data[b + 1])

            reduced_state = solve_evolve_state(reduced_operator, reduced_state, time_step, method=self.backend, lanczos_tol=max(cutoff, 0.05*maxtruncerr))

            self.cur_time += time_step
            if self.normalize:
                nm = np.linalg.norm(reduced_state)
                reduced_state = reduced_state / nm
                self.cur_state.lognm += np.log(nm)
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
                    nm = np.linalg.norm(bond_reduced_state)
                    bond_reduced_state = bond_reduced_state / nm
                    self.cur_state.lognm += np.log(nm)

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
                nm = np.linalg.norm(reduced_state)
                reduced_state = reduced_state / nm
                self.cur_state.lognm += np.log(nm)

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
                        nm = np.linalg.norm(bond_reduced_state)
                        bond_reduced_state = bond_reduced_state / nm
                        self.cur_state.lognm += np.log(nm)

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


