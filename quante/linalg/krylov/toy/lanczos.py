# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-23 01:25:18

from __future__ import annotations

from typing import Callable
import scipy.sparse.linalg as spalg
import numpy as np

__all__ = [
    "lanczos_ground_state", 
    "lanczos_ground_state2", 
    "lanczos_evolve_state", 
    "lanczos_evolve_state2", 
    "lanczos_arpack", 
]

def _eps(dtype) -> float:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        dtype = np.empty((), dtype=dtype).real.dtype
    return np.finfo(dtype).eps


class LanczosGeneral:
    def __init__(self, matvec, psi0, **kwargs):
        """
        来自：
        https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/lanczos.py
        中对 lanczos 的实现

        只能处理实对称矩阵。NumPy 版本只做前向计算，不支持梯度。
        """
        if isinstance(matvec, np.ndarray):
            self.tsr = matvec
            self.matvec = matvec.__matmul__
        else:
            self.tsr = None
            self.matvec = matvec

        self.init_vecs = np.asarray(psi0)

        self.dim = self.init_vecs.shape[0]
        self.matrix_shape = (self.dim, self.dim)

        self.max_iter = kwargs.get("max_iter", 20)
        self.tol = kwargs.get("tol", 1.0e-14)
        self.batch_shape = kwargs.get("batch_shape", None)
        self.jitter_val = kwargs.get("jitter_val", None)

    def run(self):
        self.tridiag()
        return self.eigenvalues, self.eigenvectors

    def tridiag(self):
        matmul_closure = self.matvec
        matrix_shape = self.matrix_shape
        batch_shape = () if self.batch_shape is None else tuple(self.batch_shape)
        max_iter = self.max_iter
        init_vecs = self.init_vecs
        tol = self.tol

        # Determine batch mode
        multiple_init_vecs = False

        # Get initial probe vectors - and define if not available
        if init_vecs is None:
            num_init_vecs = 1
            init_vecs = np.random.randn(*batch_shape, matrix_shape[-1], num_init_vecs)
        else:
            if init_vecs.ndim == 1:
                init_vecs = init_vecs.reshape(-1, 1)
            num_init_vecs = init_vecs.shape[-1]

        # Define some constants
        num_iter = min(max_iter, matrix_shape[-1])
        dim_dimension = -2

        # Create storage for q_mat, alpha,and beta
        q_mat = np.zeros(
            (num_iter, *batch_shape, matrix_shape[-1], num_init_vecs),
            dtype=np.float64,
        )
        t_mat = np.zeros((num_iter, num_iter, *batch_shape, num_init_vecs), dtype=np.float64)

        # Begin algorithm
        # Initial Q vector: q_0_vec
        q_0_vec = init_vecs / np.linalg.norm(init_vecs, 2, axis=dim_dimension, keepdims=True)
        q_mat[0] = q_0_vec

        # Initial alpha value: alpha_0
        r_vec = matmul_closure(q_0_vec)
        alpha_0 = np.sum(q_0_vec * r_vec, axis=dim_dimension)

        # Initial beta value: beta_0
        r_vec = r_vec - np.expand_dims(alpha_0, dim_dimension) * q_0_vec
        beta_0 = np.linalg.norm(r_vec, 2, axis=dim_dimension)

        # Copy over alpha_0 and beta_0 to t_mat
        t_mat[0, 0] = alpha_0
        if num_iter > 1:
            t_mat[0, 1] = beta_0
            t_mat[1, 0] = beta_0
            q_mat[1] = r_vec / np.expand_dims(beta_0, dim_dimension)

        # Now we start the iteration
        k = 0
        for k in range(1, num_iter):
            # Get previous values
            q_prev_vec = q_mat[k - 1]
            q_curr_vec = q_mat[k]
            beta_prev = np.expand_dims(t_mat[k, k - 1], dim_dimension)

            # Compute next alpha value
            r_vec = matmul_closure(q_curr_vec) - q_prev_vec * beta_prev
            alpha_curr = np.sum(q_curr_vec * r_vec, axis=dim_dimension, keepdims=True)
            # Copy over to t_mat
            t_mat[k, k] = np.squeeze(alpha_curr, axis=dim_dimension)

            # Copy over alpha_curr, beta_curr to t_mat
            if (k + 1) < num_iter:
                # Compute next residual value
                r_vec = r_vec - alpha_curr * q_curr_vec
                # Full reorthogonalization: r <- r - Q (Q^T r)
                correction = np.sum(np.expand_dims(r_vec, 0) * q_mat[: k + 1], axis=dim_dimension, keepdims=True)
                correction = np.sum(q_mat[: k + 1] * correction, axis=0)
                r_vec = r_vec - correction
                r_vec_norm = np.linalg.norm(r_vec, 2, axis=dim_dimension, keepdims=True)
                r_vec = r_vec / r_vec_norm

                # Get next beta value
                beta_curr = np.squeeze(r_vec_norm, axis=dim_dimension)
                # Update t_mat with new beta value
                t_mat[k, k + 1] = beta_curr
                t_mat[k + 1, k] = beta_curr

                # Run more reorthoganilzation if necessary
                inner_products = np.sum(q_mat[: k + 1] * np.expand_dims(r_vec, 0), axis=dim_dimension)
                could_reorthogonalize = False
                for _ in range(10):
                    if not np.sum(inner_products > tol):
                        could_reorthogonalize = True
                        break
                    correction = np.sum(np.expand_dims(r_vec, 0) * q_mat[: k + 1], axis=dim_dimension, keepdims=True)
                    correction = np.sum(q_mat[: k + 1] * correction, axis=0)
                    r_vec = r_vec - correction
                    r_vec_norm = np.linalg.norm(r_vec, 2, axis=dim_dimension, keepdims=True)
                    r_vec = r_vec / r_vec_norm
                    inner_products = np.sum(q_mat[: k + 1] * np.expand_dims(r_vec, 0), axis=dim_dimension)

                # Update q_mat with new q value
                q_mat[k + 1] = r_vec

                if np.sum(np.abs(beta_curr) > 1e-6) == 0 or not could_reorthogonalize:
                    break

        # Now let's transpose q_mat, t_mat into the correct shape
        num_iter = k + 1

        # num_init_vecs x batch_shape x matrix_shape[-1] x num_iter
        q_mat = np.transpose(q_mat[:num_iter], (-1, *range(1, 1 + len(batch_shape)), -2, 0)).copy()
        # num_init_vecs x batch_shape x num_iter x num_iter
        t_mat = np.transpose(t_mat[:num_iter, :num_iter], (-1, *range(2, 2 + len(batch_shape)), 0, 1)).copy()

        # If we weren't in batch mode, remove batch dimension
        if not multiple_init_vecs:
            q_mat = np.squeeze(q_mat, axis=0)
            t_mat = np.squeeze(t_mat, axis=0)

        self.q_mat = q_mat
        self.t_mat = t_mat
        eigenvalues, eigenvectors = self.tridiag2diag(t_mat)
        self.eigenvalues = eigenvalues
        self.eigenvectors = q_mat @ eigenvectors
        return q_mat, t_mat

    def tridiag2diag(self, t_mat):
        evals, evecs = np.linalg.eigh(t_mat)
        # todo: 只取正数部分是不是计算梯度的要求？
        mask = evals >= 0
        evecs = evecs * np.expand_dims(mask.astype(evecs.dtype), -2)
        evals = np.where(mask, evals, 1)
        return evals, evecs


class LanczosGroundState(LanczosGeneral):
    def __init__(self, matvec, psi0, **kwargs):
        """在 LanczosGeneral 的基础上，针对基态做了改动。"""
        super().__init__(matvec, psi0, **kwargs)
        self.E_tol = kwargs.get("E_tol", np.inf)
        self.min_gap = kwargs.get("min_gap", 1.0e-12)
        self.cutoff = kwargs.get("cutoff", _eps(self.init_vecs.dtype) * 100)
        self.orig_matvec = self.matvec
        self.matvec = self._matvec

    def _matvec(self, v):
        try:
            return self.orig_matvec(v)
        except (TypeError, ValueError):
            try:
                # 尝试分别处理实部和虚部
                return self.orig_matvec(v.real) + 1j * self.orig_matvec(v.imag)
            except (TypeError, ValueError):
                # 如果仍然失败，转换为复数类型再处理
                return self.orig_matvec(v.astype(np.complex128))

    def run(self):
        self.tridiag()
        if np.iscomplexobj(self.q_mat):
            res = self.q_mat.real @ self.groundstt + 1j * (self.q_mat.imag @ self.groundstt)
        else:
            res = self.q_mat @ self.groundstt
        return self.groundeng, res

    def tridiag(self):
        """不支持 batch mode, multiple init vecs，并且针对基态做了收敛判断。"""
        # Define some constants
        dim = self.matrix_shape[-1]
        P_tol = self.tol
        num_iter = min(self.max_iter, dim)
        init_vecs = self.init_vecs.reshape(-1)
        matmul_closure = self.matvec
        cutoff = self.cutoff

        # Begin algorithm
        # Initial Q vector: q_0_vec
        q_0_vec = init_vecs / np.linalg.norm(init_vecs)

        # Initial alpha value: alpha_0
        r_vec = matmul_closure(q_0_vec)
        alpha_0 = np.real(np.sum(q_0_vec.conj() * r_vec))

        # Create storage for q_mat, alpha,and beta
        q_mat = np.zeros((num_iter, dim), dtype=np.asarray(r_vec).dtype)
        t_mat = np.zeros((num_iter, num_iter), dtype=np.float64)
        q_mat[0] = q_0_vec.reshape(-1)

        # Initial beta value: beta_0
        r_vec = r_vec - alpha_0 * q_0_vec
        beta_0 = np.linalg.norm(r_vec)

        # Copy over alpha_0 and beta_0 to t_mat
        t_mat[0, 0] = alpha_0
        if num_iter > 1:
            t_mat[0, 1] = beta_0
            t_mat[1, 0] = beta_0

        last_E_kr0 = np.zeros(1, dtype=np.float64)

        if abs(beta_0) < cutoff or num_iter == 1:
            k = 0
            E_kr = t_mat[:1, 0]
            v_kr = np.ones((1, 1), dtype=np.float64)
        else:
            # Compute the first new vector
            q_mat[1] = (r_vec / beta_0).reshape(-1)

            # Now we start the iteration
            for k in range(1, num_iter):
                # Get previous values
                q_prev_vec = q_mat[k - 1]
                q_curr_vec = q_mat[k]
                beta_prev = t_mat[k, k - 1]

                # Compute next alpha value
                r_vec = matmul_closure(q_curr_vec) - q_prev_vec * beta_prev
                alpha_curr = np.real(np.sum(q_curr_vec.conj() * r_vec))
                # Copy over to t_mat
                t_mat[k, k] = alpha_curr

                # 对于最低能量，我们可以提前停止
                E_kr, v_kr = self.tridiag2diag(t_mat[: k + 1, : k + 1])

                # Copy over alpha_curr, beta_curr to t_mat
                if (k + 1) < num_iter:
                    # Compute next residual value
                    r_vec = r_vec - alpha_curr * q_curr_vec
                    r_vec_norm = np.linalg.norm(r_vec)
                    r_vec = r_vec / r_vec_norm

                    # Run more reorthoganilzation if necessary
                    inner_products = np.abs(np.sum(q_mat[: k + 1].conj() * r_vec, axis=1))
                    could_reorthogonalize = True
                    for _ in range(10):
                        if not np.sum(inner_products > 1.0e-5):
                            could_reorthogonalize = True
                            break
                        correction = np.sum(r_vec * q_mat[: k + 1].conj())
                        correction = np.sum(q_mat[: k + 1] * correction, axis=0)
                        r_vec = r_vec - correction
                        r_vec_norm = np.linalg.norm(r_vec)
                        r_vec = r_vec / r_vec_norm
                        inner_products = np.abs(np.sum(q_mat[: k + 1].conj() * r_vec.reshape(1, -1), axis=1))

                    # Get next beta value
                    beta_curr = r_vec_norm
                    # Update t_mat with new beta value
                    t_mat[k, k + 1] = beta_curr
                    t_mat[k + 1, k] = beta_curr

                    # Update q_mat with new q value
                    q_mat[k + 1] = r_vec

                    if np.sum(abs(beta_curr) > 1e-6) == 0 or not could_reorthogonalize:
                        break

                    RitzRes = abs(v_kr[k, 0]) * t_mat[k, k + 1]
                    gap = E_kr[1] - E_kr[0]
                    gap = self.min_gap if gap < self.min_gap else gap
                    P_err = (RitzRes / gap) ** 2
                    Delta_E0 = last_E_kr0 - E_kr[0]
                    last_E_kr0 = E_kr[0]
                    if np.less(P_err, P_tol).all() and np.less(Delta_E0, self.E_tol).all():
                        break

        self.groundeng = E_kr[0]
        self.groundstt = v_kr[:, 0]

        # Now let's transpose q_mat, t_mat into the correct shape
        num_iter = k + 1

        self.q_mat = q_mat[:num_iter].T
        self.t_mat = t_mat[:num_iter, :num_iter]

    def tridiag2diag(self, t_mat):
        return np.linalg.eigh(t_mat)


class LanczosEvolveState(LanczosGeneral):
    def __init__(self, matvec, psi0, **kwargs):
        super().__init__(matvec, psi0, **kwargs)
        self.normalize = kwargs.get("normalize", None)
        self.cutoff = kwargs.get("cutoff", _eps(self.init_vecs.dtype) * 100)
        self.orig_matvec = self.matvec
        self.matvec = self._matvec

    def _matvec(self, v):
        try:
            return self.orig_matvec(v)
        except (TypeError, ValueError):
            try:
                # 尝试分别处理实部和虚部
                return self.orig_matvec(v.real) + 1j * self.orig_matvec(v.imag)
            except (TypeError, ValueError):
                # 如果仍然失败，转换为复数类型再处理
                return self.orig_matvec(v.astype(np.complex128))

    def run(self, delta):
        self.delta = delta

        self.tridiag()
        if self.t_mat.shape[0] == 1:
            exp_dH_v = self.exp_dh_e0[0] * self.init_vecs.reshape(-1)
        else:
            if np.iscomplexobj(self.q_mat) and not np.iscomplexobj(self.exp_dh_e0):
                exp_dH_v = self.q_mat.real @ self.exp_dh_e0 + 1j * (self.q_mat.imag @ self.exp_dh_e0)
            elif not np.iscomplexobj(self.q_mat) and np.iscomplexobj(self.exp_dh_e0):
                exp_dH_v = self.q_mat @ self.exp_dh_e0.real + 1j * (self.q_mat @ self.exp_dh_e0.imag)
            else:
                exp_dH_v = self.q_mat @ self.exp_dh_e0
            resnorm = np.linalg.norm(exp_dH_v)
            exp_dH_v = exp_dH_v / resnorm

        normalize = self.normalize
        if normalize is None:
            normalize = np.real(delta) == 0.0

        if normalize:
            return exp_dH_v
        beta = np.linalg.norm(self.init_vecs)
        return (beta * self.exp_dh_e0_norm) * exp_dH_v

    def tridiag(self):
        """不支持 batch mode, multiple init vecs，并且针对基态做了收敛判断。"""
        # Define some constants
        dim = self.matrix_shape[-1]
        P_tol = self.tol
        num_iter = min(self.max_iter, dim)
        init_vecs = self.init_vecs.reshape(-1)
        matmul_closure = self.matvec
        cutoff = self.cutoff

        # Begin algorithm
        # Initial Q vector: q_0_vec
        q_0_vec = init_vecs / np.linalg.norm(init_vecs)

        # Initial alpha value: alpha_0
        r_vec = matmul_closure(q_0_vec)
        alpha_0 = np.real(np.sum(q_0_vec.conj() * r_vec))

        # Create storage for q_mat, alpha,and beta
        q_mat = np.zeros((num_iter, dim), dtype=np.asarray(r_vec).dtype)
        t_mat = np.zeros((num_iter, num_iter), dtype=np.float64)
        q_mat[0] = q_0_vec

        # Initial beta value: beta_0
        r_vec = r_vec - alpha_0 * q_0_vec
        beta_0 = np.linalg.norm(r_vec)

        # Copy over alpha_0 and beta_0 to t_mat
        t_mat[0, 0] = alpha_0
        if num_iter > 1:
            t_mat[0, 1] = beta_0
            t_mat[1, 0] = beta_0

        if abs(beta_0) < cutoff or num_iter == 1:
            k = 0
            E = t_mat[:1, 0]
            exp_dh_e0 = np.exp(self.delta * E)
            exp_dh_e0_norm = np.abs(exp_dh_e0)
            exp_dh_e0 = exp_dh_e0 / exp_dh_e0_norm
        else:
            # Compute the first new vector
            q_mat[1] = r_vec / beta_0

            # Now we start the iteration
            for k in range(1, num_iter):
                # Get previous values
                q_prev_vec = q_mat[k - 1]
                q_curr_vec = q_mat[k]
                beta_prev = t_mat[k, k - 1]

                # Compute next alpha value
                r_vec = matmul_closure(q_curr_vec) - q_prev_vec * beta_prev
                alpha_curr = np.real(np.sum(q_curr_vec.conj() * r_vec))
                # Copy over to t_mat
                t_mat[k, k] = alpha_curr

                # 对于最低能量，我们可以提前停止
                E_kr, v_kr = self.tridiag2diag(t_mat[: k + 1, : k + 1])
                # todo 这里可以改成 inplace 的操作
                expE_v = np.exp(E_kr * self.delta) * v_kr[0, :]
                if np.iscomplexobj(expE_v):
                    exp_dh_e0 = v_kr @ expE_v.real + 1j * (v_kr @ expE_v.imag)
                else:
                    exp_dh_e0 = v_kr @ expE_v
                exp_dh_e0_norm = np.linalg.norm(exp_dh_e0)
                exp_dh_e0 = exp_dh_e0 / exp_dh_e0_norm

                # Copy over alpha_curr, beta_curr to t_mat
                if (k + 1) < num_iter:
                    # Compute next residual value
                    r_vec = r_vec - alpha_curr * q_curr_vec
                    r_vec_norm = np.linalg.norm(r_vec)
                    r_vec = r_vec / r_vec_norm

                    # Run more reorthoganilzation if necessary
                    inner_products = np.abs(np.sum(q_mat[: k + 1].conj() * r_vec, axis=1))
                    could_reorthogonalize = True
                    for _ in range(10):
                        if not np.sum(inner_products > 1.0e-5):
                            could_reorthogonalize = True
                            break
                        correction = np.sum(r_vec * q_mat[: k + 1].conj())
                        correction = np.sum(q_mat[: k + 1] * correction, axis=0)
                        r_vec = r_vec - correction
                        r_vec_norm = np.linalg.norm(r_vec)
                        r_vec = r_vec / r_vec_norm
                        inner_products = np.abs(np.sum(q_mat[: k + 1].conj() * r_vec.reshape(1, -1), axis=1))

                    # Get next beta value
                    beta_curr = r_vec_norm
                    # Update t_mat with new beta value
                    t_mat[k, k + 1] = beta_curr
                    t_mat[k + 1, k] = beta_curr

                    # Update q_mat with new q value
                    q_mat[k + 1] = r_vec

                    if np.sum(abs(beta_curr) > 1e-6) == 0 or not could_reorthogonalize:
                        break

                    if abs(exp_dh_e0[k]) < P_tol:
                        break

        self.exp_dh_e0_norm = exp_dh_e0_norm
        self.exp_dh_e0 = exp_dh_e0

        # Now let's transpose q_mat, t_mat into the correct shape
        num_iter = k + 1

        self.q_mat = q_mat[:num_iter].T
        self.t_mat = t_mat[:num_iter, :num_iter]

    def tridiag2diag(self, t_mat) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(t_mat)


def lanczos_ground_state(linear_oper: Callable[[np.ndarray], np.ndarray], v: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    通过 lanczos 计算最低能量。

    NumPy 版本只支持前向传播；``linear_oper`` 可以是矩阵，也可以是 matvec 函数。

    默认参数：
    ```
    params = {
    "max_iter": 5,
    "tol": 1e-10,
    "jitter_val": None,  # 加一些扰动，让程序更稳定
    "E_tol": np.inf,  # 是否优化最低能量
    "min_gap": 1.e-12,  # 最小能量差
    "cutoff": np.finfo(v.dtype).eps * 100,  # 如果新 Krylov 向量的范数太小，则中止的截止值
    }
    ```
    """
    v = np.asarray(v)
    params = {
        "max_iter": 5,
        "tol": 1e-10,
        "jitter_val": None,
        "E_tol": np.inf,
        "min_gap": 1.0e-12,
        "cutoff": _eps(v.dtype) * 100,
    }
    params.update(kwargs)
    return LanczosGroundState(linear_oper, v, **params).run()


def lanczos_evolve_state(linear_oper: Callable[[np.ndarray], np.ndarray], v: np.ndarray, delta, **kwargs) -> np.ndarray:
    """
    类版 Lanczos 时间演化，对应原版本中前一个 ``lanczos_evolve_state``。
    """
    v = np.asarray(v)
    params = {
        "max_iter": 20,
        "tol": 1e-10,
        "jitter_val": None,
        "normalize": None,
        "cutoff": _eps(v.dtype) * 100,
    }
    params.update(kwargs)
    return LanczosEvolveState(linear_oper, v, **params).run(delta)


########################################


def lanczos_ground_state2(matvec: Callable[[np.ndarray], np.ndarray], psi0: np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
    """
    从初始猜测 `|psi0>` 迭代地构建 Krylov 空间的正交基计算基态：

    `|psi0>`, `H|psi0>`, `H^2|psi0>`, ... `H^N |psi0>`

    这一组向量构成 Krylov 空间，将 `H` 投影到其中并求解，得到 "Ritz" 特征值/特征向量。最后，可以使用基将解转换回原始空间。

    一个重要策略是在若干步之后（隐式地）重新启动算法。这里**不**进行这种操作：当我们使用这些类时，通常有一个显式的外部循环，直到收敛，例如 DMRG 中的 "sweeps"。

    # todo，如果 psi0 是 list[ndarray] 的结构，如何实现？

    备注
    -----
    Ritz 残差 `RitzRes` 根据
    http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual 计算。
    给定间隙，Ritz 残差给出了波函数误差的界限，
    ``err < (RitzRes/gap)**2``。间隙是从完整的 Lanczos 谱估计的。
    """
    psi0 = np.asarray(psi0)
    paras = {
        "N_min": 2,
        "N_max": 20,
        "P_tol": 1.0e-14,
        "min_gap": 1.0e-12,
        "reortho": False,
        "cutoff": _eps(psi0.dtype) * 100,
        "E_tol": np.inf,
    }
    paras.update(kwargs)
    eng, vec, N = _lanczos_ground_state(matvec, psi0, **paras)
    # N 是迭代次数
    return eng, vec


def _lanczos_ground_state(matvec, psi0, N_min, N_max, P_tol, min_gap, reortho, cutoff, E_tol):
    bases = []  # 用 list 因为不确定会用几个本征态
    Es = np.zeros((N_max, N_max), dtype=np.float64)
    h = np.zeros((N_max + 1, N_max + 1), dtype=np.float64)

    # 构建 Krylov 空间
    beta = np.linalg.norm(psi0)

    if beta < cutoff:
        raise ValueError(f"Norm of self.psi0 too small: {beta}")

    # 因为要反复用，所以把这两个取出来，不用每次从 self 里面找
    w = psi0.copy()

    for k in range(N_max):
        # 计算矩阵元
        w = w / beta
        bases.append(w)
        w = matvec(w)
        alpha = np.real(w.conj() @ bases[-1])
        h[k, k] = alpha
        w = w - alpha * bases[-1]

        # 本征求解
        if k == 0:
            Es[0, 0] = h[0, 0]
            eigenvector = np.ones(1, dtype=np.float64)
        else:
            eng, vec = np.linalg.eigh(h[: k + 1, : k + 1])
            Es[k, : k + 1] = eng  # 保存本征值
            eigenvector = vec[:, 0]  # 保存最小值对应的本征向量

        # 构建下一个基矢和矩阵元
        if reortho:
            for b in bases[:-1]:
                w = w - (w.conj() @ b) * b
        elif k > 0:
            w = w - beta * bases[-2]
        beta = np.linalg.norm(w)
        h[k, k + 1] = h[k + 1, k] = beta

        # 判断是否停止
        if abs(beta) < cutoff:
            break

        if k + 1 < N_min:
            continue

        Es_k = Es[k, :]  # current energies
        RitzRes = abs(eigenvector[k]) * h[k, k + 1]
        gap = max(Es_k[1] - Es_k[0], min_gap)
        P_err = (RitzRes / gap) ** 2
        Delta_E0 = Es[k - 1, 0] - Es_k[0]

        if P_err < P_tol and Delta_E0 < E_tol:
            break

    E0 = Es[k, 0]

    if k == 0:
        return E0, psi0.copy() / np.linalg.norm(psi0), k + 1  # no better estimate available

    res = np.stack(bases).T @ eigenvector.astype(bases[0].dtype)
    resnorm = np.linalg.norm(res)
    if abs(1.0 - resnorm) > 1.0e-5:
        print("Krylov 正交性不能保证，尝试设置 reortho = True")

    return E0, res / resnorm, k + 1


def lanczos_evolve_state2(matvec: Callable[[np.ndarray], np.ndarray], psi0: np.ndarray, delta, **kwargs) -> np.ndarray:
    """
    从初始猜测 `|psi0>` 迭代地构建 Krylov 空间的正交基计算 `exp(delta H) |psi0>`：

    `|psi0>`, `H|psi0>`, `H^2|psi0>`, ... `H^N |psi0>`

    这一组向量构成 Krylov 空间，将 `H` 投影到其中得到矩阵三对角 `h`

    此时： `exp(delta h) e_0` 就对应 `exp(delta H) |psi0>`

    其中 `e_0 = (1, 0, 0, ...)`
    """
    psi0 = np.asarray(psi0)
    paras = {
        "N_min": 2,
        "N_max": 20,
        "P_tol": 1.0e-14,
        "reortho": False,
        "cutoff": _eps(psi0.dtype) * 100,
        "normalize": None,
    }
    paras.update(kwargs)
    vec, N = _lanczos_evolve_state(matvec, psi0, delta, **paras)
    # N 是迭代次数
    return vec


def _lanczos_evolve_state(matvec, psi0, delta, N_min, N_max, P_tol, reortho, cutoff, normalize):
    bases = []
    h = np.zeros([N_max + 1, N_max + 1], dtype=np.float64)

    # 构建 Krylov 空间
    beta = np.linalg.norm(psi0)

    if beta < cutoff:
        raise ValueError(f"Norm of self.psi0 too small: {beta}")

    w = psi0.copy()

    for k in range(N_max):
        # 计算矩阵元
        w = w / beta
        bases.append(w)
        w = matvec(w)

        alpha = np.real(w.conj() @ bases[-1])
        h[k, k] = alpha
        w = w - alpha * bases[-1]

        # 本征求解
        if k == 0:
            E = h[0, 0]
            exp_dE = np.exp(delta * E)
            exp_dh_e0_norm = np.abs(exp_dE)
            exp_dh_e0 = np.array([exp_dE / exp_dh_e0_norm])
        else:
            eng, vec = np.linalg.eigh(h[: k + 1, : k + 1])
            tmp = np.exp(eng * delta) * np.conj(vec[0, :])
            exp_dh_e0 = vec.astype(tmp.dtype) @ tmp

            exp_dh_e0_norm = np.linalg.norm(exp_dh_e0)
            exp_dh_e0 = exp_dh_e0 / exp_dh_e0_norm

        # 构建下一个基矢和矩阵元
        if reortho:
            for b in bases[:-1]:
                w = w - (w.conj() @ b) * b
        elif k > 0:
            w = w - beta * bases[-2]
        beta = np.linalg.norm(w)
        h[k, k + 1] = h[k + 1, k] = beta

        # 判断是否停止
        if abs(beta) < cutoff:
            break

        if k + 1 < N_min:
            continue

        if abs(exp_dh_e0[k]) < P_tol:
            break

    if k == 0:
        exp_dH_v = exp_dh_e0[0] * psi0
    else:
        exp_dH_v = np.stack(bases).T @ exp_dh_e0
        resnorm = np.linalg.norm(exp_dH_v)
        if abs(1.0 - resnorm) > 1.0e-5:
            print("Krylov 正交性不能保证，尝试设置 reortho = True")
        exp_dH_v = exp_dH_v / resnorm

    if normalize is None:
        normalize = np.real(delta) == 0.0

    if normalize:
        return exp_dH_v, k + 1
    beta = np.linalg.norm(psi0)
    return (beta * exp_dh_e0_norm) * exp_dH_v, k + 1


def lanczos_arpack(matvec:Callable[[np.ndarray], np.ndarray], psi0:np.ndarray, **kwargs) -> tuple[float, np.ndarray]:
    """使用 `scipy.sparse.linalg.eigsh` 计算基态
    """
    tol = kwargs.get("P_tol", 1e-14)
    ncv = kwargs.get("N_min", None)
    which = kwargs.get("which", 'SA')
    dim = psi0.shape[0]
    lo = spalg.LinearOperator(shape=(dim,dim), matvec=matvec, dtype=psi0.dtype) # type: ignore
    Es, Vs = spalg.eigsh(lo, k=1, v0=psi0, which=which, tol=tol, ncv=ncv)
    # k = 1 if dim < 5 else 3
    # Es, Vs = spalg.eigs(lo, k=k, v0=psi0, which='LM', tol=tol, ncv=ncv)
    # show(Es)
    return Es[0], Vs[:, 0] #+ 1e-6*np.random.randn(dim)

