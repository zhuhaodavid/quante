# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-09 18:07:00
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 19:29:53

import torch as tc


from typing import Callable
from torch.autograd import Function

class Lanczos(Function):
    """
    自动微分的 Lanczos 对角化算法。
    
    这部分代码来自：
    
    https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/functions/_diagonalization.py
    
    #TODO: 学习 https://arxiv.org/pdf/1509.07838.pdf 推导 dmrg 中乘法的反向传播公式
    
    #TODO: 如何实现 representation_tree？
    """
    @staticmethod
    def forward(ctx, lo, tsr, memory_efficient):
        """
        正常传播
       
        lo 需要具有的特征包括：
        
        - matrix_shape: (dim, dim)
        - dtype: 数据类型
        - device: 设备
        - max_iter: 最大迭代次数
        - batch_shape: 批次形状
        - jitter_val: 随机值
        
        需要包括的方法为：
        
        - tridiag: 计算三对角矩阵
        - tridiag2diag: 从三对角矩阵计算特征值和特征向量 
        
        """
        ctx.matrix_shape = lo.matrix_shape
        ctx.dtype = lo.dtype
        ctx.device = lo.device
        ctx.max_iter = lo.max_iter
        ctx.batch_shape = lo.batch_shape
        
        lo.tridiag()
        q_mat, t_mat = lo.q_mat.contiguous(), lo.t_mat.contiguous()
        # q_mat: num_init_vecs x batch_shape x matrix_shape[-1] x num_iter
        # t_mat: num_init_vecs x batch_shape x num_iter x num_iter
        
        jitter_val = lo.jitter_val

        if lo.batch_shape is None:
            q_mat = q_mat.unsqueeze(-3)
            t_mat = t_mat.unsqueeze(-3)
        if t_mat.ndimension() == 3:  # If we only used one probe vector
            q_mat = q_mat.unsqueeze(0)
            t_mat = t_mat.unsqueeze(0)

        if jitter_val is not None:
            mins = tc.diagonal(t_mat, dim1=-1, dim2=-2).min(dim=-1, keepdim=True)[0]
            jitter_mat = tc.diag_embed(jitter_val * mins).expand_as(t_mat)
            t_mat_with_jitter = t_mat + jitter_mat
        else:
            t_mat_with_jitter = t_mat
        eigenvalues, eigenvectors = lo.tridiag2diag(t_mat_with_jitter)
        
        eigenvalues = eigenvalues.to(q_mat.dtype)
        eigenvectors = eigenvectors.to(q_mat.dtype)
        
        # Get orthogonal matrix and eigenvalues
        q_mat = q_mat.matmul(eigenvectors)
        
        if ctx.batch_shape is None:
            q_mat = q_mat.squeeze(1)
            eigenvalues = eigenvalues.squeeze(1)
        
        q_mat = q_mat.squeeze(0)
        eigenvalues = eigenvalues.squeeze(0)
        
        if memory_efficient:
            ctx._linear_op = tsr.matmul
        to_save = [tsr] + [q_mat, eigenvalues]
        ctx.save_for_backward(*to_save)
        
        return eigenvalues, q_mat

    @staticmethod
    def backward(ctx, evals_grad_output, evecs_grad_output):
        """
        反向传播，目前只支持矩阵，反向传播得到一个完整的矩阵
        """

        q_mat = ctx.saved_tensors[-2]
        eigenvalues = ctx.saved_tensors[-1]

        # (\tilde K)_{ij} = 1_{i\neq j} (\sigma_i - \sigma_j)^{-1}
        # add a small amount of jitter to ensure that no zeros are produced
        kmat = (eigenvalues.unsqueeze(-1) - eigenvalues.unsqueeze(-2) + 1e-10).reciprocal()
        tc.diagonal(kmat, dim1=-1, dim2=-2).zero_()

        # dU = U(\tilde K^T \hadamard (U^T dL/dU)U^T
        inner_term = kmat.mT * q_mat.mT.matmul(evecs_grad_output)
        term1 = q_mat.matmul(inner_term).matmul(q_mat.mT)

        # d\Sigma = U dL/d\Sigma U^T
        term2 = q_mat.matmul(tc.diag_embed(evals_grad_output)).matmul(q_mat.mT)

        # finally sum the two
        dL_dM = term1 + term2

        return None, dL_dM, None, None

class LanczosGeneral:
    def __init__(self, matvec, psi0, **kwargs):
        """
        来自：
        https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/lanczos.py
        中对 lanczos 的实现
        
        只能处理实对称矩阵
        """
        if isinstance(matvec, tc.Tensor):
            self.tsr = matvec
            self.matvec = matvec.matmul
        else:
            self.tsr = None
            self.matvec = matvec
        
        self.init_vecs = psi0
        
        self.dim = psi0.shape[0]
        self.matrix_shape = (self.dim, self.dim)
        self.device = psi0.device
        
        self.max_iter = kwargs.get('max_iter', 20)
        self.tol = kwargs.get('tol', 1.e-14)
        self.batch_shape = kwargs.get('batch_shape', None)
        self.jitter_val = kwargs.get('jitter_val', None)
        
    def run(self):
        if self.tsr is not None:
            if tc.is_complex(self.tsr):
                raise ValueError("这个代码中对复矩阵的处理是错误的")
            engs, vecs = Lanczos.apply(self, self.tsr, False)
            return engs, vecs
        else:
            self.tridiag()
            return self.eigenvalues, self.eigenvectors
    
    def tridiag(self):
        matmul_closure = self.matvec
        matrix_shape = self.matrix_shape
        batch_shape = tc.Size([]) if self.batch_shape is None else self.batch_shape
        device = self.device
        max_iter = self.max_iter
        init_vecs = self.init_vecs
        tol = self.tol
        
        # Determine batch mode
        multiple_init_vecs = False

        # Get initial probe ectors - and define if not available
        if init_vecs is None:
            init_vecs = tc.randn(matrix_shape[-1], num_init_vecs, dtype=tc.float64, device=device)
            init_vecs = init_vecs.expand(*batch_shape, matrix_shape[-1], num_init_vecs)

        else:
            
            num_init_vecs = init_vecs.size(-1)

        # Define some constants
        num_iter = min(max_iter, matrix_shape[-1])
        dim_dimension = -2

        # Create storage for q_mat, alpha,and beta
        # q_mat - batch version of Q - orthogonal matrix of decomp
        # alpha - batch version main diagonal of T
        # beta - batch version of off diagonal of T
        q_mat = tc.zeros(
            num_iter,
            *batch_shape,
            matrix_shape[-1],
            num_init_vecs,
            dtype=tc.float64,
            device=device,
        )
        t_mat = tc.zeros(num_iter, num_iter, *batch_shape, num_init_vecs, dtype=tc.float64, device=device)

        # Begin algorithm
        # Initial Q vector: q_0_vec
        q_0_vec = init_vecs / tc.norm(init_vecs, 2, dim=dim_dimension).unsqueeze(dim_dimension)
        q_mat[0].copy_(q_0_vec)

        # Initial alpha value: alpha_0
        r_vec = matmul_closure(q_0_vec)
        alpha_0 = q_0_vec.mul(r_vec).sum(dim_dimension)

        # Initial beta value: beta_0
        r_vec.sub_(alpha_0.unsqueeze(dim_dimension).mul(q_0_vec))
        beta_0 = tc.norm(r_vec, 2, dim=dim_dimension)

        # Copy over alpha_0 and beta_0 to t_mat
        t_mat[0, 0].copy_(alpha_0)
        t_mat[0, 1].copy_(beta_0)
        t_mat[1, 0].copy_(beta_0)

        # Compute the first new vector
        q_mat[1].copy_(r_vec.div_(beta_0.unsqueeze(dim_dimension)))

        # Now we start the iteration
        for k in range(1, num_iter):
            # Get previous values
            q_prev_vec = q_mat[k - 1]
            q_curr_vec = q_mat[k]
            beta_prev = t_mat[k, k - 1].unsqueeze(dim_dimension)

            # Compute next alpha value
            r_vec = matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)
            alpha_curr = q_curr_vec.mul(r_vec).sum(dim_dimension, keepdim=True)
            # Copy over to t_mat
            t_mat[k, k].copy_(alpha_curr.squeeze(dim_dimension))

            # Copy over alpha_curr, beta_curr to t_mat
            if (k + 1) < num_iter:
                # Compute next residual value
                r_vec.sub_(alpha_curr.mul(q_curr_vec))
                # Full reorthogonalization: r <- r - Q (Q^T r)
                correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
                correction = q_mat[: k + 1].mul(correction).sum(0)
                r_vec.sub_(correction)
                r_vec_norm = tc.norm(r_vec, 2, dim=dim_dimension, keepdim=True)
                r_vec.div_(r_vec_norm)

                # Get next beta value
                beta_curr = r_vec_norm.squeeze_(dim_dimension)
                # Update t_mat with new beta value
                t_mat[k, k + 1].copy_(beta_curr)
                t_mat[k + 1, k].copy_(beta_curr)

                # Run more reorthoganilzation if necessary
                inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)
                could_reorthogonalize = False
                for _ in range(10):
                    if not tc.sum(inner_products > tol):
                        could_reorthogonalize = True
                        break
                    correction = r_vec.unsqueeze(0).mul(q_mat[: k + 1]).sum(dim_dimension, keepdim=True)
                    correction = q_mat[: k + 1].mul(correction).sum(0)
                    r_vec.sub_(correction)
                    r_vec_norm = tc.norm(r_vec, 2, dim=dim_dimension, keepdim=True)
                    r_vec.div_(r_vec_norm)
                    inner_products = q_mat[: k + 1].mul(r_vec.unsqueeze(0)).sum(dim_dimension)

                # Update q_mat with new q value
                q_mat[k + 1].copy_(r_vec)

                if tc.sum(beta_curr.abs() > 1e-6) == 0 or not could_reorthogonalize:
                    break

        # Now let's transpose q_mat, t_mat intot the correct shape
        num_iter = k + 1

        # num_init_vecs x batch_shape x matrix_shape[-1] x num_iter
        q_mat = q_mat[:num_iter].permute(-1, *range(1, 1 + len(batch_shape)), -2, 0).contiguous()
        # num_init_vecs x batch_shape x num_iter x num_iter
        t_mat = t_mat[:num_iter, :num_iter].permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous()

        # If we weren't in batch mode, remove batch dimension
        if not multiple_init_vecs:
            q_mat.squeeze_(0)
            t_mat.squeeze_(0)

        # We're done!
        return q_mat, t_mat

    def tridiag2diag(self, t_mat):
        orig_device = t_mat.device

        if t_mat.size(-1) < 32:
            retr = tc.linalg.eigh(t_mat.cpu())
        else:
            retr = tc.linalg.eigh(t_mat)

        evals, evecs = retr
        # todo: 只取正数部分是不是计算梯度的要求？
        mask = evals.ge(0)
        evecs = evecs * mask.type_as(evecs).unsqueeze(-2)
        evals = evals.masked_fill_(~mask, 1)

        return evals.to(orig_device), evecs.to(orig_device)


class LanczosGroundState(LanczosGeneral):
    def __init__(self, matvec, psi0, **kwargs):
        """ 在 LanczosGeneral 的基础上，针对基态做了改动 """
        super().__init__(matvec, psi0, **kwargs)
        self.E_tol = kwargs.get("E_tol", tc.inf)
        self.min_gap = kwargs.get("min_gap", 1.e-12)
        self.cutoff = kwargs.get("cutoff", tc.finfo(psi0.dtype).eps * 100)
        self.orig_matvec = matvec
        self.matvec = self._matvec
        
    def _matvec(self, v):
        try:
            return self.orig_matvec(v)
        except RuntimeError:
            try:
                # 尝试分别处理实部和虚部
                return self.orig_matvec(v.real) + 1j * self.orig_matvec(v.imag)
            except RuntimeError:
                # 如果仍然失败，转换为复数类型再处理
                return self.orig_matvec(v.to(dtype=tc.complex128))

    def run(self):
        if self.tsr is not None:
            engs, vecs = Lanczos.apply(self, self.tsr, False)
            print("警告：正在使用带梯度的方法计算，梯度并不可靠（todo）")
            return engs[0].real, vecs[:,0]
        else:
            self.tridiag()
            if tc.is_complex(self.q_mat):
                res = self.q_mat.real @ self.groundstt + 1j * (self.q_mat.imag @ self.groundstt)
            else:
                res = self.q_mat @ self.groundstt
            return self.groundeng, res
    
    def tridiag(self):
        """不支持 batch mode, multiple init vecs，并且针对基态做了收敛判断"""
        # Define some constants
        dim = self.matrix_shape[-1]
        P_tol = self.tol
        num_iter = min(self.max_iter, dim)
        init_vecs = self.init_vecs
        matmul_closure = self.matvec
        cutoff = self.cutoff
        
        # Begin algorithm
        # Initial Q vector: q_0_vec
        q_0_vec = init_vecs / tc.norm(init_vecs)

        # Initial alpha value: alpha_0
        r_vec = matmul_closure(q_0_vec)
        alpha_0 = q_0_vec.conj().mul(r_vec).sum().real
        
        # Create storage for q_mat, alpha,and beta
        q_mat = tc.zeros(
            num_iter,
            dim,
            dtype=r_vec.dtype,
            device=self.device,
        )
        t_mat = tc.zeros(num_iter, num_iter, dtype=r_vec.dtype.to_real(), device=self.device)
        q_mat[0].copy_(q_0_vec.reshape(-1))
        
        # Initial beta value: beta_0
        r_vec.sub_(alpha_0.mul(q_0_vec))
        beta_0 = tc.norm(r_vec)

        # Copy over alpha_0 and beta_0 to t_mat
        t_mat[0, 0].copy_(alpha_0)
        t_mat[0, 1].copy_(beta_0)
        t_mat[1, 0].copy_(beta_0)
        
        last_E_kr0 = tc.zeros(1, dtype=tc.float64, device=self.device)
        
        if beta_0.abs() < cutoff:
            k = 0
            E_kr = t_mat[:1,0]
            v_kr = tc.ones(1, 1, dtype=tc.float64, device=self.device)
        else:
            # Compute the first new vector
            q_mat[1].copy_(r_vec.div_(beta_0).reshape(-1))

            # Now we start the iteration
            for k in range(1, num_iter):
                # Get previous values
                q_prev_vec = q_mat[k - 1]
                q_curr_vec = q_mat[k]
                beta_prev = t_mat[k, k - 1]

                # Compute next alpha value
                r_vec = matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)  #!!! 主要的时间消耗都在这里
                alpha_curr = q_curr_vec.conj().mul(r_vec).sum().real
                # Copy over to t_mat
                t_mat[k, k].copy_(alpha_curr)
    
                # 对于最低能量，我们可以提前停止
                E_kr, v_kr = self.tridiag2diag(t_mat[: k + 1, : k + 1])
                
                # Copy over alpha_curr, beta_curr to t_mat
                if (k + 1) < num_iter:
                    # Compute next residual value
                    r_vec.sub_(alpha_curr.mul(q_curr_vec))
                    r_vec_norm = tc.norm(r_vec)
                    r_vec.div_(r_vec_norm)

                    # Run more reorthoganilzation if necessary
                    inner_products = q_mat[: k + 1].conj().mul(r_vec).sum().abs()
                    could_reorthogonalize = True
                    for _ in range(10):
                        if not tc.sum(inner_products > 1.e-5):
                            could_reorthogonalize = True
                            break
                        correction = r_vec.mul(q_mat[: k + 1].conj()).sum()
                        correction = q_mat[: k + 1].mul(correction).sum(0)
                        r_vec.sub_(correction)
                        r_vec_norm = tc.norm(r_vec)
                        r_vec.div_(r_vec_norm)
                        inner_products = q_mat[: k + 1].conj().mul(r_vec.unsqueeze(0)).sum().abs()

                    # Get next beta value
                    beta_curr = r_vec_norm
                    # Update t_mat with new beta value
                    t_mat[k, k + 1].copy_(beta_curr)
                    t_mat[k + 1, k].copy_(beta_curr)
                    
                    # Update q_mat with new q value
                    q_mat[k + 1].copy_(r_vec)

                    if tc.sum(beta_curr.abs() > 1e-6) == 0 or not could_reorthogonalize:
                        break
              
                    RitzRes = abs(v_kr[k, 0]) * t_mat[k, k + 1]
                    gap = E_kr[1] - E_kr[0]
                    gap = tc.where(gap < self.min_gap, tc.tensor(self.min_gap, dtype=gap.dtype, device=gap.device), gap)
                    P_err = (RitzRes / gap)**2
                    Delta_E0 = last_E_kr0 - E_kr[0]
                    last_E_kr0 = E_kr[0]
                    if tc.less(P_err, P_tol).all() and tc.less(Delta_E0, self.E_tol).all():
                        break
        
        self.groundeng = E_kr[0]
        self.groundstt = v_kr[:, 0]
        
        # Now let's transpose q_mat, t_mat intot the correct shape
        num_iter = k + 1
        
        # num_init_vecs x batch_shape x matrix_shape[-1] x num_iter
        self.q_mat = q_mat[:num_iter].T
        # num_init_vecs x batch_shape x num_iter x num_iter
        self.t_mat = t_mat[:num_iter, :num_iter]

        # We're done!

    def tridiag2diag(self, t_mat):
        orig_device = t_mat.device

        if t_mat.size(-1) < 32:
            retr = tc.linalg.eigh(t_mat.cpu())
        else:
            retr = tc.linalg.eigh(t_mat)

        evals, evecs = retr
        return evals.to(orig_device), evecs.to(orig_device)


class LanczosEvolveState(LanczosGeneral):
    def __init__(self, matvec, psi0, **kwargs):
        super().__init__(matvec, psi0, **kwargs)
        self.normalize = kwargs.get("normalize", None)
        self.cutoff = tc.finfo(psi0.dtype).eps * 100
        self.orig_matvec = matvec
        self.matvec = self._matvec
        
    def _matvec(self, v):
        try:
            return self.orig_matvec(v)
        except RuntimeError:
            try:
                # 尝试分别处理实部和虚部
                return self.orig_matvec(v.real) + 1j * self.orig_matvec(v.imag)
            except RuntimeError:
                # 如果仍然失败，转换为复数类型再处理
                return self.orig_matvec(v.to(dtype=tc.complex128))
            
    
    def run(self, delta):
        self.delta = delta
        
        if self.tsr is not None:
            raise NotImplementedError
            engs, vecs = Lanczos.apply(self, self.tsr, False)
            # todo: 通过 engs, vecs 能否计算 exp(-iHt) * psi0 ?
            # q_mat 是投影，也就是说要计算：
            # q_mat @ V @ exp(-iEt) @ V†[:,0]
            # 其中 q_mat @ V = vecs，那 V†[:,0] 要怎么得到？
        else:
            self.tridiag()
            if self.t_mat.shape[0] == 1:
                exp_dH_v = self.exp_dh_e0[0] * self.init_vecs.reshape(-1)
            else:
                if tc.is_complex(self.q_mat) and not tc.is_complex(self.exp_dh_e0):
                    exp_dH_v = self.q_mat.real @ self.exp_dh_e0 + 1j * (self.q_mat.imag @ self.exp_dh_e0)
                elif not tc.is_complex(self.q_mat) and tc.is_complex(self.exp_dh_e0):
                    exp_dH_v = self.q_mat @ self.exp_dh_e0.real + 1j * (self.q_mat @ self.exp_dh_e0.imag)
                else:
                    exp_dH_v = self.q_mat @ self.exp_dh_e0
                resnorm = tc.norm(exp_dH_v)
                exp_dH_v.div_(resnorm)
               
            normalize = self.normalize 
            if normalize is None:
                normalize = delta.real == 0.
            
            if normalize:
                return exp_dH_v
            else:
                beta = tc.norm(self.init_vecs)
                return (beta * self.exp_dh_e0_norm) * exp_dH_v
    
    def tridiag(self):
        """不支持 batch mode, multiple init vecs，并且针对基态做了收敛判断"""
        # Define some constants
        dim = self.matrix_shape[-1]
        P_tol = self.tol
        num_iter = min(self.max_iter, dim)
        init_vecs = self.init_vecs.reshape(-1)
        matmul_closure = self.matvec
        cutoff = self.cutoff

        # Begin algorithm
        # Initial Q vector: q_0_vec
        q_0_vec = init_vecs / tc.norm(init_vecs)

        # Initial alpha value: alpha_0
        r_vec = matmul_closure(q_0_vec)
        alpha_0 = q_0_vec.conj().mul(r_vec).sum().real
        
        # Create storage for q_mat, alpha,and beta
        q_mat = tc.zeros(
            num_iter,
            dim,
            dtype=r_vec.dtype,
            device=self.device,
        )
        t_mat = tc.zeros(num_iter, num_iter, dtype=r_vec.dtype.to_real(), device=self.device)
        q_mat[0].copy_(q_0_vec)
        
        # Initial beta value: beta_0
        r_vec.sub_(alpha_0.mul(q_0_vec))
        beta_0 = tc.norm(r_vec)

        # Copy over alpha_0 and beta_0 to t_mat
        t_mat[0, 0].copy_(alpha_0)
        t_mat[0, 1].copy_(beta_0)
        t_mat[1, 0].copy_(beta_0)
        
        if beta_0.abs() < cutoff:
            k = 0
            E = t_mat[:1,0]
            exp_dh_e0 = tc.exp(self.delta * E)
            exp_dh_e0_norm = tc.abs(exp_dh_e0)
            exp_dh_e0.div_(exp_dh_e0_norm)
        else:
            # Compute the first new vector
            q_mat[1].copy_(r_vec.div_(beta_0))

            # Now we start the iteration
            for k in range(1, num_iter):
                # Get previous values
                q_prev_vec = q_mat[k - 1]
                q_curr_vec = q_mat[k]
                beta_prev = t_mat[k, k - 1]

                # Compute next alpha value
                r_vec = matmul_closure(q_curr_vec) - q_prev_vec.mul(beta_prev)  #!!! 主要的时间消耗都在这里
                alpha_curr = q_curr_vec.conj().mul(r_vec).sum().real
                # Copy over to t_mat
                t_mat[k, k].copy_(alpha_curr)
    
                # 对于最低能量，我们可以提前停止
                E_kr, v_kr = self.tridiag2diag(t_mat[: k + 1, : k + 1])
                # todo 这里可以改成 inplace 的操作
                # tmp.mul_(tc.exp(E_kr * self.delta))  # 错误，需要clone
                expE_v = tc.exp(E_kr * self.delta) * v_kr[0, :]
                if tc.is_complex(expE_v):
                    exp_dh_e0 = v_kr @ expE_v.real + 1j * (v_kr @ expE_v.imag)
                else:
                    exp_dh_e0 = v_kr @ expE_v
                exp_dh_e0_norm = tc.norm(exp_dh_e0)
                exp_dh_e0.div_(exp_dh_e0_norm)
                
                # Copy over alpha_curr, beta_curr to t_mat
                if (k + 1) < num_iter:
                    # Compute next residual value
                    r_vec.sub_(alpha_curr.mul(q_curr_vec))
                    r_vec_norm = tc.norm(r_vec)
                    r_vec.div_(r_vec_norm)

                    # Run more reorthoganilzation if necessary
                    inner_products = q_mat[: k + 1].conj().mul(r_vec).sum().abs()
                    could_reorthogonalize = True
                    for _ in range(10):
                        if not tc.sum(inner_products > 1.e-5):
                            could_reorthogonalize = True
                            break
                        correction = r_vec.mul(q_mat[: k + 1].conj()).sum()
                        correction = q_mat[: k + 1].mul(correction).sum(0)
                        r_vec.sub_(correction)
                        r_vec_norm = tc.norm(r_vec)
                        r_vec.div_(r_vec_norm)
                        inner_products = q_mat[: k + 1].conj().mul(r_vec.unsqueeze(0)).sum().abs()

                    # Get next beta value
                    beta_curr = r_vec_norm
                    # Update t_mat with new beta value
                    t_mat[k, k + 1].copy_(beta_curr)
                    t_mat[k + 1, k].copy_(beta_curr)
                    
                    # Update q_mat with new q value
                    q_mat[k + 1].copy_(r_vec)

                    if tc.sum(beta_curr.abs() > 1e-6) == 0 or not could_reorthogonalize:
                        break
                    
                    if tc.abs(exp_dh_e0[k]) < P_tol:
                        break
        
        self.exp_dh_e0_norm = exp_dh_e0_norm
        self.exp_dh_e0 = exp_dh_e0
        
        # Now let's transpose q_mat, t_mat intot the correct shape
        num_iter = k + 1
   
        # num_init_vecs x batch_shape x matrix_shape[-1] x num_iter
        self.q_mat = q_mat[:num_iter].T
        # num_init_vecs x batch_shape x num_iter x num_iter
        self.t_mat = t_mat[:num_iter, :num_iter]

        # We're done!

    def tridiag2diag(self, t_mat) -> tuple[tc.Tensor, tc.Tensor]:
        orig_device = t_mat.device

        if t_mat.size(-1) < 32:
            retr = tc.linalg.eigh(t_mat.cpu())
        else:
            retr = tc.linalg.eigh(t_mat)

        evals, evecs = retr
        return evals.to(orig_device), evecs.to(orig_device)


def lanczos_ground_state(linear_oper:Callable[[tc.Tensor],tc.Tensor], v:tc.Tensor, **kwargs) -> tuple[tc.Tensor, tc.Tensor]:
    """
    通过 lanczos 计算最低能量
    
    如果 linear_oper 是张量，那么运行自动微分（待测试）
    
    默认参数：
    ```
    params = {
    "max_iter": 20,
    "tol": 1e-10,
    "jitter_val": None,  # 加一些扰动，让程序更稳定
    "E_tol": tc.inf,  # 是否优化最低能量
    "min_gap": 1.e-12,  # 最小能量差
    "cutoff": tc.finfo(v.dtype).eps * 100,  # 如果新 Krylov 向量的范数太小，则中止的截止值
    }
    ```
    
    否则只能前向传播
    
    Examples
    --------
    >>> import torch
    >>> import quante.torch_utils.tclinalg as qtc
    >>> mat = torch.randn(100, 100, dtype=torch.float64)
    >>> mat = mat + mat.H
    >>> v = torch.randn(100, dtype=torch.float64)
    >>> eng, vec = qtc.lanczos_ground_state(mat.matmul, v.reshape(-1,1))
    >>> import quante as qt
    >>> eng, vec2 = qt.linalg.lanczos_ground_state(mat.numpy().dot, v.numpy())
    >>> print(vec.numpy() - vec2)
    """
    # 参数设置：
    params = {
        "max_iter": 20,
        "tol": 1e-10,
        "jitter_val": None,  # 加一些扰动，让程序更稳定
        "E_tol": tc.inf,  # 是否优化最低能量
        "min_gap": 1.e-12,  # 最小能量差
        "cutoff": tc.finfo(v.dtype).eps * 100,  # 如果新 Krylov 向量的范数太小，则中止的截止值
    }
    params.update(kwargs)
    return LanczosGroundState(linear_oper, v, **params).run()


def lanczos_evolve_state(linear_oper:Callable[[tc.Tensor],tc.Tensor], v:tc.Tensor, delta, **kwargs) -> tuple[tc.Tensor, tc.Tensor]:
    """
    通过 lanczos 计算最低能量
    
    如果 linear_oper 是张量，那么运行自动微分（待测试）
    
    否则只能前向传播
    
    Examples
    --------
    >>> import torch
    >>> import quante.torch_utils.tclinalg as qtc
    >>> mat = torch.randn(100, 100, dtype=torch.float64)
    >>> mat = mat + mat.H
    >>> v = torch.randn(100, dtype=torch.float64)
    >>> vec = qtc.lanczos_evolve_state(mat.matmul, v.reshape(-1,1), 0.1)
    >>> import quante as qt
    >>> vec2 = qt.linalg.lanczos_evolve_state(mat.numpy().dot, v.numpy(), 0.1)
    >>> print(vec.numpy() - vec2)
    """
    # 参数设置：
    params = {
        "max_iter": 20,
        "tol": 1e-10,
        "jitter_val": None,  # 加一些扰动，让程序更稳定
        "normalize": None,
        "cutoff": tc.finfo(v.dtype).eps * 100,  # 如果新 Krylov 向量的范数太小，则中止的截止值
    }
    params.update(kwargs)
    return LanczosEvolveState(linear_oper, v, **params).run(delta)
