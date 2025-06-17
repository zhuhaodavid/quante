# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-01-18 15:44:48
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 11:33:41

# 定义了 ProjOper, ProjMPO, ProjMPS, ProjSumMPO, ProjMPOMPS 等类
# 它们关系为：

# ProjOper
#    |
#    |--- ProjMPO
#    |
#    |--- ProjMPS
# 
# ProjSumMPO
# 
# ProjMPOMPS


import torch as tc
from typing import TypeVar
from functools import reduce
T = TypeVar('T')

from . import MPS, MPO
from . import tensor_operations as tf

class ProjOper:
    """具体的实例需要包括：
    - 属性：ndim, lognm, shape
    - 方法：copy, contract_left_env, contract_right_env, (dmrg)get_matmul_func, (dmrg)to_matrix

    
    ProjOper `PH` 表示的网络图示（`PH.set_position_(psi, 3)`）：

    .. code-block:: text
    
        o--o--o-      -o--o--o--o--o--o <psi|
        |  |  |  |  |  |  |  |  |  |  |
        o--o--o--o--o--o--o--o--o--o--o H
        |  |  |  |  |  |  |  |  |  |  |
        o--o--o-      -o--o--o--o--o--o |psi>
                ↑        ↑
            lpos=2    rpos=5
    """
    def __init__(self, L, dtype, device, nsite, ifnorm=False) -> None:
        self.lpos = -1
        self.rpos = self.L = L
        self.dtype = dtype
        self.device = device
        self.LR = [None] * L
        self.LRlognm = [tc.tensor(0., dtype=tc.float64, device=device) for _ in range(L)]
        self.nsite = nsite
        self.ifnorm = ifnorm
    
    @property
    def lrlognm(self):
        res = tc.tensor(0., dtype=tc.float64, device=self.device)
        if self.lpos >= 0:
            res += self.LRlognm[self.lpos]
        elif self.rpos < self.L:
            res += self.LRlognm[self.rpos]
        return res

    @property
    def site_range(self):
        return range(self.lpos + 1, self.rpos)
    
    def set_nsite(self, nsite):
        self.nsite = nsite
    
    def __len__(self):
        return self.L

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
                Lenv = self.contract_left_env(Lenv, ll, psi)
                if self.ifnorm:
                    nm = tc.linalg.norm(Lenv)
                    Lenv /= nm
                    self.LR[ll + 1] = Lenv
                    self.LRlognm[ll + 1] = (tc.log(nm) if ll == -1 else tc.log(nm) + self.LRlognm[ll])
                else:
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
            rl = min(self.rpos, self.L)
            Renv = self.rproj()
            while rl > k:
                Renv = self.contract_right_env(Renv, rl, psi)
                if self.ifnorm:
                    nm = tc.linalg.norm(Renv)
                    Renv /= nm
                    self.LR[rl - 1] = Renv
                    self.LRlognm[rl - 1] = (tc.log(nm) if rl == self.L else tc.log(nm) + self.LRlognm[rl])
                else:
                    # 检查 Lenv 中是否有 inf, nan:
                    if not tc.isfinite(Renv).all():
                        raise ValueError(f"inf or nan in Lenv at {rl}")
                    self.LR[rl - 1] = Renv
                rl -= 1
            self.rpos = k

    def lproj(self):
        if self.lpos <= -1:
            return tc.tensor([1.], dtype=self.dtype, device=self.device).reshape(*[1]*self.ndim)
        return self.LR[self.lpos]
    
    def rproj(self):
        if self.rpos >= self.L:
            return tc.tensor([1.], dtype=self.dtype, device=self.device).reshape(*[1]*self.ndim)
        return self.LR[self.rpos]
     
    def set_position_(self, psi:MPS, pos:int):
        self.dtype = psi.data[0].dtype
        self.makeL_(psi, pos - 1)
        self.makeR_(psi, pos + self.nsite)
    

class ProjMPO(ProjOper):
    def __init__(self, H, nsite=2, ifnorm=False) -> None:
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
        self.mid = H
        self.ndim = 3
        super().__init__(L=len(H), dtype=H.dtype, device=H.device, nsite=nsite, ifnorm=ifnorm)
    
    @property
    def phys_dim(self):
        return self.mid.phys_dim
    
    @property
    def lognm(self):
        return self.mid.lognm
    
    def contract_left_env(self, Lenv, ll, psi):
        return tf._ProjMPO_contract_left_env(self.mid.data[ll+1].to(dtype=self.dtype, device=self.device), psi.data[ll+1], Lenv)
    
    def contract_right_env(self, Renv, rl, psi):
        return tf._ProjMPO_contract_right_env(self.mid.data[rl - 1].to(dtype=self.dtype, device=self.device), psi.data[rl - 1], Renv)
    

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

    
    def noiseterm(self, phi:MPS, drt='left'):
        assert self.nsite == 2, "Only two-site ProjMPO currently supported"
        if drt == 'right':  # 如果向右移动需要计算哦左侧的噪声？
            nt = tf._noise_proj_left(
                self.lproj(), self.mid[self.lpos+1], phi
            )
        elif drt == 'left':
            nt = tf._noise_proj_right(
                phi, self.mid[self.rpos-1], self.rproj()
            )
        else:
            raise ValueError(f"Unknown ortho: {drt}, should be 'left' or 'right'")
        return nt @ nt.conj().T



class ProjMPS(ProjOper):
    def __init__(self, M, nsite=2, ifnorm=False) -> None:
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
        self.mid = M
        self.ndim = 2
        super().__init__(L=len(M), dtype=M.dtype, device=M.device, nsite=nsite, ifnorm=ifnorm)
       
    @property
    def lognm(self):
        return self.mid.lognm
   
    def contract_left_env(self, Lenv, ll, phi):
        return tf._ProjMPS_contract_left_env(self.mid.data[ll+1].to(dtype=self.dtype, device=self.device), phi.data[ll+1], Lenv)
    
    def contract_right_env(self, Renv, rl, phi):
        return tf._ProjMPS_contract_right_env(self.mid.data[rl - 1].to(dtype=self.dtype, device=self.device), phi.data[rl - 1], Renv)
    
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
        return (pm.reshape(-1, 1).conj() @ pm.reshape(1, -1)) 

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
        return lambda v: ((pm.reshape(-1) @ v.reshape(-1)) * pm.reshape(*v.shape).conj()) 


class ProjSumMPO:
    def __init__(self, Hs:list[MPO], ifnorm) -> None:
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
        self.Hs = [ProjMPO(H, nsite=2, ifnorm=ifnorm) for H in Hs]
    

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
            #?? lognm? any better way?
            res = funcs[0](v) * tc.exp(self.Hs[0].mid.lognm)
            for i in range(1,len(funcs)):
                res += funcs[i](v)  * tc.exp(self.Hs[i].mid.lognm)
            return res
        return matmul

    def to_matrix(self):
        lognms = [tc.exp(H.mid.lognm).item() for H in self.Hs]
        return sum(coef*H.to_matrix() for H, coef in zip(self.Hs, lognms))
    
    def noiseterm(self, phi:MPS, drt='left'):
        nt = self.Hs[0].noiseterm(phi, drt)
        for H in self.Hs[1:]:
            nt += H.noiseterm(phi, drt)
        return nt

    @property
    def shape(self):
        return self.Hs[0].shape
    
    @property
    def lrlognm(self):
        return tc.tensor(0., dtype=tc.float64, device=self.Hs[0].mid.device)


class ProjMPOMPS:
    def __init__(self, H, mpsv, weight, ifnorm=False):
        if isinstance(H, MPO):
            self.PH = ProjMPO(H, ifnorm=ifnorm)
        else:
            self.PH = H
            self.lognm = H.lognm
        self.pm = [ProjMPS(m, ifnorm=ifnorm) for m in mpsv]
        self.weight = weight

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
        PHlrlognm = self.PH.lrlognm
        def matmul(v):
            res = matmul1(v)
            for i, matmulp in enumerate(matmul2):
                res += matmulp(v) * tc.exp(self.pm[i].lrlognm*2 - PHlrlognm - self.PH.lognm) * self.weight[i]
            return res
        return matmul

    def to_matrix(self):
        mat1 = self.PH.to_matrix()
        PHlrlognm = self.PH.lrlognm
        for i, p in enumerate(self.pm):
            mat1 += p.to_matrix() * tc.exp(p.lrlognm*2 - PHlrlognm - self.PH.lognm) * self.weight[i]

        # mat2 = self.PH.to_matrix()
        # PHlrlognm = self.PH.lrlognm
        # for i, p in enumerate(self.pm):
        #     mat2 += p.to_matrix() * tc.exp(p.lrlognm*2 - PHlrlognm - self.PH.lognm) * self.weight[i]
        
        return mat1
    
    def noiseterm(self, phi:MPS, drt='left'):
        return self.PH.noiseterm(phi, drt)

    @property
    def lrlognm(self):
        return self.PH.lrlognm
    