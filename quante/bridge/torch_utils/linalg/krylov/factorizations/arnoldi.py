# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:25:08
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 02:29:03

import torch as tc
import numpy as np
from ..orthonormal import OrthonormalBasis
from ..dense.packedhessenberg import PackedHessenberg

# from julia import Main
# Main.eval("using KrylovKit: ArnoldiFactorization, scale, arnoldirecurrence!!")


class ArnoldiFactorization:
    def __init__(self, k, V, H, r):
        self.k = k  # current Krylov dimension
        self.V = V  # basis of length k
        self.H = H  # stores the Hessenberg matrix in packed form
        self.r = r  # residual
        # self.data = Main.fact
    
    def __len__(self):
        # return Main.length(self.data)
        return self.k
    
    def length(self):
        # return Main.length(self.data)
        return len(self.V)
    
    @property
    def dtype(self):
        return self.H.dtype
    #     return self.H.dtype
        # if np.isrealobj(self.H):
        #     return tc.float64
        # else:
        #     return tc.complex128
    
    # def sizehint_(self, n):
    #     self.V.basis = self.V.basis[:n]  # todo: 是否会有内存问题
    #     self.H = self.H[:(n * n + 3 * n) >> 1]

    def normres(self):
        # return Main.normres(self.data)
        return abs(self.H[-1])
    
    def orthogonalize_(self, v, x, orth):
        b = self.V.basis
        if orth == 'ModifiedGramSchmidt2':
            v, s = self.orthogonalize_(v, x, 'ModifiedGramSchmidt')
            return self.reorthogonalize_(v, s, 'ModifiedGramSchmidt')
        elif orth == 'ModifiedGramSchmidt':
            # # !! main consumption
            for i, q in enumerate(b):
                s = tc.vdot(q, v)
                v.sub_(q, alpha=s)
                x[i] = s
            return v, x
            # tmp = (b.conj() @ v)
            # v.sub_(b.T @ tmp)
            # x[:] = tmp.cpu().numpy()
            # return v, x
        else:
            raise NotImplementedError(f"orthogonalization method {orth} not implemented")

    def reorthogonalize_(self, v, x, orth):
        b = self.V.basis
        if orth == 'ModifiedGramSchmidt':
            # !! main consumption
            for i, q in enumerate(b):
                s = tc.vdot(q, v)
                v.sub_(q, alpha=s)
                x[i] += s
            return v, x
            # ?? why this is slower ??
            # tmp = (b.conj() @ v)
            # v.sub_(b.T @ tmp)
            # x[:] += tmp.cpu().numpy()
            # return v, x
        else:
            raise NotImplementedError(f"reorthogonalization method {orth} not implemented")
        
    def recurrence_(self, operator, h, orth):
        w = operator @ self.V.basis[-1]
        r, h = self.orthogonalize_(w, h, orth)
        return r, tc.linalg.norm(r).item()
    
    # def update_from_jl(self):
    #     self.k = self.data.k
    #     self.H = np.array(self.data.H)
    #     self.r = tc.tensor(self.data.r)
    #     for i in range(len(list(self.data.V))):
    #         self.V.data[i] = tc.tensor(list(self.data.V)[i])
    #     self.V.num = Main.length(self.data)
    
    # def update_to_jl(self):
    #     Main.jlk = self.k
    #     Main.jlH = self.H
    #     Main.jlr = self.r.numpy()
    #     jlV = []
    #     for i in range(self.V.num):
    #         jlV.append(self.V.data[i].numpy())
    #     Main.jlV = jlV
    #     Main.eval("""
    #     fact.V = OrthonormalBasis(jlV)
    #     fact.k = jlk
    #     fact.H = jlH
    #     fact.r = jlr
    #     """)
        
   
    def expand_(self, iterator, verbosity=0):
        # self.update_from_jl()
        
        self.k += 1
        k = self.k
        r = self.r
        H = self.H
        beta = self.normres()
        self.V.append(r/beta)
        m = len(H)
        tmp = np.zeros(m+k+1, dtype=H.dtype)
        tmp[:len(H)] = H
        H = tmp
        r, beta = self.recurrence_(iterator.operator, H[m:m+k], iterator.orth) 
        H[m+k] = beta
        self.r = r
        self.H = H
        
        # self.update_to_jl()
        return self
       

    def rayleighquotient(self):
        data, n = self.H, self.k
        return PackedHessenberg(data, n)
    
    def shrink_(self, k):
        # Main.keep = k
        # Main.eval("shrink!(fact, keep)")
        # return 
        if len(self) <= k:
            return self
        
        V = self.V
        H = self.H
        while len(V) > k + 1:
            V.pop()
        r = V.pop()
        self.H = H[:((k * k + 3 * k) >> 1)]
        self.k = k
        self.r = r * self.normres()

class ArnoldiIterator:
    def __init__(self, A, x0, orth):
        self.operator = A
        self.x0 = x0
        self.orth = orth

    def initialize(self, krylovdim, verbosity=0):
        
        # Main.krylovdim = krylovdim
        # Main.verbosity = verbosity+2
        # Main.eval("""
        # # initialize arnoldi factorization
        # fact = initialize(iter; verbosity=verbosity-2)
        # sizehint!(fact, krylovdim)
        # """)
        # return None, ArnoldiFactorization(Main.fact)


        # initialize without using eltype
        x0 = self.x0
        beta0 = tc.linalg.norm(x0).item()
        if np.isclose(beta0, 0):
            raise ValueError("initial vector should not have norm zero")
        
        A = self.operator
        Ax0 = A @ x0  # todo: generalize
        alpha = (tc.vdot(x0, Ax0) / (beta0 * beta0)).item()
        
        v = tc.zeros_like(x0)
        v.add_(x0, alpha=1/beta0)

        r = tc.zeros_like(x0)
        r.add_(Ax0, alpha=1/beta0)
        beta_old = tc.linalg.norm(r)
        r.sub_(v, alpha=alpha)
        beta = tc.linalg.norm(r).item()

        # possibly reorthogonalize
        if self.orth in ['ClassicalGramSchmidt2','ModifiedGramSchmidt2']:
            dalpha = tc.vdot(v, r).item()
            alpha += dalpha
            r.sub_(v, alpha=dalpha)
            beta = tc.linalg.norm(r).item()
        else:
            raise NotImplementedError(f"orthogonalization method {self.orth} not implemented")
        
        V = OrthonormalBasis(v, krylovdim)
        H = np.array([alpha, beta])
        if verbosity > 0:
            print(f"Arnoldi iteration step 1: normres = {beta}")
        
        # Main.A = A.numpy()
        # Main.x0 = x0.numpy()
        # Main.eval("iter = ArnoldiIterator(A, x0, KrylovKit.ModifiedGramSchmidt2())")
        # Main.juliav = v.numpy()
        # Main.eval("V = OrthonormalBasis([juliav])")
        # Main.H = H
        # Main.r = r.numpy()
        # Main.krylovdim = krylovdim
        # Main.verbosity = verbosity + 2
        # Main.eval("fact = ArnoldiFactorization(1, V, H, r)")
        # Main.eval("sizehint!(fact, krylovdim)")
        
        return ArnoldiFactorization(1, V, H, r)
    
