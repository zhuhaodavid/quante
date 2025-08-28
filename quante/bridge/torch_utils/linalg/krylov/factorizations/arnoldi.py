# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:25:08
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-28 22:12:33

import torch as tc
import numpy as np
from ..orthonormal import OrthonormalBasis
from ..dense.packedhessenberg import PackedHessenberg

class ArnoldiFactorization:
    def __init__(self, k, V, H, r):
        self.k = k  # current Krylov dimension
        self.V = V  # basis of length k
        self.H = H  # stores the Hessenberg matrix in packed form
        self.r = r  # residual
    
    def __len__(self):
        return self.k
    
    @property
    def dtype(self):
        return self.H.dtype
        # if np.isrealobj(self.H):
        #     return tc.float64
        # else:
        #     return tc.complex128
    
    # def sizehint_(self, n):
    #     self.V.basis = self.V.basis[:n]  # todo: 是否会有内存问题
    #     self.H = self.H[:(n * n + 3 * n) >> 1]

    def normres(self):
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
    
    def expand_(self, iterator, verbosity=0):
        self.k += 1
        k = self.k
        V = self.V
        H = self.H
        r = self.r
        beta = self.normres()
        V.data[V.num].copy_(r / beta)
        V.num += 1
        m = len(H)
        tmp = np.zeros(m+k+1, dtype=H.dtype)
        tmp[:len(H)] = H
        H = tmp
        # r, β = arnoldirecurrence!!(iter.operator, V, view(H, (m + 1):(m + k)), iter.orth)
        r, beta = self.recurrence_(iterator.operator, H[m:m+k], iterator.orth)
        H[m+k] = beta
        self.r = r
        self.H = H
        if verbosity > 0:
            print(f"Arnoldi iteration step {k}: normres = {beta}")
        return self

    def rayleighquotient(self):
        data, n = self.H, self.k
        return PackedHessenberg(data, n)
    
    def shrink_(self, k):
        if len(self) <= k:
            return self
        
        V = self.V
        H = self.H
        while V.num > k + 1:
            V.num -= 1
        r = V.basis[-1]
        V.num -= 1
        self.H = H[:((k * k + 3 * k) >> 1)]
        self.k = k
        self.r = r * self.normres()

class ArnoldiIterator:
    def __init__(self, A, x0, orth):
        self.operator = A
        self.x0 = x0
        self.orth = orth

    def initialize(self, krylovdim, verbosity=0):
        # initialize without using eltype
        x0 = self.x0
        A = self.operator
        v = tc.zeros((krylovdim, len(x0)), dtype=x0.dtype)

        beta0 = tc.linalg.norm(x0).item()
        if np.isclose(beta0, 0):
            raise ValueError("initial vector should not have norm zero")
        Ax0 = A @ x0  # todo: generalize
        alpha = (tc.vdot(x0, Ax0) / (beta0 * beta0)).item()
        # this line determines the vector type that we will henceforth use
        v[0].copy_(x0)
        v[0].div_(beta0)
        Ax0.div_(beta0)
        r = Ax0
        beta_old = tc.linalg.norm(r)
        r.sub_(v[0], alpha=alpha)
        beta = tc.linalg.norm(r).item()
        # possibly reorthogonalize
        if self.orth in ['ClassicalGramSchmidt2','ModifiedGramSchmidt2']:
            dalpha = tc.vdot(v[0], r).item()
            alpha += dalpha
            r.sub_(v[0], alpha=dalpha)
            beta = tc.linalg.norm(r).item()
        else:
            raise NotImplementedError(f"orthogonalization method {self.orth} not implemented")
        V = OrthonormalBasis(v, 1)
        H = np.array([alpha, beta])
        if verbosity > 0:
            print(f"Arnoldi iteration step 1: normres = {beta}")
        return ArnoldiFactorization(1, V, H, r) 

