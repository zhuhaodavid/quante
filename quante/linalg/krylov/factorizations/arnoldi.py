# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:25:08
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 22:04:23

import warnings
import numpy as np

from ..orthonormal import OrthonormalBasis
from ..dense.packedhessenberg import PackedHessenberg
from ..krylovkit import LinearAlgebraUtils as lau
from ..krylovkit import overload_methods

class ArnoldiIterator:
    def __init__(self, A, x0, orth):
        self.operator = A
        self.x0 = x0
        self.orth = orth
        overload_methods(x0)
        lau.update_device(x0)
    
    def initialize(self, krylovdim, verbosity=0):
        x0 = self.x0
        beta0 = lau.norm(x0)
        if np.isclose(beta0, 0):
            raise ValueError("initial vector should not have norm zero")
        Ax0 = lau.apply(self.operator, x0)
        alpha = lau.inner(x0, Ax0) / (beta0 * beta0)
        v = lau.zeros_like(Ax0)
        lau.add_(v, x0, alpha=1/beta0)
        r = lau.div_(Ax0, beta0)
        beta_old = lau.norm(r)
        lau.sub_(r, v, alpha=alpha)
        # should we use real(dα) here?
        beta = lau.norm(r)
        # possibly reorthogonalize
        if self.orth in ['ClassicalGramSchmidt2','ModifiedGramSchmidt2']:
            dalpha = lau.inner(v, r)
            alpha += dalpha
            lau.sub_(r, v, alpha=dalpha)
             # should we use real(dα) here?
            beta = lau.norm(r)
        else:
            raise NotImplementedError(f"orthogonalization method {self.orth} not implemented")
        V = OrthonormalBasis(v, krylovdim)
        H = np.array([alpha, beta])
        if verbosity > 3:
            warnings.info(
                f"Arnoldi iteration step 1: normres = {beta}"
            )
        return ArnoldiFactorization(1, V, H, r)
    

class ArnoldiFactorization:
    def __init__(self, k, V, H, r):
        self.k = k  # current Krylov dimension
        self.V = V  # basis of length k
        self.H = H  # stores the Hessenberg matrix in packed form
        self.r = r  # residual
    
    def __len__(self):
        return self.k
    
    def length(self):
        return len(self.V)
    
    @property
    def dtype(self):
        return self.H.dtype

    def normres(self):
        return abs(self.H[-1])

    def expand_(self, iterator, verbosity=0):
        self.k += 1
        k = self.k
        r = self.r
        H = self.H
        beta = self.normres()
        self.V.append(r/beta)
        m = len(H)
        H = np.resize(H, m + k + 1)
        r, beta = self.recurrence_(
            iterator.operator, H[m:m+k], iterator.orth
        ) 
        H[m + k] = beta
        self.r = r
        self.H = H
        if verbosity > 3:
            warnings.info(
                f"Arnoldi expansion to dimension {k}: "
                f"subspace normres = {beta}"
            )
        return self

    def recurrence_(self, operator, h, orth):
        w = lau.apply(operator, self.V.basis[-1])
        r, h = self.orthogonalize_(w, h, orth)
        return r, lau.norm(r)
     
    def orthogonalize_(self, v, x, orth):
        b = self.V.basis
        if orth == 'ModifiedGramSchmidt2':
            v, s = self.orthogonalize_(v, x, 'ModifiedGramSchmidt')
            return self.reorthogonalize_(v, s, 'ModifiedGramSchmidt')
        elif orth == 'ModifiedGramSchmidt':
            # # !! main consumption
            for i, q in enumerate(b):
                s = lau.inner(q, v)
                lau.sub_(v, q, alpha=s)
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
                s = lau.inner(q, v)
                lau.sub_(v, q, alpha=s)
                x[i] += s
            return v, x
            # ?? why this is slower ??
            # tmp = (b.conj() @ v)
            # v.sub_(b.T @ tmp)
            # x[:] += tmp.cpu().numpy()
            # return v, x
        else:
            raise NotImplementedError(f"reorthogonalization method {orth} not implemented")

    

    def rayleighquotient(self):
        data, n = self.H, self.k
        return PackedHessenberg(data, n)

    def shrink_(self, k, verbosity):
        if len(self) <= k:
            return self
        
        V = self.V
        H = self.H
        while len(V) > k + 1:
            V.pop()
        r = V.pop()
        self.H = H[:((k * k + 3 * k) >> 1)]
        self.k = k
        beta = self.normres()
        if verbosity > 3:
            warnings.info(
                f"Arnoldi reduction to dimension {k}: subspace normres = {beta}"
            )
        lau.mul_(r, self.normres())
        self.r = r


