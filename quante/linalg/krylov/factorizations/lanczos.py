# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-30 19:20:44
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 13:40:49

import numpy as np
import warnings

from ..dense.orthonormal import OrthonormalBasis
from ..krylovkit import LinearAlgebraUtils

class LanczosIterator(LinearAlgebraUtils):
    def __init__(self, A, x0, orth, keepvecs=True, lau=None):
        self.operator = A
        self.x0 = x0
        if (not keepvecs and orth in [
            "ClassicalGramSchmidt2",
            "ModifiedGramSchmidt2",
            "ClassicalGramSchmidtIR",
            "ModifiedGramSchmidtIR"
        ]):
            raise ValueError("Cannot use reorthogonalization without keeping all Krylov vectors.")
        self.orth = orth
        self.keepvecs = keepvecs
        super().__init__(x0, lau)
        self.lau.update_device(x0)
    
    def initialize(self, krylovdim, verbosity):
        x0 = self.x0
        beta0 = self.lau.norm(x0)
        if np.isclose(beta0, 0):
            raise ValueError("initial vector should not have norm zero.")
        Ax0 = self.lau.apply(self.operator, x0)
        alpha = self.lau.inner(x0, Ax0) / (beta0 * beta0)
        v = self.lau.zeros_like(Ax0)
        self.lau.add_(v, x0, alpha=1/beta0)
        r = self.lau.div_(Ax0, beta0)
        beta_old = self.lau.norm(r)
        self.lau.sub_(r, v, alpha=alpha)
        beta = self.lau.norm(r)
        # possibly reorthogonalize
        if self.orth in ['ClassicalGramSchmidt2','ModifiedGramSchmidt2']:
            dalpha = self.lau.inner(v, r)
            alpha += dalpha
            self.lau.sub_(r, v, alpha=dalpha)
            beta = self.lau.norm(r)
        else:
            raise NotImplementedError(f"orthogonalization method {self.orth} not implemented")
        if verbosity >= 1:
            warn_nonhermitian(alpha, 0., beta)
        basis_num = krylovdim if self.keepvecs else 2
        V = OrthonormalBasis(v, basis_num, self.lau)
        alphas = [np.real(alpha)]
        betas = [beta]
        if verbosity > 3:
            warnings.info(
                f"Lanczos initiation at dimension 1: "
                f"subspace normres = {beta}"
            )
        return LanczosFactorization(1, V, alphas, betas, r, lau=self.lau)

class LanczosFactorization:
    def __init__(self, k, V, alphas, betas, r, lau):
        self.k = k  # current Krylov dimension
        self.V = V  # basis of length k
        self.alphas = alphas
        self.betas = betas
        self.r = r  # residual
        self.lau = lau
    
    def __len__(self):
        return self.k
    
    def length(self):
        return len(self.V)
    
    @property
    def dtype(self):
        return np.float64
    
    def normres(self):
        return self.betas[self.k-1]
    
    def expand_(self, iterator, verbosity=0):
        betaold = self.normres()
        r = self.r
        self.lau.div_(r, betaold)
        self.V.append(r)
        r, alpha, beta = self.lanczosrecurrence(
            iterator.operator, self.V, 
            betaold, iterator.orth
        )
        if verbosity >= 1:
            warn_nonhermitian(alpha, betaold, beta)
        
        self.alphas.append(np.real(alpha))
        self.betas.append(beta)

        if not iterator.keepvecs:
            self.V.popfirst() 
            # remove oldest V if not keepvecs
        
        self.k += 1
        self.r = r
        if verbosity > 3:
            warnings.info(
                f"Lanczos expansion to dimension {self.k}: subspace normres = {beta}"
            )
        return self

    def lanczosrecurrence(self, operator, V, beta, orth):
        if orth == "ModifiedGramSchmidt2":
            v = V.basis[-1]
            w = self.lau.apply(operator, v)
            w = self.lau.add_(w, V.basis[-2], -beta)
            w, alpha = self.orthogonalize_(w, v, orth)
            
            s = alpha
            for q in self.V.basis:
                w, s = self.orthogonalize_(w, q, orth)
            alpha += s
            beta = self.lau.norm(w)
            return w, alpha, beta
        else:
            raise NotImplementedError(f"orthogonalization method {orth} not implemented")

    def shrink_(self, k, verbosity):
        if len(self) != len(self.V):
            raise ValueError("we cannot shrink LanczosFactorization without keeping Lanczos vectors.")
        if len(self) <= k:
            return self
        
        V = self.V
        while len(V) > k + 1:
            V.pop()
        r = V.pop()
        self.alphas = self.alphas[:k]
        self.betas = self.betas[:k]
        self.k = k
        beta = self.normres()
        if verbosity > 3:
            warnings.info(
                f"Arnoldi reduction to dimension {k}: subspace normres = {beta}"
            )
        self.lau.mul_(r, self.normres())
        self.r = r


    def orthogonalize_(self, v, q, orth):
        if orth == "ModifiedGramSchmidt2":
            s = self.lau.inner(q, v)
            v = self.lau.add_(v, q, -s)
            return v, s
        else:
            raise NotImplementedError(f"reorthogonalization method {orth} not implemented")

def warn_nonhermitian(alpha, beta1, beta2):
    n = np.linalg.norm([alpha, beta1, beta2])
    # do we need calculate n?
    if abs(np.imag(alpha)) / n > np.finfo(type(n)).eps ** (2 / 5):
        warnings.warn(f"ignoring imaginary component {np.imag(alpha)} from total weight {n}: operator might not be hermitian?")
    
