# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:18:32
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-28 21:26:32

import torch as tc
import numpy as np
from scipy.linalg import schur
from ..factorizations.arnoldi import ArnoldiIterator
from .eigsolve import EIGSORT
from ..dense.linalg import permuteschur
from ..dense.reflector import householder
from ..krylovkit import ConvergenceInfo

class Arnoldi:
    def __init__(self, krylovdim=30, maxiter=100, tol=1.e-12, orth='ModifiedGramSchmidt2', eager=False, verbosity=0):
        self.krylovdim = krylovdim
        self.maxiter = maxiter
        self.tol = tol
        self.orth = orth
        self.eager = eager
        self.verbosity = verbosity

    def eigsolve(self, A, x0, howmany, which):
        T, U, fact, converged, numiter, numops = self.schursolve(A, x0, howmany, which)
        if np.isrealobj(T) and howmany < len(fact) and T[howmany, howmany-1] != 0:
            howmany += 1
        if converged > howmany:
            howmany = converged
        d = min(howmany, T.shape[1]) 
        TT = T[:d, :d]

        # todo: julia-krylov use trevc, do we need that?
        values, V = np.linalg.eig(TT)
        V /= np.linalg.norm(V, axis=0, keepdims=True)        
        V = U[:, :d] @ V

        vectors = [
            sum(fact.V.basis[i] * V[i,j] for i in range(len(fact))) 
            for j in range(d)
        ]  # todo: can we do better?

        residuals = np.array([
            fact.r * V[-1,i] for i in range(d)
        ])
        normresiduals = [fact.normres() * abs(V[-1,i]) for i in range(d)]

        if self.verbosity > 0:
            if converged < howmany:
                print(
                    f"Arnoldi eigsolve finished without convergence after {numiter} iterations: \n"
                    f" *  {converged} eigenvalues converged\n"
                    f" *  norm of residuals = {normresiduals}\n"
                    f" *  number of operators = {numops}"
                )
            else:
                print(
                    f"Arnoldi eigsolve finished after {numiter} iterations:"
                    f" *  {converged} eigenvalues converged\n"
                    f" *  norm of residuals = {normresiduals}\n"
                    f" *  number of operators = {numops}"
                )
        
        return values, vectors, ConvergenceInfo(converged, residuals, normresiduals, numiter, numops)

    def schursolve(self, A, x0, howmany, which):
        krylovdim = self.krylovdim
        maxiter = self.maxiter
        if howmany > krylovdim:
            raise ValueError(f"krylov dimension {krylovdim} too small to compute {howmany} eigenvalues")
        
        ## FIRST ITERATION: setting up
        numiter = 1
        # initialize arnoldi factorization
        _iter = ArnoldiIterator(A, x0, self.orth)
        fact = _iter.initialize(krylovdim, verbosity=self.verbosity-2)
        numops = 1
        # fact.sizehint_(self.krylovdim)
        beta = fact.normres()
        tol = self.tol

        # allocate storage
        HH = np.zeros((krylovdim + 1, krylovdim), dtype=fact.dtype)
        
        # initialize storage
        K = len(fact) # == 1
        converged = 0
        while True:
            beta = fact.normres()
            K = len(fact) 
            if beta <= tol:
                if K < howmany:
                    print(f"Invariant subspace of dimension {K} (up to requested tolerance `tol = {tol}`), which is smaller than the number of requested eigenvalues (i.e. `howmany == {howmany}`); setting `howmany = {K}`.")
                    howmany = K
                
            if (K == krylovdim) or (beta <= tol) or (self.eager and (K >= howmany)):  # process
                            
                H = HH[:K, :K]
                f = HH[K, :K]
                fact.rayleighquotient().copyto_(H)
                
                # todo: julia-krylov use zhseqr, but
                # the result is different from that 
                # obtained by cython, scipy.linalg.
                # cython_lapack.lapack.zhseqr, 
                # why is that ?
                T, U = schur(H)
                values = np.diag(T)
                by, rev = EIGSORT[which]
                p = np.argsort([by(v) for v in values])
                if rev:
                    p = p[::-1]
                T, U = permuteschur(T, U, p)

                # # for benchmark with julia
                # import quante as qt
                # T, U = qt.basicfun.load_hdf5('../data_schur.h5', [f'{numops}/T', f'{numops}/U'])
                # T = T.transpose()
                # U = U.transpose()

                H = T 
                f[:] = U[K-1, :] * beta
                converged = 0
                while converged < len(fact) and abs(f[converged]) <= tol:
                    converged += 1
                if np.isrealobj(T):
                    raise NotImplementedError("real Schur not implemented")
                
                if converged >= howmany:
                    break
                elif self.verbosity >= 1:
                    msg = f"Arnoldi schursolve in iter {numiter}, krylovdim = {K}: "
                    msg += f"{converged} values converged, normres = ({abs(f[0]):.2e}"
                    for i in range(1,howmany):
                        msg += ", "
                        msg += f"{abs(f[i]):.2e}"
                    msg += ")"
                    print(msg)
            
            if K < krylovdim: # expand
                fact = fact.expand_(_iter, verbosity=self.verbosity-2)
                numops += 1
            else: # shrink
                if numiter == maxiter:
                    break
                
                # Determine how many to keep
                keep = (3 * krylovdim + 2 * converged)//5 # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged
                if np.isrealobj(H) and H[keep, keep-1] != 0:  # we are in the middle of a 2x2 block
                    raise NotImplementedError("real Schur not implemented")
                
                # Restore Arnoldi form in the first keep columns
                for j in range(keep):
                    H[keep, j] = f[j]
                for j in range(keep,0,-1):
                    h, nu = householder(H, j, np.arange(j), j - 1)
                    H[j, j-1] = nu
                    H[j, :(j-1)] = 0.
                    h.lmul_(H, j)
                    h.rmul_(H[:j,:])
                    h.rmul_(U)
                
                fact.rayleighquotient().copy_from(H)

                # Update B by applying U
                fact.V.basistransform_(U[:, :keep])
                r = fact.r
                r.div_(fact.normres())
                fact.V.data[keep].copy_(r)

                # Shrink Arnoldi factorization
                fact.shrink_(keep)
                numiter += 1

        # Implement the Schur decomposition and solve the eigenvalue problem
        return T, U, fact, converged, numiter, numops
