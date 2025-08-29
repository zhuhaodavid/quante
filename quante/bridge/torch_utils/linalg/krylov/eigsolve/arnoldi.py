# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:18:32
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 02:37:12

import numpy as np
from ..factorizations.arnoldi import ArnoldiIterator
from .eigsolve import EIGSORT
from ......linalg.decomp.schur import permuteschur_, hschur_, schur2eigvals, schur2eigvecs
from ..dense.reflector import householder, _householder_
from ..krylovkit import ConvergenceInfo

# usejulia = True
# if usejulia:
#     from julia import Main
#     Main.eval("using KrylovKit, LinearAlgebra")
#     Main.eval("using KrylovKit: hschur!, permuteschur!, basistransform!, scale!!, schur2eigvals, schur2eigvecs, cols, apply, eigselector, _schursolve, eigsort, householder, _householder!, Householder")

class Arnoldi:
    def __init__(self, krylovdim=30, maxiter=100, tol=1.e-12, orth='ModifiedGramSchmidt2', eager=False, verbosity=0):
        self.krylovdim = krylovdim
        self.maxiter = maxiter
        self.tol = tol
        self.orth = orth
        self.eager = eager
        self.verbosity = verbosity

    def eigsolve(self, A, x0, howmany, which):
        T, U, fact, converged = self.schursolve(A, x0, howmany, which)

        l = len(fact)
        B = fact.V.basis

        if np.isrealobj(T) and howmany < l and T[howmany, howmany-1] != 0:
            howmany += 1

        if converged > howmany:
            howmany = converged
        d = min(howmany, T.shape[1])
        TT = T[:d, :d]
        values = schur2eigvals(TT)
        V = schur2eigvecs(TT)
        # V = np.linalg.eig(TT)[1]

        V = U[:, :d] @ V
        vectors = [
            sum(B[i] * V[i,j] for i in range(l)) 
            for j in range(d)
        ]
        
        for i in range(len(values)):
            print(np.linalg.norm(
                A.numpy() @ vectors[i].numpy() - values[i] * vectors[i].numpy()
            ))
        return values, vectors

        # exit()

        
    def schursolve(self, A, x0, howmany, which):
        krylovdim = self.krylovdim
        verbosity = self.verbosity
        eager = self.eager
        maxiter = self.maxiter
        tol = self.tol

        # Main.howmany = howmany
        # Main.eager = eager = self.eager
        # Main.maxiter = maxiter = self.maxiter
        # Main.tol = tol = self.tol

        if howmany > self.krylovdim:
            raise ValueError(f"krylov dimension {self.krylovdim} too small to compute {howmany} eigenvalues")

        Aiter = ArnoldiIterator(A, x0, orth=self.orth)
        fact = Aiter.initialize(krylovdim, verbosity=verbosity-2)

        HH = np.zeros((krylovdim + 1, krylovdim), dtype=fact.dtype, order='F')
        UU = np.zeros((krylovdim, krylovdim), dtype=fact.dtype, order='F')

        β = fact.normres()
        K = fact.length()

        # create a dict
        converged = 0
        numiter = 1
        numops = 1
        while True:
            β = fact.normres()
            K = fact.length()

            if β <= tol:
                if K < howmany:
                    print("Invariant subspace of dimension ", K, " (up to requested tolerance `tol = ", tol, "`), which is smaller than the number of requested eigenvalues (i.e. `howmany == ", howmany, "`); setting `howmany = ", K, "`.")
            
            if K == krylovdim or β <= tol or (eager and K >= howmany): # process
                H = HH[:K, :K]
                U = UU[:K, :K]
                f = HH[K, :K]

                U[:] = 0.
                for i in range(U.shape[0]):
                    U[i,i] = 1.0
                
                fact.rayleighquotient().copyto_(H)
                T, U, values = hschur_(
                    np.asfortranarray(H), np.asfortranarray(U)
                )
                by, rev = EIGSORT[which]
                if rev:
                    p = np.argsort([-by(v) for v in values], stable=True)
                else:
                    p = np.argsort([by(v) for v in values], stable=True)
                T, U = permuteschur_(T, U, p)
                H[:] = T
                f[:] = U[K-1, :] * β

                converged = 0
                while converged < fact.length() and abs(f[converged]) <= tol:
                    converged += 1
                
                T = H
                if np.isrealobj(T) and 0 < converged < fact.length() and T[converged, converged-1] != 0:
                    converged -= 1

                if converged >= howmany:
                    break

            if K < krylovdim: # expand
                fact.expand_(Aiter, verbosity=verbosity - 2)
                numops += 1
            else: # shrink
                if numiter == maxiter:
                    break
                keep = (3 * krylovdim + 2 * converged)//5
                if np.isrealobj(H) and H[keep, keep-1] != 0:  # we are in the middle of a 2x2 block
                    keep += 1
                    if keep >= krylovdim:
                        raise ValueError(f"krylov dimension {krylovdim} too small to compute {howmany} eigenvalues")
                
                for j in range(keep):
                    H[keep, j] = f[j]

                # U = np.array(Main.U)
                for j in range(keep, 0, -1):
                    h, nu = householder(H, j, np.arange(j), j-1)
                    H[j, j-1] = nu
                    H[j, :(j-1)] = 0.
                    h.lmul_(H)
                    h.rmul_(H[:j, :])
                    h.rmul_(U)

                fact.rayleighquotient().copy_from(H)
                fact.V.basistransform_(U[:, :keep])
                fact.r.div_(fact.normres())
                fact.V.set(keep, fact.r)
                fact.shrink_(keep)
                numiter += 1

        return T, U, fact, converged
