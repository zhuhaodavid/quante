# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:18:32
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 14:45:15

import numpy as np
from ..factorizations.arnoldi import ArnoldiIterator
from .eigsolve import EIGSORT
from ......linalg.decomp.schur import permuteschur_, hschur_, schur2eigvals, schur2eigvecs
from ..dense.reflector import householder, _householder_
from ..krylovkit import ConvergenceInfo, KrylovDefault
import warnings

# usejulia = True
# if usejulia:
#     from julia import Main
#     Main.eval("using KrylovKit, LinearAlgebra")
#     Main.eval("using KrylovKit: hschur!, permuteschur!, basistransform!, scale!!, schur2eigvals, schur2eigvecs, cols, apply, eigselector, _schursolve, eigsort, householder, _householder!, Householder")

class Arnoldi(KrylovDefault):
    def __init__(self, **kwargs):
        super().__init__()
        self.eager = kwargs.pop('eager', False)
        self.update_params(kwargs)

    def eigsolve(self, A, x0, howmany, which):
        T, U, fact, converged, numiter, numops = self.schursolve(A, x0, howmany, which)
        howmany_p = howmany
        if np.isrealobj(T) and howmany < fact.length() and T[howmany, howmany-1] != 0:
            howmany_p += 1
        elif T.shape[0] < howmany:
            howmany_p = T.shape[0]
        if converged > howmany:
            howmany_p = converged
        TT = T[:howmany_p, :howmany_p]
        values = schur2eigvals(TT)

        # Compute eigenvectors
        V = U[:, :howmany_p] @ schur2eigvecs(TT)
        vectors = fact.V.basistransform(V)
        residuals = [fact.r * v for v in V[-1,:]]
        normresiduals = [fact.normres() * abs(v) for v in V[-1,:]]
        
        if (converged < howmany) and self.verbosity >= 1:
            warnings.warn(
                f"Arnoldi eigsolve stopped without convergence after {numiter} iterations:"
                f" * {converged} eigenvalues converged"
                f" * norm of residuals = {fact.normres()}"
                f" * number of operations = {numops}"
            )
        elif self.verbosity >= 2:
            warnings.info(
                f"Arnoldi eigsolve finished after {numiter} iterations: "
                f" * {converged} eigenvalues converged"
                f" * norm of residuals = {fact.normres()}"
                f" * number of operations = {numops}"
            )
        return values, vectors, ConvergenceInfo(
            converged, residuals, normresiduals, numiter, numops
        )
        
    def schursolve(self, A, x0, howmany, which):
        krylovdim = self.krylovdim
        maxiter = self.maxiter
        
        if howmany > self.krylovdim:
            raise ValueError(
                f"krylov dimension {krylovdim} too small to compute {howmany} eigenvalues"
            )
        
        ## FIRST ITERATION: setting up
        numiter = 1
        # initialize arnoldi factorization
        Aiter = ArnoldiIterator(A, x0, orth=self.orth)
        fact = Aiter.initialize(krylovdim, verbosity=self.verbosity-2)
        numops = 1
        beta = fact.normres()
        tol = self.tol

        # allocate storage
        HH = np.zeros((krylovdim + 1, krylovdim), dtype=fact.dtype, order='F')
        UU = np.zeros((krylovdim, krylovdim), dtype=fact.dtype, order='F')
        ff = np.zeros(krylovdim, dtype=fact.dtype, order='F')
        
        # initialize storage
        K = fact.length()  # == 1
        converged = 0
        while True:
            beta = fact.normres()
            K = fact.length()

            if beta <= tol and K < howmany:
                if self.verbosity >= 1:
                    warnings.warn(
                        f"Invariant subspace of dimension {K} "
                        f"(up to requested tolerance `tol = {tol}`), "
                        f"which is smaller than the number of requested eigenvalues "
                        f"(i.e. `howmany == {howmany}`)."
                    )

            if K == krylovdim or beta <= tol or (self.eager and K >= howmany): # process
                H = HH[:K, :K]
                U = UU[:K, :K]
                f = ff[:K]

                U[:] = 0.
                for i in range(U.shape[0]):
                    U[i,i] = 1.0
                fact.rayleighquotient().copyto_(H)

                # compute dense schur factorization
                T, U, values = hschur_(H, U)
                by, rev = EIGSORT[which]
                if rev:
                    p = np.argsort([-by(v) for v in values], stable=True)
                else:
                    p = np.argsort([by(v) for v in values], stable=True)
                T, U = permuteschur_(T, U, p)
                f[:] = np.conj(U[K-1, :]) * beta
                converged = 0
                factlength = fact.length()
                while converged < factlength and abs(f[converged]) <= tol:
                    converged += 1
                if (np.isrealobj(T) and 0 < converged < factlength 
                    and T[converged, converged-1] != 0):
                    converged -= 1

                if converged >= howmany or beta <= tol:
                    break
                elif self.verbosity >= 3:
                    warnings.info(
                        f"Arnoldi schursolve in iteration {numiter}, step = {K}: "
                        f"{converged} values converged, normres = {abs(f[:howmany])**2}"
                    )

            if K < krylovdim: # expand
                fact.expand_(Aiter, verbosity=self.verbosity)
                numops += 1
            else: # shrink
                if numiter == maxiter:
                    break
                
                # Determine how many to keep
                keep = (3 * krylovdim + 2 * converged)//5
                # strictly smaller than krylovdim since converged < howmany <= krylovdim, 
                # at least equal to converged

                if not np.isclose(H[keep, keep-1], 0):
                    # we are in the middle of a 2x2 block; this cannot happen 
                    # if keep == converged, so we can decrease keep. 
                    # however, we have to make sure that we do not end up with keep = 0
                    if keep > 1:
                        keep -= 1 # conservative choice
                    else:
                        keep += 1
                        if krylovdim == 2:
                            if self.verbosity >= 1:
                                warnings.warn(
                                    "Arnoldi iteration got stuck in a 2x2 block, "
                                    "consider increasing the Krylov dimension"
                                )
                            
                # Restore Arnoldi form in the first keep columns before shrinking
                restorearnoldiform_(U, H, f, keep)
                # Copy H back into compact Hessenberg form
                fact.rayleighquotient().copy_from(H)
                # Update the basis
                fact.V.basistransform_(U[:, :keep])
                fact.r.div_(fact.normres())
                fact.V.set(keep, fact.r)
                # Everything is set up to shrink Arnoldi factorization
                fact.shrink_(keep)
                numiter += 1

        return T, U, fact, converged, numiter, numops

def restorearnoldiform_(U, H, f, keep):
    for j in range(keep):
        H[keep, j] = np.conj(f[j])

    # U = np.array(Main.U)
    for j in range(keep, 0, -1):
        h, nu = householder(H, j, np.arange(j), j-1)
        H[j, j-1] = nu
        H[j, :(j-1)] = 0.
        h.lmul_(H)
        h.rmul_(H[:j, :])
        h.rmul_(U)

