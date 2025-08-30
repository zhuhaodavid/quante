# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-30 19:06:52
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 00:28:34

import warnings
import numpy as np
from scipy.linalg import eigh_tridiagonal

from ..krylovkit import KrylovDefault, ConvergenceInfo, EIGSORT
from ..factorizations.lanczos import LanczosIterator
from ..dense.reflector import householder

from ..krylovkit import LinearAlgebraUtils as lau

class Lanczos(KrylovDefault):
    def __init__(self, **kwargs):
        super().__init__()
        self.eager = kwargs.pop('eager', False)
        self.update_params(kwargs)

    def eigsolve(self, A, x0, howmany, which):
        D, U, fact, converged, numiter, numops = self._eigsolve(
            A, x0, howmany, which
        )
             
        howmany_p = howmany
        if converged > howmany:
            howmany_p = converged
        elif len(D) < howmany:
            howmany_p = len(D)
        values = D[:howmany_p]

        # Compute eigenvectors
        V = U[:,:howmany_p]

        # Compute convergence information
        vectors = fact.V.basistransform(V)
        # residuals = [fact.r * v for v in V[-1,:]]
        # residuals is the list constaining
        #     `residual[i] = f(vectors[i]) - values[i] * vectors[i]`
        # however it is too large, we do not return it
        normresiduals = [fact.normres() * abs(v) for v in V[-1,:]]

        if (converged < howmany) and self.verbosity >= 1:
            warnings.warn(
                f"Lanczos eigsolve stopped without convergence after {numiter} iterations:"
                f" * {converged} eigenvalues converged"
                f" * norm of residuals = {fact.normres()}"
                f" * number of operations = {numops}"
            )
        elif self.verbosity >= 2:
            warnings.info(
                f"Lanczos eigsolve finished after {numiter} iterations: "
                f" * {converged} eigenvalues converged"
                f" * norm of residuals = {fact.normres()}"
                f" * number of operations = {numops}"
            )
        return values, vectors, ConvergenceInfo(
            converged, normresiduals, numiter, numops
        )

    
    def _eigsolve(self, A, x0, howmany, which):
        krylovdim = self.krylovdim
        maxiter = self.maxiter
        if howmany > krylovdim:
            raise ValueError(
                f"krylov dimension {krylovdim} too small to compute {howmany} eigenvalues"
            )
        
        ## FIRST ITERATION: setting up
        # Initialize Lanczos factorization
        Liter = LanczosIterator(A, x0, self.orth)
        fact = Liter.initialize(krylovdim, verbosity=self.verbosity)
        numops = 1
        numiter = 1
        beta = fact.normres()
        tol = self.tol
        
        # allocate storage, do we need this?
        HH = np.zeros((krylovdim + 1, krylovdim), dtype=fact.dtype, order='F')
        UU = np.zeros((krylovdim, krylovdim), dtype=fact.dtype, order='F')

        converged = 0
        while True:
            beta = fact.normres()
            K = fact.length()

            # diagonalize Krylov factorization           
            if beta <= tol and K < howmany:
                if self.verbosity >= 1:
                    warnings.warn(
                        f"Invariant subspace of dimension {K} "
                        f"(up to requested tolerance `tol = {tol}`), "
                        f"which is smaller than the number of requested eigenvalues "
                        f"(i.e. `howmany == {howmany}`)."
                    )
            
            if K == krylovdim or beta <= tol or (self.eager and K >= howmany): # process
               
                U = UU[:K, :K]
                U[:] = 0.
                np.fill_diagonal(U, 1.0)
                f = HH[K, :K]
                
                # compute eigenvalues
                if K == 1:
                    D = np.array([fact.alpha[0]], dtype=fact.dtype)
                    f[0] = beta
                    converged = (beta <= tol)
                else:
                    D, U = eigh_tridiagonal(
                        fact.alphas[:K], fact.betas[:K-1], 
                        lapack_driver='stemr')
                    by, rev = EIGSORT[which]
                    p = sorted(range(len(D)), 
                               key=lambda i: by(D[i]), reverse=rev)
                    D[:] = D[p]
                    U[:] = U[:, p]
                    f[:] = U[K-1, :] * beta
                    converged = 0
                    while converged < K and abs(f[converged]) <= tol:
                        converged += 1
                
                if converged >= howmany or beta <= tol:
                    break
                elif self.verbosity >= 3:
                    warnings.info(
                        f"Lanczos eigsolve in iteration {numiter}, step = {K}: "
                        f"{converged} values converged, normres = {beta}"
                    )
            
            if K < krylovdim: # expand Krylov factorization
                fact.expand_(Liter, verbosity=self.verbosity)
                numops += 1
            else: ## shrink and restart
                if numiter == maxiter:
                    break
                 
                
                # Determine how many to keep
                keep = (3 * krylovdim + 2 * converged) // 5
                # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged

                # Restore Lanczos form in the first keep columns
                H = HH[:keep+1, :keep]
                H[:] = 0.
                for j in range(keep):
                    H[j,j] = D[j]
                    H[keep, j] = f[j]
                for j in range(keep, 0, -1):
                    h, nu = householder(H, j, np.arange(j), j-1)
                    H[j, j-1] = nu
                    H[j, :(j-1)] = 0.
                    h.lmul_(H)
                    h.rmul_(H[:j, :])
                    h.rmul_(U)
                for j in range(keep):
                    fact.alphas[j] = H[j, j]
                    fact.betas[j] = H[j+1, j]
                   
                # Update B by applying U using Householder reflections
                fact.V.basistransform_(U[:, :keep])
                lau.div_(fact.r, fact.normres())
                fact.V.set(keep, fact.r)
                # Shrink Lanczos factorization
                fact.shrink_(keep, self.verbosity)
        return D, U, fact, converged, numiter, numops