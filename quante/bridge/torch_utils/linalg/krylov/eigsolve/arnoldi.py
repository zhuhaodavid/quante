# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:18:32
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-29 16:02:50

import numpy as np
from scipy.linalg import schur
from ..factorizations.arnoldi import ArnoldiIterator
from .eigsolve import EIGSORT
from ..dense.linalg import permuteschur
from ..dense.reflector import householder
from ..krylovkit import ConvergenceInfo

usejulia = True
usecython = True

if usecython:
    import pyximport
    pyximport.install(language_level=3)
    from ..dense.hseqr import dhseqr_

if usejulia:
    from julia import Main
    Main.eval("using KrylovKit, LinearAlgebra")
    Main.eval("using KrylovKit: hschur!, permuteschur!, basistransform!, scale!!, schur2eigvals, schur2eigvecs, cols, apply, eigselector, _schursolve, eigsort, householder")

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

        l = Main.length(fact)
        B = fact.V.basis

        if np.isrealobj(T) and howmany < l and T[howmany, howmany-1] != 0:
            howmany += 1

        if converged > howmany:
            howmany = converged
        d = min(howmany, T.shape[1])
        TT = T[:d, :d]
        values = Main.schur2eigvals(TT)
        V = Main.schur2eigvecs(TT)

        V = U[:, :d] @ V
        vectors = [
            sum(B[i] * V[i,j] for i in range(l)) 
            for j in range(d)
        ]
        
        for i in range(len(values)):
            print(np.linalg.norm(
                A.numpy() @ vectors[i] - values[i] * vectors[i]
            ))

        exit()

        
    def schursolve(self, A, x0, howmany, which):
        krylovdim = self.krylovdim
        verbosity = self.verbosity

        Main.howmany = howmany
        Main.eager = eager = self.eager
        Main.maxiter = maxiter = self.maxiter
        Main.tol = tol = self.tol

        if howmany > self.krylovdim:
            raise ValueError(f"krylov dimension {self.krylovdim} too small to compute {howmany} eigenvalues")

        Aiter = ArnoldiIterator(A, x0, orth=self.orth)
        fact = Aiter.initialize(krylovdim, verbosity=verbosity-2)

        HH = np.zeros((krylovdim + 1, krylovdim), dtype=np.float64)
        UU = np.zeros((krylovdim, krylovdim), dtype=np.float64)

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
                T, U = schur(H)
                values = np.linalg.eigvals(H)
                by, rev = EIGSORT[which]
                if rev:
                    p = np.argsort([-by(v) for v in values], stable=True)
                else:
                    p = np.argsort([by(v) for v in values], stable=True)
                
                Main.T = T
                Main.U = U
                Main.p = p + 1
                res1, res2, _ = Main.eval("permuteschur!(T, U, p)")
                T[:] = np.array(res1)
                H[:] = T[:]
                U[:] = np.array(res2)

                U[K-1, :] *= β
                f[:] = U[K-1, :]

                converged = 0
                while converged < Main.length(Main.fact) and abs(f[converged]) <= tol:
                    converged += 1
                
                T = H
                if np.isrealobj(T) and 0 < converged < Main.length(Main.fact) and T[converged, converged-1] != 0:
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
                Main.H = H
                Main.keep = keep

                Main.eval("""
                for j in keep:-1:1
                    h, ν = householder(H, j + 1, 1:j, j)
                    H[j + 1, j] = ν
                    H[j + 1, 1:(j - 1)] .= 0
                    lmul!(h, H)
                    rmul!(view(H, 1:j, :), h')
                    rmul!(U, h')
                    # println(j)
                end
                copyto!(rayleighquotient(fact), H) # copy back into fact

                # Update B by applying U
                B = basis(fact)
                basistransform!(B, view(U, :, 1:keep))
                r = residual(fact)
                B[keep + 1] = scale!!(r, 1 / normres(fact))

                """)
                fact.shrink_(keep)
                numiter += 1

        T = np.array(Main.T)
        U = np.array(Main.U)
        return T, U, Main.fact, converged

    
        # krylovdim = self.krylovdim
        # maxiter = self.maxiter
        # if howmany > krylovdim:
        #     raise ValueError(f"krylov dimension {krylovdim} too small to compute {howmany} eigenvalues")
        
        # ## FIRST ITERATION: setting up
        # numiter = 1
        # # initialize arnoldi factorization
        # # _iter = ArnoldiIterator(A, x0, self.orth)
        # # fact = _iter.initialize(krylovdim, verbosity=self.verbosity-2)

        # Main.A = A.numpy()
        # Main.x0 = x0.numpy()
        # Main.krylovdim = krylovdim
        # Main.eval("orth = KrylovKit.ModifiedGramSchmidt2()")
        # Main.eval("iter = ArnoldiIterator(A, x0, orth)")
        # Main.eval("fact = initialize(iter; verbosity=0 - 2)")

        # numops = 1
        # Main.eval("sizehint!(fact, krylovdim)")
        # beta = Main.eval("normres(fact)")
        
        # # beta = fact.normres()
        # tol = self.tol

        # # allocate storage
        # HH = np.zeros((krylovdim + 1, krylovdim), dtype=np.float64)
        
        # # initialize storage
        # # K = len(fact) # == 1
        # K = Main.eval("length(fact)")
        # converged = 0
        # while True:
        #     # beta = fact.normres()
        #     # K = len(fact) 
        #     beta = Main.eval("normres(fact)")
        #     K = Main.eval("length(fact)")
            
        #     if beta <= tol:
        #         if K < howmany:
        #             print(f"Invariant subspace of dimension {K} (up to requested tolerance `tol = {tol}`), which is smaller than the number of requested eigenvalues (i.e. `howmany == {howmany}`); setting `howmany = {K}`.")
        #             howmany = K
                
        #     if (K == krylovdim) or (beta <= tol) or (self.eager and (K >= howmany)):  # process
                            
        #         H = HH[:K, :K]
        #         f = HH[K, :K]
        #         Main.H = H
        #         Main.eval("copyto!(H, rayleighquotient(fact))")
        #         H = Main.H
        #         # fact.rayleighquotient().copyto_(H)

        #         if usejulia:
        #             U = np.eye(H.shape[0], dtype=H.dtype, order='F')
        #             Main.U = U
        #             Main.H = H
        #             Main.eval("T, U, values = hschur!(H, U)")
        #             Main.eval("p = sortperm(values; by=abs, rev=true)")
        #             Main.eval("T, U = permuteschur!(T, U, p)")
        #             T = Main.T
        #             U = Main.U
        #         else:
        #             # todo: julia-krylov use zhseqr, but
        #             # the result is different from that 
        #             # obtained by cython, scipy.linalg.
        #             # cython_lapack.lapack.zhseqr, 
        #             # why is that ?
        #             T, U = schur(H, output='real')
        #             # todo: rewrite not using eigvals
        #             values = np.linalg.eigvals(H) 
        #             by, rev = EIGSORT[which]
        #             if rev:
        #                 p = np.argsort([-by(v) for v in values], stable=True)
        #             else:
        #                 p = np.argsort([by(v) for v in values], stable=True)
        #             T, U = permuteschur(T, U, p)                   
               
        #         H = T 
        #         f[:] = U[K-1, :] * beta
        #         converged = 0
        #         lenfact = Main.eval("length(fact)")
        #         while converged < lenfact and abs(f[converged]) <= tol:
        #             converged += 1
        #         lenfact = Main.eval("length(fact)")
        #         if np.isrealobj(T) and 0 < converged < lenfact and T[converged, converged-1] != 0:
        #             converged -= 1

        #         if converged >= howmany:
        #             break
        #         elif self.verbosity >= 1:
        #             msg = f"Arnoldi schursolve in iter {numiter}, krylovdim = {K}: "
        #             msg += f"{converged} values converged, normres = ({abs(f[0]):.2e}"
        #             for i in range(1,howmany):
        #                 msg += ", "
        #                 msg += f"{abs(f[i]):.2e}"
        #             msg += ")"
        #             print(msg)
            
        #     if K < krylovdim: # expand
        #         Main.eval("fact = expand!(iter, fact; verbosity=0 - 2)")
        #         # fact = fact.expand_(_iter, verbosity=self.verbosity-2)
        #         numops += 1
        #     else: # shrink
        #         if numiter == maxiter:
        #             break
                
        #         # Determine how many to keep
        #         keep = (3 * krylovdim + 2 * converged)//5 # strictly smaller than krylovdim since converged < howmany <= krylovdim, at least equal to converged
        #         if np.isrealobj(H) and H[keep, keep-1] != 0:  # we are in the middle of a 2x2 block
        #             keep += 1
        #             if keep >= krylovdim:
        #                 raise ValueError(f"krylov dimension {krylovdim} too small to compute {howmany} eigenvalues")

        #         # Restore Arnoldi form in the first keep columns
        #         for j in range(keep):
        #             H[keep, j] = f[j]
        #         for j in range(keep,0,-1):
        #             h, nu = householder(H, j, np.arange(j), j - 1)
        #             H[j, j-1] = nu
        #             H[j, :(j-1)] = 0.
        #             h.lmul_(H, j)
        #             h.rmul_(H[:j,:])
        #             h.rmul_(U)
                
        #         # fact.rayleighquotient().copy_from(H)
        #         # # Update B by applying U
        #         # fact.V.basistransform_(U[:, :keep])

        #         # r = fact.r
        #         # r.div_(fact.normres())
        #         # fact.V.copy(r, keep)

        #         # Shrink Arnoldi factorization
        #         # fact.shrink_(keep)

                
        #         Main.H = H
        #         Main.U = U
        #         Main.keep = keep
        #         Main.eval("copyto!(rayleighquotient(fact), H)")
        #         Main.eval("B = basis(fact)")
        #         Main.eval("basistransform!(B, view(U, :, 1:keep))")
        #         Main.eval("r = residual(fact)")
        #         Main.eval("B[keep + 1] = scale!!(r, 1 / normres(fact))")
        #         Main.eval("fact = shrink!(fact, keep)")
        #         numiter += 1

        #         # if numops == 140:
        #         #     print(fact.V.data[0,0].item())
        #         #     print(beta)
        #         #     exit()

        # # Implement the Schur decomposition and solve the eigenvalue problem
        # fact = Main.fact
        # return T, U, fact, converged, numiter, numops
