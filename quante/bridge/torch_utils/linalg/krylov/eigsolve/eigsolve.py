# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:19:37
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 15:03:41

import numpy as np
from ..krylovkit import LinearAlgebraUtils as lau

def eigsolve(
    A,
    x0,
    howmany=1,
    which='LM',
    issymmetric=False,
    ishermitian=False,
    **kwargs
):
    """Compute at least `howmany` eigenvalues from the linear map encoded in the matrix `A` or by
    the function `A`. Return eigenvalues, eigenvectors and a `ConvergenceInfo`.
    

    Parameters
    ----------
    A : torch.tensor or callable
        a linear map.
        can be any object that has method `__matmul__` used by `LinearAlgebraUtils`
    x0 : torch.tensor
        a starting vector.
        The dtype of `x0` determines whether real or complex arithmetic is used.
        The device of `x0` determines where the computation is performed.
    howmany : int, optional
        how many eigenvalues should be computed, by default 1
    which : str, optional
        which eigenvalues should be targeted, by default 'LM'
        - `LM`: eigenvalues of largest magnitude
        - `LR`: eigenvalues with largest (most positive) real part
        - `SR`: eigenvalues with smallest (most negative) real part
        - `LI`: eigenvalues with largest (most positive) imaginary part, only if x0 is complex
        - `SI`: eigenvalues with smallest (most negative) imaginary part, only if x0 is complex
    issymmetric : bool, optional
        if the linear map is symmetric, only meaningful if x0 is real, by default False
    ishermitian : bool, optional
        if the linear map is Hermitian, by default False
    **kwargs : optional
        The extra parameters defined in `KrylovDefault`.
        - `verbosity`: level of verbosity, by default 1
        - `tol`: tolerance for convergence, by default 1e-12
        - `krylovdim`: dimension of the Krylov space, by default 30
        - `maxiter`: maximum number of iterations, by default 100
        - `orth`: the orthogonalization method to be used, by default 'ModifiedGramSchmidt2'
        available: 'ModifiedGramSchmidt2', 'ModifiedGramSchmidt', to be implemented:
        'ClassicalGramSchmidt', 'ClassicalGramSchmidt2', 'ClassicalGramSchmidtIR', 'ModifiedGramSchmidtIR'
        - `eager`: whether to perform eager execution, by default False

    Returns
    -------
    values : np.ndarray
        the computed eigenvalues of length at least `howmany`, but could
        be longer if more eigenvalues were converged at the same cost.
    vectors : tc.tensor
        the corresponding eigenvectors, of the same length as `values`
    info : ConvergenceInfo
        has the following properties:
        - `info.converged`: int, how many eigenvalues and eigenvectors were actually
        converged to the specified tolerance `tol`
        - `info.residual`: list of tc.tensor, a list of the same length as `values` containing the
        residuals `info.residual[i] = f(vectors[i]) - values[i] * vectors[i]`
        - `info.normres`: list, list of the same length as `values` containing the
        norm of the residual `info.normres[i] = norm(info.residual[i])`
        - `info.numops`: number of times the linear map was applied
        - `info.numiter`: number of times the Krylov subspace was restarted

    Notes
    -----
    According to `issymmetric` and `ishermitian`, one specifies the algorithm explicitly
    as either `Lanczos` or `BlockLanczos`, for real symmetric or complex hermitian problems, or
    `Arnoldi`, for general problems. Note that these names refer to the process for
    building the Krylov subspace, but the actual algorithm is an implementation of the
    Krylov-Schur algorithm, which can dynamically shrink and grow the Krylov subspace, i.e. the
    restarts are so-called thick restarts where a part of the current Krylov subspace is kept.
    """
    alg = eigselector(
        x0, issymmetric=issymmetric, ishermitian=ishermitian,
        **kwargs
    )
    assert which in ['LM', 'LR', 'SR', 'LI', 'SI']
    if lau.isrealobj(x0):
        by, _ = EIGSORT[which]
        if by(1j) != by(-1j):
            raise ValueError(f"Eigenvalue selector which = {which} invalid because it does not treat"
            f"'λ' and 'conj(λ)' equally: work in complex arithmetic by providing a complex starting vector 'x0'")
    return alg.eigsolve(A, x0, howmany, which)


def eigselector(x0, issymmetric=False, ishermitian=False, **kwargs):
    # todo: lanczos, block lanczos
    if (lau.isrealobj(x0) and issymmetric) or ishermitian:
        raise NotImplementedError("Real symmetric or Hermitian matrix eigensolver not implemented yet.")
    else:
        from .arnoldi import Arnoldi
        return Arnoldi(**kwargs)


EIGSORT = {
    # "name": (sortfunction,  if_revert)
    "LM": (abs, True),
    "LR": (np.real, True),
    "SR": (np.real, False),
    "LI": (np.imag, True),
    "SI": (np.imag, False)
}

