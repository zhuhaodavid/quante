# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:19:37
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-31 18:21:50

import numpy as np
from typing import Literal

from .lanczos import Lanczos
from .arnoldi import Arnoldi
from ..krylovkit import EIGSORT
from ...matops import isherm as fuc_isherm

def eigsolve(
    A,
    x0,
    howmany=1,
    which:Literal[
        'LM', 'LR', 'SR', 'LI', 'SI'
    ] ='LM',
    isherm=None,
    lau=None,
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
    isherm : bool, optional
        if the linear map is Hermitian, by default False
        For large matrix, this keyword should be input for best performence (!!!)
    lau : class, optional
        linear algera utils, a class that contain method as the one
        "NpLinearAlgebraUtils" in "../krylovkit.py".
        by default None, i.e., interally determined by type of x0
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
        norm of the residual `info.normres[i] = norm(f(vectors[i]) - values[i] * vectors[i])`
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
    if isherm is None:
        try:
            isherm = fuc_isherm(A)
        except:
            raise Exception("Failed to determine if matrix is Hermitian")

    if x0.ndim == 2 and x0.shape[1] == 1:
        x0 = x0.squeeze()
    if isherm:
        assert which in ['LM', 'LR', 'SR']
        print("running Lanczos ...")
        return Lanczos(**kwargs).eigsolve(A, x0, howmany, which, lau)
    else:
        assert which in ['LM', 'LR', 'SR', 'LI', 'SI']
        if isrealobj(x0):
            by, _ = EIGSORT[which]
            if by(1j) != by(-1j):
                raise ValueError(f"Eigenvalue selector which = {which} invalid "
                f"because it does not treat 'λ' and 'conj(λ)' equally: work in "
                f"complex arithmetic by providing a complex starting vector 'x0'")
        print("running Arnoldi ...")
        return Arnoldi(**kwargs).eigsolve(A, x0, howmany, which, lau)

def isrealobj(x0):
    if isinstance(x0 ,np.ndarray):
        return np.isrealobj(x0)
    else:
        return not x0.is_complex()