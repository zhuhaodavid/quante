# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 20:42:04
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 20:20:13

import numpy as _np
import math
from scipy import sparse as _sparse
from typing import Literal, overload

def _permutation_parity(perm):
    # Compute the parity (sign) of a permutation using a simple inversion count
    perm = _np.array(perm)
    parity = 1
    for i in range(len(perm)):
        for j in range(i+1, len(perm)):
            if perm[i] > perm[j]:
                parity *= -1
    return parity

def sky_anti_symmetrize(Jmat, hermitize=False):
    L = Jmat.shape[0]
    q2 = Jmat.ndim//2
    # hermitize
    if hermitize:
        for idx in _np.ndindex((L,)*q2):
            for jdx in _np.ndindex((L,)*q2):
                if idx == jdx:
                    # real diagonal elements
                    Jmat[idx + jdx] = _np.real(Jmat[idx + jdx])
                elif idx < jdx:
                    # complex off-diagonal elements
                    Jmat[idx + jdx] = _np.conj(Jmat[jdx + idx])
                    continue
    # anti-symmetrize
    for idx in _np.ndindex((L,)*q2):
        for jdx in _np.ndindex((L,)*q2):
            idx_sorted = tuple(sorted(idx))
            jdx_sorted = tuple(sorted(jdx))
            if idx == idx_sorted and jdx == jdx_sorted:
                continue
            Pi = _permutation_parity(idx)
            Pj = _permutation_parity(jdx)
            Jmat[idx + jdx] = Jmat[idx_sorted + jdx_sorted] * Pi * Pj
    return Jmat

@overload
def syk4_dirac(L:int, Nf:int, J:float|_np.ndarray=1., sparse:Literal[False]=False, basis=None) -> _np.ndarray:
    ...

@overload
def syk4_dirac(L:int, Nf:int, J:float|_np.ndarray=1., sparse:Literal[True]=True, basis=None) -> _sparse.csr_matrix:
    ...

def syk4_dirac(L:int, Nf:int, J:float|_np.ndarray=1., sparse=False, basis=None) -> _sparse.csr_matrix:
    r"""generate the SYK4 Dirac Hamiltonian matrix.

    The (complex) SYK4 Dirac Hamiltonian is defined as:
    .. math::
        H = 1/(2N)^(3/2) \sum_{ijkl} J_{ijkl} c_i^\dagger c_j^\dagger c_k c_l.
    
    for standard SYK4 model, we usually have diagonal coupling constants :math:`J_{ijkl}` 
    such that:
    .. math::
        E[(\Re J_{ijij})^2] = J^2,
        E[(\Im J_{ijij})^2] = 0, 
    
    and off-diagonal coupling constants such that:
    .. math::
        E[(\Re J_{ijkl})^2] = E[(\Im J_{ijkl})^2] = J^2 / 2,

    Parameters
    ----------
    L : int
        The number of Dirac fermion modes.
    J : ndarray or float, optional
        coupling constants for the SYK4 model,
        - If a float, it will be used as the coupling constant for all terms.
        - If a numpy array, it should be a 4D array of shape (L, L, L, L).
        The matrix will only use the anti-symmetrized part of the array, see
        following notes for details.
        defaults to None.
    Nf : int, optional
        The number of fermions, by default None.
    sparse : bool, optional
        If True, return a sparse matrix, otherwise return a dense matrix. Defaults to False.
    
    Notes
    -----
    When `J` is a numpy array, it should be a 4D array of shape (L, L, L, L).

    >>> Jmat = np.random.randn(L,L,L,L) + 1j * np.random.randn(L,L,L,L)
    >>> mat = qt.generate.matrix.syk4_dirac(L, L//2, J=Jmat, sparse=False)

    This will generate the same matrix as (but faster) following code:,

    >>> qt.generate.matrix.sky_anti_symetrize(Jmat, hermitize=False)
    >>> builder = qt.generate.operas.fermion.builder()
    >>> for i1, i2, j1, j2 in np.ndindex((L,)*4):
    >>>     builder += "++--", [i1, i2, j1, j2], Jmat[i1, i2, j1, j2]
    >>> basis = qt.generate.basis.fermion_basis(L=L, Nf=L//2)
    >>> mat = builder.build().to_matrix(basis)/(2*L)**(3/2)

    In this oper form, `Jmat` needs to be anti-symmetrized, so as
    to obtain the correct Hamiltonian matrix.
   
    Returns
    -------
    csr_matrix or ndarray
        The SYK4 Dirac Hamiltonian matrix.

    Raises
    ------
    ValueError
        If J is not a number or a numpy array.
    """
    if basis is None:
        from ....bridge.quspin_utils import fermion_basis
        basis = fermion_basis(L=L, Nf=Nf)
    else:
        assert basis._Np == Nf, "The number of fermions in the basis must match the input Nf."
    if isinstance(J, _np.ndarray):
        Jmat = J.reshape(L, L, L, L)
    elif isinstance(J, (int, float)):
        Jmat = (J/_np.sqrt(2)) * (_np.random.randn(L**2,L**2) + 1j * _np.random.randn(L**2,L**2))
        _np.fill_diagonal(Jmat, _np.random.randn(L**2)*J)
        Jmat = Jmat.reshape(L, L, L, L)
        sky_anti_symmetrize(Jmat, hermitize=True)
    else:
        raise ValueError("J must be a number or a numpy array.")

    nnz_eachcol = 1 + Nf*(L-Nf) + math.comb(Nf, 2) * math.comb(L-Nf, 2)

    Ns = basis.Ns
    from .nbfuc.syk_nb import make_syk_matrix
    col, mat = make_syk_matrix(L, Nf, Jmat, basis.states, Ns, nnz_eachcol)
    
    # make the sparse matrix
    indptr = _np.arange(0, mat.size+1, col.shape[0])
    indices = col.T.flatten()
    data = mat.T.flatten() / ((2 * L)**(3/2))
    mat = _sparse.csc_matrix((data, indices, indptr), shape=(Ns, Ns))

    return mat.tocsr() if sparse else mat.toarray()

    
def syk4_majorana(L:int, J:_np.ndarray, sparse=False):
    """generate the SYK4 Majorana Hamiltonian matrix.

    This function is from the QuSpin package.

    Parameters
    ----------
    L : int
        The number of Majorana fermion modes.
    J : _np.ndarray
        The coupling constants for the SYK4 model, should be a 4D array of shape (L, L, L, L).
        If a float, it will be used as the coupling constant for all terms.
    sparse : bool, optional
        If True, return a sparse matrix, otherwise return a dense matrix. Defaults to False.

    Returns
    -------
    csr_matrix or ndarray
        The SYK4 Majorana Hamiltonian matrix.
    """
    op_list = [
        (
            "xxxx",
            (i, j, k, l),
            J[i, j, k, l],
        )
        for i in range(L)
        for j in range(i+1, L)
        for k in range(j+1, L)
        for l in range(k+1, L)
    ]
    from ....bridge.quspin_utils.quspin_extension_wrap.basis.basis_general.fermion import spinless_fermion_basis_general
    basis = spinless_fermion_basis_general(L)
    mat = basis._make_matrix(op_list, dtype=J.dtype)/4

    return mat.tocsr() if sparse else mat.toarray()
 