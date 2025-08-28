# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:41:17
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-28 20:39:15

import torch as tc
import numpy as np

class Householder:
    def __init__(self, beta, v, r):
        self.beta = beta
        self.v = v
        self.r = r
    
    def lmul_(self, A, j, cols=None):
        if cols is None:
            cols = range(A.shape[1])
        v = self.v
        r = self.r
        beta = self.beta
        if np.isclose(beta, 0.):
            return A
        for k in cols:
            mu = np.vdot(v, A[r,k]) * beta
            A[r,k] -= mu * v
        return A
    
    def rmul_(self, A, rows=None):
        if rows is None:
            rows = range(A.shape[0])
        beta, v, r = np.conj(self.beta), self.v, self.r
        if np.isclose(beta, 0.):
            return A
        w = np.zeros(len(rows), dtype=A.dtype)

        l = 0
        for k in r:
            w += A[rows, k] * v[l]
            l += 1
        l = 0
        for k in r:
            vl = beta * np.conj(v[l])
            A[rows, k] -= w * vl
            l += 1
        return A 

def householder(A, row, r, k):
    # np.argwhere() 
    i = np.flatnonzero(r == k)
    if len(i) == 0:
        raise ValueError(f"k = {k} should be in the range r = {r}")
    beta, v, nu = _householder_(np.conjugate(A[row, r]), i[0])
    return Householder(beta, v, r), nu

def _householder_(v, i):
    beta = 0.
    sigma = np.linalg.norm(v)**2 - np.abs(v[i])**2
    # vi = tc.clone(v[i])
    vi = v[i]
    nu = np.sqrt(np.abs(vi)**2 + sigma)
    if sigma == 0 and vi == nu:
        beta = 0.
    else:
        if np.real(vi) < 0:
            vi -= nu
        else:
            vi = ((vi - np.conj(vi)) * nu - sigma) / (np.conj(vi) + nu)
        
        v[:] /= vi
        v[i] = 1.
        beta = - np.conj(vi) / nu
    return beta, v, nu


# def checksquare(A):
#     m, n = A.shape
#     if m != n:
#         raise ValueError(f"matrix should be square, got {m}x{n}")
#     return m
    
# def schur2eigvals(T, which):
#     n = checksquare(T)
#     if len(set(which)) != len(which):
#         raise ValueError("indices in `which` should be unique")
#     return [T[i,i] for i in which]

# def schur2eigvecs(T):
#     n = checksquare(T)
#     VR = np.empty_like(T)
#     VL = np.empty((n,0), dtype=T.dtype)
#     trevc, = get_lapack_funcs(('trevc',), (T,))
#     exit()

