# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:41:17
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 22:09:04

import numpy as np

class Householder:
    def __init__(self, beta, v, r):
        self.beta = beta
        self.v = v
        self.r = r
    
    def lmul_(self, A, cols=None):
        if self.beta == 0.:
            return A
        if cols is None:
            cols = range(A.shape[1])
        
        for k in cols:
            mu = 0
            i = 0
            for j in self.r:
                mu += np.conj(self.v[i]) * A[j, k]
                i += 1
            mu *= self.beta
            i = 0
            for j in self.r:
                A[j, k] -= mu * self.v[i]
                i += 1
        return A

    
    def rmul_(self, A, rows=None):
        beta = np.conj(self.beta)
        if beta == 0.:
            return A
        if rows is None:
            rows = range(A.shape[0])
        
        w = np.zeros(len(rows), dtype=A.dtype)
        l = 0
        for k in self.r:
            j = 0
            vl = self.v[l]
            for i in rows:
                w[j] += A[i, k] * vl
                j += 1
            l += 1
        l = 0
        for k in self.r:
            j = 0
            vl = beta * np.conj(self.v[l])
            for i in rows:
                A[i, k] -= w[j] * vl
                j += 1
            l += 1
        return A
        
def householder(A, row, r, k):
    i = np.flatnonzero(r == k)
    if len(i) == 0:
        raise ValueError()
    A[row, r] = np.conj(A[row, r])
    beta, v, nu = _householder_(A[row, r], i[0])
    return Householder(beta, v, r), nu

def _householder_(v, i):
    beta = 0.
    sigma = 0.
    for k in range(i):
        sigma += abs(v[k]) ** 2
    for k in range(i+1, len(v)):
        sigma += abs(v[k]) ** 2
    vi = v[i]
    nu = np.sqrt(np.abs(vi)**2 + sigma)
    
    if sigma == 0 and vi == nu:
        beta = 0.
    else:
        if np.real(vi) < 0:
            vi = vi - nu
        else:
            vi = ((vi - np.conj(vi)) * nu - sigma) / (np.conj(vi) + nu)
        for k in range(i):
            v[k] /= vi
        v[i] = 1.
        for k in range(i+1, len(v)):
            v[k] /= vi
        beta = - np.conj(vi) / nu
    return beta, v, nu
