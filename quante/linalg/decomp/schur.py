# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:37:49
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 13:03:16

from scipy.linalg import get_lapack_funcs, schur
import numpy as np

def is_fortran_format(x):
    return x.strides[0] < x.strides[1]

try:
    import pyximport
    pyximport.install(
        language_level=3,
        setup_args={
            "include_dirs": [np.get_include()],
        }
    )
    has_cython = True
except ImportError:
    has_cython = False

hseqr_available = False
if has_cython:
    try:
        from .src.hseqr import dhseqr_, zhseqr_
        hseqr_available = True
    except ImportError:
        hseqr_available = False

# hseqr_available = False
if hseqr_available:
    def hschur_(H, Z):
        if is_fortran_format(H) and is_fortran_format(Z):
            if np.isrealobj(H):
                return dhseqr_(H, Z)
            else:
                return zhseqr_(H, Z)
        else:
            raise ValueError("Input arrays must be Fortran-contiguous")
else:
    def hschur_(H, Z):
        H[:], Z[:] = schur(H, output='real')
        values = schur2eigvals(H)
        return H, Z, values


def schur2eigvals(T):
    if not np.isrealobj(T):
        return np.diag(T)
    else:
        n = T.shape[0]
        D = np.zeros(n, dtype=np.complex128)
        for i in range(n):
            if i < n-1 and not np.isclose(T[i+1,i], 0.):
                halftr = (T[i, i] + T[i + 1, i + 1]) / 2
                diff = (T[i, i] - T[i + 1, i + 1]) / 2
                d = diff * diff + T[i, i + 1] * T[i + 1, i]  # = halftr*halftr - det
                D[i] = halftr + 1j * np.sqrt(-d)
            elif i > 0 and not np.isclose(T[i,i-1], 0.):
                halftr = (T[i, i] + T[i - 1, i - 1]) / 2
                diff = -(T[i, i] - T[i - 1, i - 1]) / 2
                d = diff * diff + T[i, i - 1] * T[i - 1, i]  # = halftr*halftr - det
                D[i] = halftr - 1j * np.sqrt(-d)
            else:
                D[i] = T[i,i]
        return D
        
trevc_available = False
if has_cython:
    try:
        from .src.trevc import dtrevc, ztrevc
        trevc_available = True
    except:
        trevc_available = False

if trevc_available:
    def schur2eigvecs(T):
        T = np.asfortranarray(T)
        if not np.isrealobj(T):
            VR = ztrevc(T)
            return _normalizevecs_(VR)
        else:
            n = T.shape[0]
            VRp = dtrevc(T)
            VR = np.empty((n, n), dtype=np.complex128)
            i = 0
            while i < n:
                if i == n-1 or np.isclose(T[i+1,i], 0.):
                    for k in range(n):
                        VR[k, i] = VRp[k, i]
                    i += 1
                else:
                    for k in range(n):
                        VR[k, i] = VRp[k, i] + 1j * VRp[k, i + 1]
                        VR[k, i + 1] = VRp[k, i] - 1j * VRp[k, i + 1]
                    i += 2
            return _normalizevecs_(VR)
else:
    def schur2eigvecs(T):
        return np.linalg.eig(T)[1]


def _normalizevecs_(VR):
    norms = np.linalg.norm(VR, axis=0)
    return VR / norms

def permuteschur_(T_in, Q_in, order):
    """
    Reorder complex Schur decomposition (T, Q) according to `order`.
    Using LAPACK trexc directly.
    """
    T = np.asfortranarray(T_in)
    Q = np.asfortranarray(Q_in)
    trexc, = get_lapack_funcs(('trexc',), (T_in,))
    p = order + 1
    n = T.shape[0]
    i = 0
    while i < len(order):
        ifirst = int(p[i])
        ilast = i + 1
        if ifirst == n or np.isclose(T[ifirst, ifirst - 1], 0.):
            T, Q, info = trexc(T, Q, ifirst, ilast, overwrite_a=True, overwrite_q=True)
            if info != 0:
                raise RuntimeError(f"trexc failed with info={info}")
            for k in range(i+1, len(p)):
                if p[k] < p[i]:
                    p[k] += 1
            i += 1
        else:
            if p[i+1] != ifirst + 1:
                raise ValueError("cannot split 2x2 blocks when permuting schur decomposition")
            T, Q, info = trexc(T, Q, ifirst, ilast, overwrite_a=True, overwrite_q=True)
            if info != 0:
                raise RuntimeError(f"trexc failed with info={info}")
            for k in range(i+2, len(p)):
                if p[k] < p[i]:
                    p[k] += 2
            i += 2
    T_in[:], Q_in[:] = T, Q
    return T_in, Q_in
