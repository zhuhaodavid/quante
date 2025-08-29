# cython: language_level=3
from numpy cimport PyArray_DATA
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as lapack
ctypedef np.float64_t real64
ctypedef np.complex128_t cplx128

def dtrevc(np.ndarray T):
    cdef int n = <int>T.shape[0]
    cdef int ldt = n
    cdef int ldvl = n
    cdef int ldvr = n
    cdef int mm = n
    cdef int m
    cdef char side = 'R'
    cdef char howmny = 'A'
    cdef np.ndarray VL = np.zeros((n, n), dtype=np.float64, order='F')
    cdef np.ndarray VR = np.zeros((n, n), dtype=np.float64, order='F')
    cdef np.ndarray work = np.empty(3*n, dtype=np.float64)
    cdef int info

    lapack.dtrevc(
        &side, &howmny, NULL, &n,
        <real64*>PyArray_DATA(T), &ldt,
        <real64*>PyArray_DATA(VL), &ldvl,
        <real64*>PyArray_DATA(VR), &ldvr,
        &mm, &m,
        <real64*>PyArray_DATA(work),
        &info
    )
    return VR

def ztrevc(np.ndarray T):
    cdef int n = <int>T.shape[0]
    cdef int ldt = n
    cdef int ldvl = n
    cdef int ldvr = n
    cdef int mm = n
    cdef int m
    cdef char side = 'R'
    cdef char howmny = 'A'
    cdef np.ndarray VL = np.zeros((n, n), dtype=np.complex128, order='F')
    cdef np.ndarray VR = np.zeros((n, n), dtype=np.complex128, order='F')
    cdef np.ndarray work = np.empty(2*n, dtype=np.complex128)
    cdef np.ndarray rwork = np.empty(n, dtype=np.float64)
    cdef int info

    lapack.ztrevc(
        &side, &howmny, NULL, &n,
        <cplx128*>PyArray_DATA(T), &ldt,
        <cplx128*>PyArray_DATA(VL), &ldvl,
        <cplx128*>PyArray_DATA(VR), &ldvr,
        &mm, &m,
        <cplx128*>PyArray_DATA(work),
        <real64*>PyArray_DATA(rwork),
        &info
    )
    return VR