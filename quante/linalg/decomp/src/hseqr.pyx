# cython: language_level=3
from numpy cimport PyArray_DATA
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_lapack as lapack  # 提供 zhseqr/chseqr
ctypedef np.complex128_t cplx128
ctypedef np.float64_t    real64

def zhseqr_(np.ndarray H_in, np.ndarray Z_in):
    cdef int n = <int>H_in.shape[0]
    if H_in.ndim != 2 or H_in.shape[0] != H_in.shape[1]:
        raise ValueError("H must be square")
    cdef char job = 'S'      # 求 Schur 形
    cdef char compz = 'V'          # 是否返回 Schur 向量
    cdef int ilo = 1, ihi = n
    cdef int ldh = H_in.strides[1] // H_in.itemsize
    cdef int ldz = Z_in.strides[1] // Z_in.itemsize
    cdef object dtype = H_in.dtype
    cdef np.ndarray w = np.empty(n, dtype=dtype, order="F")
    cdef np.ndarray work = np.empty(1, dtype=dtype, order="F")
    cdef int lwork = -1
    cdef int info

    cdef np.ndarray H
    cdef np.ndarray Z

    for i in range(2):
        lapack.zhseqr(
            &job, &compz, &n, &ilo, &ihi,
            <cplx128*>PyArray_DATA(H_in), &ldh, <cplx128*>PyArray_DATA(w),
            <cplx128*>PyArray_DATA(Z_in), &ldz, <cplx128*>PyArray_DATA(work), &lwork, &info,
        )
        if info != 0:
            raise RuntimeError(f"zhseqr(work query) failed, info={info}")
        if i == 0:
            lwork = <int>work[0].real
            work = np.empty(lwork, dtype=dtype)
    return H_in, Z_in, w

def dhseqr_(np.ndarray H_in, np.ndarray Z_in):
    H_in = np.array(H_in, dtype=np.float64, order='F', copy=True)
    Z_in = np.array(Z_in, dtype=np.float64, order='F', copy=True)
    cdef int n = <int>H_in.shape[0]
    if H_in.ndim != 2 or H_in.shape[0] != H_in.shape[1]:
        raise ValueError("H must be square")
    cdef char job = 'S'      # 求 Schur 形
    cdef char compz = 'V'          # 是否返回 Schur 向量
    cdef int ilo = 1, ihi = n
    cdef int ldh = n
    cdef int ldz = n
    cdef object dtype = H_in.dtype
    cdef np.ndarray WR = np.empty(n, dtype=np.float64, order="F")
    cdef np.ndarray WI = np.empty(n, dtype=np.float64, order="F")
    cdef np.ndarray work = np.empty(1, dtype=dtype, order="F")
    cdef int lwork = -1
    cdef int info

    cdef np.ndarray H
    cdef np.ndarray Z

    for i in range(2):
        lapack.dhseqr(
            &job, &compz, &n, &ilo, &ihi,
            <real64*>PyArray_DATA(H_in), &ldh, <real64*>PyArray_DATA(WR), <real64*>PyArray_DATA(WI),
            <real64*>PyArray_DATA(Z_in), &ldz, <real64*>PyArray_DATA(work), &lwork, &info,
        )
        if info != 0:
            raise RuntimeError(f"zhseqr(work query) failed, info={info}")
        if i == 0:
            lwork = <int>work[0].real
            work = np.empty(lwork, dtype=dtype)
    return H_in, Z_in, WR + 1j * WI

