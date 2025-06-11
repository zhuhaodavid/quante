# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:44:12
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-11 22:45:30

import numpy as np
from ...basicfun.utils_numba import njit, pnjit, prange, vectorize, numba_cache_dir, config

config.CACHE_DIR = numba_cache_dir
@pnjit
def _spectral_form_factor_single(engs, t):
    """
    Calculate the spectral form factor of a matrix ensemble.

    Parameters
    ----------
    engs : ndarray
        The spectrum of the matrix ensemble.
    times : ndarray
        The time points to evaluate the spectral form factor.

    Returns
    -------
    sff : ndarray
        The spectral form factor at the given time points.
    """
    # return np.mean(np.abs(np.sum(np.exp(1j*engs*t), axis=1))**2)
    sff = 0
    for i in prange(engs.shape[0]):
        sff += np.abs(np.sum(np.exp(1j*engs[i]*t)))**2
    return sff/engs.shape[0]
 
config.CACHE_DIR = numba_cache_dir
@pnjit
def _spectral_form_factor(engs, ts):
    """
    Calculate the spectral form factor of a matrix ensemble.

    Parameters
    ----------
    engs : ndarray
        The spectrum of the matrix ensemble.
    times : ndarray
        The time points to evaluate the spectral form factor.

    Returns
    -------
    sff : ndarray
        The spectral form factor at the given time points.
    """
    iternum, dim = engs.shape
    mat = np.empty((iternum, dim), dtype=float)
    for i in prange(iternum):
        mat[i,:] = engs[i]
    sff = np.empty_like(ts)
    l = len(ts)
    for j in prange(l):
        sff[j] = np.mean(np.abs(np.sum(np.exp(1j*mat*ts[j]), axis=1))**2)
    return sff
 

