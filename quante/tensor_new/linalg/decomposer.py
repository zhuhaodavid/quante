# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2024-12-11 11:26:19
# @Last Modified by:   dzwang
# @Last Modified time: 2024-12-13 10:16:05
import numpy as np
from ...linalg.svd_robust import TruncationError


__all__ = ["truncate", "qr", "rq", "svd"]


def truncate(S, chi_max=None, svd_min=None, trunc_cut=None):
    pass


def qr(tsr:np.ndarray, *, lr_indx=None) -> tuple[np.ndarray, np.ndarray]:
    pass


def rq(tsr:np.ndarray, *, lr_indx=None) -> tuple[np.ndarray, np.ndarray]:
    pass


def svd(tsr, *, lr_indx=None, trunc_para=(None, None, None), full_matrices:bool = False):
    pass
