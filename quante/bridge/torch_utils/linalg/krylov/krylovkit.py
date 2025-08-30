# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:46:01
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 14:37:49

import torch as tc
import numpy as np
import warnings

class KrylovDefault:
    def __init__(self):
        self.orth = "ModifiedGramSchmidt2"
        self.krylovdim = 30
        self.maxiter = 100
        self.blockkrylovdim = 100
        self.tol = 1.e-12
        self.verbosity = 1

    def update_params(self, kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown parameter: {key}", UserWarning)

class ConvergenceInfo:
    def __init__(self, converged, residual, normres, numiter, numops):
        self.converged = converged
        self.residual = residual
        self.normres = np.array(normres)
        self.numiter = numiter
        self.numops = numops
    
    def __repr__(self):
        msg = "ConvergenceInfo: "
        if self.converged == 0:
            msg += "no converged values"
        elif self.converged == 1:
            msg += "one converged value"
        else:
            msg += f"{self.converged} converged values"

        msg += f" after {self.numiter} iterations and {self.numops} applications of the linear map;\n"
        msg += f" norms of residuals are given by {self.normres}."
        return msg


class LinearAlgebraUtils:
    device = 'cpu'

    @staticmethod
    def update_device(x0):
        LinearAlgebraUtils.device = x0.device
    
    @staticmethod
    def apply(A, x):
        return (A @ x).to(device=LinearAlgebraUtils.device)

    @staticmethod
    def norm(x):
        return tc.linalg.norm(x).item()

    @staticmethod
    def inner(x, y):
        return tc.vdot(x, y).item()
    
    @staticmethod
    def zeros_like(x):
        return tc.zeros_like(x)
    
    @staticmethod
    def zeros(shape, dtype=None):
        return tc.zeros(shape, dtype=dtype, 
                        device=LinearAlgebraUtils.device)

    @staticmethod
    def add_(x, y, alpha=None):
        if alpha is None:
            x.add_(y)
        else:
            x.add_(y, alpha=alpha)
        return x

    @staticmethod
    def sub_(x, y, alpha=None):
        if alpha is None:
            x.sub_(y)
        else:
            x.sub_(y, alpha=alpha)
        return x

    @staticmethod
    def div_(x, alpha):
        x.div_(alpha)
        return x

    @staticmethod
    def matmul(A, B):
        """
        假设 A 是 numpy array, B 是 torch tensor
        最终要得到 torch tensor
        """
        if np.iscomplexobj(A) and not B.is_complex():
            return tc.tensor(A, dtype=tc.complex128, device=B.device) @ B.to(dtype=tc.complex128)
        else:
            return tc.tensor(A, dtype=B.dtype, device=B.device) @ B
    
    @staticmethod
    def isrealobj(x):
        return not x.is_complex()
    