# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:46:01
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 22:18:40

import numpy as np
import warnings

EIGSORT = {
    # "name": (sortfunction,  if_revert)
    "LM": (abs, True),
    "LR": (np.real, True),
    "SR": (np.real, False),
    "LI": (np.imag, True),
    "SI": (np.imag, False)
}

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

class NpLinearAlgebraUtils:

    @staticmethod
    def update_device(x0):
        pass

    @staticmethod
    def apply(A, x):
        return A @ x

    @staticmethod
    def norm(x):
        return np.linalg.norm(x)

    @staticmethod
    def inner(x, y):
        return np.vdot(x, y)

    @staticmethod
    def zeros_like(x):
        return np.zeros_like(x)

    @staticmethod
    def zeros(shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def add_(x, y, alpha=None):
        if alpha is None:
            x[:] += y
        else:
            x[:] += y * alpha
        return x

    @staticmethod
    def sub_(x, y, alpha=None):
        if alpha is None:
            x[:] -= y
        else:
            x[:] -= y * alpha
        return x

    @staticmethod
    def div_(x, alpha):
        x[:] /= alpha
        return x

    @staticmethod
    def mul_(x, alpha):
        x[:] *= alpha
        return x

    @staticmethod
    def matmul(A, B):
        return A @ B

    @staticmethod
    def isrealobj(x):
        return np.isrealobj(x)

class LinearAlgebraUtils:
    @staticmethod
    def isrealobj(x):
        if isinstance(x, np.ndarray):
            return np.isrealobj(x)
        else:
            try:
                return not x.is_complex()
            except:
                raise ValueError(f"type {type(x)} not supported")

def overload_methods(x0):
    lau = LinearAlgebraUtils
    if isinstance(x0, np.ndarray):
        newlau = NpLinearAlgebraUtils
    else:
        import torch as tc
        if isinstance(x0, tc.Tensor):
            from ...bridge.torch_utils.linalg.krylov import TcLinearAlgebraUtils
            newlau = TcLinearAlgebraUtils
        else:
            raise ValueError(f"Unsupported tensor type: {type(x0)}")
    for attr in dir(newlau):
        if attr.startswith("__"):
            continue
        func = getattr(newlau, attr)
        if callable(func):
            setattr(lau, attr, staticmethod(func))