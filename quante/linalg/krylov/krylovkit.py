# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:46:01
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-02 15:16:36

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
    def __init__(self, converged, normres, numiter, numops):
        self.converged = converged
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
    def __init__(self, x0, lau):
        if lau is None:
            if isinstance(x0, np.ndarray):
                from ..evolve.nbfuc.expm_mul_nb import NpLinearAlgebraUtils
                self.lau = NpLinearAlgebraUtils
            else:
                from ...bridge.torch_utils.linalg.krylov import TcLinearAlgebraUtils
                self.lau = TcLinearAlgebraUtils

