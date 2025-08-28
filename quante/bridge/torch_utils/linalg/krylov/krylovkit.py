# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:46:01
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-28 16:48:39

import numpy as np

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
