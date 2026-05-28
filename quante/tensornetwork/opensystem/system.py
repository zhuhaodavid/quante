# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-27 22:00:44
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-28 02:10:22

from typing import Callable, List, Optional, Text, Tuple
from inspect import getfullargspec
from copy import copy
from functools import lru_cache
from ...linalg import matops as opr

import numpy as np
from numpy import ndarray
from scipy.linalg import expm

class System:
    def __init__(
            self,
            ham: ndarray,
            jump_ops: Optional[List[ndarray]] = None,
    ) -> None:
        """Create a System object. """
        self.ham = ham
        self.dim = self.ham.shape[0]
        self.master_eq = False if jump_ops is None else True
        self.jump_ops = [] if jump_ops is None else jump_ops
    
    @lru_cache(4)
    def liouvillian(self) -> ndarray:
        liouvillian = -1j * opr.commutator(self.ham)
        for op in self.jump_ops:
            op_dagger = op.conjugate().T
            liouvillian += (opr.left_right_super(op, op_dagger) \
                            - 0.5 * opr.acommutator(np.dot(op_dagger, op)))
        return liouvillian

    def get_propagators(self, dt):
        """Prepare propagator functions for the system. """
        first_step = expm(self.liouvillian()*dt/2.0)
        second_step = expm(self.liouvillian()*dt/2.0)
        return first_step, second_step

    def get_unitary_propagators(self, dt):
        """Prepare propagator functions for the system. """
        first_step = expm(-1j*self._hamiltonian*dt/2.0)
        second_step = expm(-1j*self._hamiltonian*dt/2.0)
        return first_step, second_step
