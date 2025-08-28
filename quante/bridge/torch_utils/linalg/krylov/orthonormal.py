# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:27:31
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-28 22:12:55

import torch as tc
import numpy as np

class OrthonormalBasis:
    def __init__(self, basis:list, num:int):
        self.data = basis
        self.num = num
    
    @property
    def basis(self):
        return self.data[:self.num]
    
    def basistransform_(self, U):
        """
        Transform the orthonormal basis `b` by the matrix `U`. For `b` an orthonormal basis,
        the matrix `U` should be real orthogonal or complex unitary; it is up to the user to ensure
        this condition is satisfied. The new basis vectors are given by

        ```
            b[j] ← b[i] * U[i,j]
        ```
        b[k,j] ← b[k,i] * U[i,j]

        and are stored in `b`, so the old basis vectors are thrown away. Note that, by definition,
        the subspace spanned by these basis vectors is exactly the same.

        Parameters
        ----------
        U : np.ndarray
            The transformation matrix.
        """
        b = self.data
        m, n = U.shape
        if m != len(b):
            raise ValueError(f"Dimension mismatch: {m} != {len(b)}")
        
        # # todo: optimize
        # K = len(b[0])
        # new_b = [tc.zeros_like(b[0], dtype=tc.complex128) for _ in range(n)]
        # for i in range(m):
        #     for j in range(n):
        #         new_b[j].add_(b[i], alpha=U[i,j])
        # self.basis[:n] = [res1[:, i] for i in range(n)]

        self.data[:n] = tc.tensor(U.T) @ self.data[:m]
        