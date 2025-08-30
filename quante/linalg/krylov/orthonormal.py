# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:27:31
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 22:09:41

from .krylovkit import LinearAlgebraUtils as lau

class OrthonormalBasis:
    def __init__(self, basis, maxdim):
        self.data = lau.zeros((maxdim, len(basis)), dtype=basis.dtype)
        self.data[0] = basis
        self.num = 1
   
    def pop(self):
        self.num -= 1
        return self.data[self.num]
    
    def popfirst(self):
        self.num -= 1
        self.data[:self.num] = self.data[1:self.num+1]
   
    def append(self, v):
        self.data[self.num] = v
        self.num += 1
   
    def __len__(self):
        return self.num

    @property
    def basis(self):
        return self.data[:self.num]
   
    def set(self, keep, r):
        self.data[keep][:] = r

    def basistransform_(self, U):
        m, n = U.shape
        self.data[:n, :] = lau.matmul(U.T, self.data[:m, :])

    def basistransform(self, U):
        m, n = U.shape
        return lau.matmul(U.T, self.data[:m, :])
