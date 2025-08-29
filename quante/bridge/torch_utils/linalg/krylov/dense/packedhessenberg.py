# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:34:18
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 02:29:37

import numpy as np
# from julia import Main

class PackedHessenberg:
    def __init__(self, data, n):
        self.data = data
        self.n = n
        assert len(data) >= ((n * n + 3 * n - 2) >> 1)
    
    @property
    def shape(self):
        return self.n, self.n
    
    def __len__(self):
        return self.n*self.n
    
    def isempty(self):
        return self.n == 0
    
    def copyto_(self, dest):
        if self.isempty():
            return dest
        
        for i in range(self.n):
            for j in range(self.n):
                if i > j + 1:
                    dest[i,j] = 0.
                else:
                    dest[i,j] = self.data[((j * j + 3 * j - 1) >> 1) + i + 1]
        
        return dest
    
    def copy_from(self, src):
        for i in range(self.n):
            for j in range(self.n):
                if i > j + 1:
                    assert np.isclose(src[i,j], 0.)
                else:
                    self.data[((j * j + 3 * j - 1) >> 1) + i + 1] = src[i,j]

