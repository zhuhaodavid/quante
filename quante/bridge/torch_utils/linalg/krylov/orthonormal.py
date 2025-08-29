# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-08-28 16:27:31
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-30 02:31:32

import torch as tc
# from julia import Main

# Main.eval("using KrylovKit: OrthonormalBasis")
# fm = 0

class OrthonormalBasis:
    def __init__(self, basis, maxdim):
        self.data = tc.zeros((maxdim, len(basis)), dtype=basis.dtype)
        self.data[0] = basis
        self.num = 1
        # elif fm == 1:
        #     self.data = [basis]
        # elif fm == 2:
        #     Main.basis = basis.numpy()
        #     Main.eval("b = OrthonormalBasis(basis)")
    
    def pop(self):
        self.num -= 1
        return self.data[self.num]
        # elif fm == 1:
        #     return self.data.pop()
        # elif fm == 2:
        #     state = Main.eval("pop!(b)")
        #     return tc.tensor(state)
    
    def append(self, v):
        self.data[self.num] = v
        self.num += 1
        # elif fm == 1:
        #     self.data.append(v)
        # else:
        #     Main.v = v.numpy()
        #     Main.eval("push!(b, v)")
    
    def __len__(self):
        return self.num
        # if fm == 0:
        # elif fm == 1:
        #     return len(self.data)
        # elif fm == 2:
        #     return Main.eval("length(b)")

    @property
    def basis(self):
        return self.data[:self.num]
        # if fm == 0:
        #     return self.data[:self.num]
        # elif fm == 1:
        #     return self.data
        # elif fm == 2:
        #     raise Exception()
            
    
    def set(self, keep, r):
        self.data[keep][:] = r
        # if fm == 0:
        #     self.data[keep][:] = r
        # elif fm == 1:
        #     self.data[keep] = r
        # elif fm == 2:
        #     raise Exception()
            

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
        if m != len(self):
            raise ValueError(f"Dimension mismatch: {m} != {len(b)}")

        # if fm == 0:
            # self.data[:n] = tc.tensor(U.T) @ self.data[:m]
            # todo: optimize
            # tmp = U.T
            # new_b = tc.zeros((n, b.shape[1]), dtype=b[0].dtype)
            # for k in range(b.shape[1]):
            #     for i in range(m):
            #         for j in range(n):
            #             new_b[j, k] += tmp[j,i] * b[i, k]
        new_b = tc.tensor(U.T, dtype=self.data.dtype) @ self.data[:m, :]
        self.data[:n, :] = new_b

        # elif fm == 1:
        #     # # # todo: optimize
        #     new_b = [tc.zeros_like(b[0], dtype=b[0].dtype) for _ in range(n)]
        #     for i in range(m):
        #         for j in range(n):
        #             new_b[j].add_(b[i], alpha=U[i,j])
        #     self.basis[:n] = new_b



