# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 20:55:05
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-12 10:08:29

import numpy as np
from .....basicfun.utils_numba import njit, numba_cache_dir, config, prange

config.CACHE_DIR = numba_cache_dir
@njit
def _permucoef(bstate, Li, Lj, L):
    Lmin, Lmax = min(Li, Lj), max(Li, Lj)
    res = 1
    for i in range(Lmin+1, Lmax):
        mask = 1 << (L - 1 - i)
        if bstate & mask:
            res *= (-1)
    return res

config.CACHE_DIR = numba_cache_dir
@njit
def findstate(s_list:np.ndarray, sb:int) -> int:
    # for i in range(len(basis.states)):
    #     assert i == findstate(basis.states, basis.states[i]), f"State {i} not found correctly."   
    M = len(s_list)
    b_min = 0
    b_max = M-1
    for _ in range(int(np.log2(M)) + 1):
        b = b_min + (b_max - b_min)//2
        if sb > s_list[b]:
            b_max = b
        elif sb < s_list[b]:
            b_min = b + 1
        else:
            return b
    return -1

config.CACHE_DIR = numba_cache_dir
@njit(parallel=True)
def make_syk_matrix(L, Nup, Jmat, bstates, Ns, nnz_eachcol):
    res = np.zeros((nnz_eachcol, Ns), dtype=Jmat.dtype)
    col = np.zeros((nnz_eachcol, Ns), dtype=bstates.dtype)
    for i in prange(Ns):
        bstate = bstates[i]
        
        # diagonal term
        # c+_j c+_i c_j c_i
        for Li in range(L):
            mask_i = 1 << (L - 1 - Li)
            if bstate & mask_i:  # should be 1
                for Lj in range(Li): 
                    # only need to consider Lj < Li
                    mask_i = 1 << (L - 1 - Lj)
                    if bstate & mask_i:
                        # should be 1
                        res[0, i] += - Jmat[Lj,Li,Lj,Li] * 4
                        col[0, i] = i

        # Find positions of 1s and 0s in bstate (occupied and unoccupied sites)
        pos_1 = np.empty(Nup, dtype=np.int64)
        pos_0 = np.empty(L-Nup, dtype=np.int64)
        idx1 = 0
        idx0 = 0
        mask = 1 << (L - 1)
        for Li in range(L):
            if bstate & mask:
                pos_1[idx1] = Li
                idx1 += 1
            else:
                pos_0[idx0] = Li
                idx0 += 1
            mask >>= 1
                
        # Hamming distance 2
        # c+_i c+_k c_j c_k
        indx = 1
        for Lj in pos_1:
            for Li in pos_0:
                mask_j = 1 << (L - 1 - Lj)
                mask_i = 1 << (L - 1 - Li)
                for Lk in range(L):
                    if Lk == Li or Lk == Lj:
                        # repeat is not allowed
                        continue
                    mask_k = 1 << (L - 1 - Lk)
                    if bstate & mask_k:  
                        # c_k works
                        pc1 = _permucoef(bstate, Li, Lj, L)
                        pc2 = _permucoef(bstate, Lj, Li, L)
                        if Li < Lk:
                            if Lj < Lk:
                                res[indx, i] += Jmat[Li, Lk, Lj, Lk] * (-4) * pc1
                            else:
                                res[indx, i] += Jmat[Li, Lk, Lk, Lj] * 4 * pc2
                        else:
                            if Lj < Lk:
                                res[indx, i] += Jmat[Lk, Li, Lj, Lk] * 4 * pc2
                            else:
                                res[indx, i] += Jmat[Lk, Li, Lk, Lj] * (-4) * pc2
                        col[indx, i] = findstate(bstates, bstate ^ mask_i ^ mask_j)
                indx += 1
        
        # Hamming distance 4
        # c+_l c+_k c_j c_i
        for indx1 in range(Nup):
            Li = pos_1[indx1]
            for indx2 in range(indx1):
                Lj = pos_1[indx2]
                for indx3 in range(L-Nup):
                    Lk = pos_0[indx3]
                    if Lk == Li or Lk == Lj:
                        # repeat is not allowed
                        continue
                    for indx4 in range(indx3):
                        Ll = pos_0[indx4]
                        if Ll == Li or Ll == Lj:
                            # repeat is not allowed
                            continue
                        # c+_l c+_k c_j c_i
                        pc = _permucoef(bstate, Li, Lk, L)
                        tmpstate = bstate ^ (1 << (L - 1 - Li)) ^ (1 << (L - 1 - Lk))
                        pc *= _permucoef(tmpstate, Lj, Ll, L)
                        tmpstate ^= (1 << (L - 1 - Lj)) ^ (1 << (L - 1 - Ll))
                        res[indx, i] += Jmat[Ll,Lk,Lj,Li] * (-4) * pc
                        col[indx, i] = findstate(bstates, tmpstate)
                        indx += 1
    return col, res

