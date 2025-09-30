# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-28 13:00:49
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-28 14:25:15


import numpy as np
from .....basicfun.utils_numba import njit, config, numba_cache_dir
from ..bitsoperation import cyclebits, reflectbits, next_combination, findstate, invertbits


#############################################
# Nup
#############################################
config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8[:]))(i8,i8)")
def construct_Nup_basis(L: int, Nup: int) -> tuple[int, np.ndarray]:
    if L == Nup:
        return 1, np.array([0], dtype=np.int64)
    
    s_list = []
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << (L - Nup)) - 1

    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        s_list.append(s)
        s = next_combination(s)
    
    s_list = np.array(s_list, dtype=np.int64)
    
    return len(s_list), s_list

config.CACHE_DIR = numba_cache_dir
@njit
def convert_project_Nup_to_full_space(proj:np.ndarray, L:int, s_list:np.ndarray) -> np.ndarray:
    M, N = proj.shape
    res = np.zeros((1<<L, N), dtype=proj.dtype)
    for i in range(M):
        for j in range(N):
            mele = proj[i, j]
            # if np.isclose(mele, 0):
            #     continue
            res[s_list[i], j] = mele
    return res

###############################################
# pblock
###############################################

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8)")
def construct_pblock_basis(L:int, p:int) -> tuple[int, np.ndarray]:
    """
    - p: -1, 1
    """
    s_list = []
    for s in range(1<<L):
        if reflectbits(s, L) < s:
            continue
        if p==-1 and reflectbits(s, L) == s:
            continue
        s_list.append(s)
    return len(s_list), np.array(s_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8))(i8,i8)")
def representative_pblock(s, L):
    t = reflectbits(s, L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit('f8[:](i8,i8,i8)')
def pblock_state(r:int, L:int, p:int) -> np.ndarray:
    vec = np.zeros(1<<L, dtype=np.float64)
    t = r
    for _ in range(L):
        vec[t] += p
        t = reflectbits(t,L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8)')
def projective_pblock(s_list:list, M:int, L:int, p:int) -> np.ndarray:
    proj = np.zeros((1<<L, M), dtype=np.float64)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[t,j] += p
            t = reflectbits(t,L)
            if t == r:
                break
    # 归一化每一列
    for j in range(M):
        norm = np.linalg.norm(proj[:, j])
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit
def recover_pblock(state, L, p, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = reflectbits(t,L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += p * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += p * coef / np.sqrt(2)
    return res

###############################################
# zblock
###############################################

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8)")
def construct_zblock_basis(L:int, z:int) -> tuple[int, np.ndarray]:
    """
    - z: -1, 1
    """
    s_list = []
    for s in range(1<<L):
        if invertbits(s, L) < s:
            continue
        if z==-1 and invertbits(s, L) == s:
            continue
        s_list.append(s)
    return len(s_list), np.array(s_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8))(i8,i8)")
def representative_zblock(s:int, L:int) -> tuple[int, int]:
    t = invertbits(s, L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit
def recover_zblock(state, L, pz, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = invertbits(t,L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += pz * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += pz * coef / np.sqrt(2)
    return res

################################################
# pzblock
################################################

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8)")
def construct_pzblock_basis(L:int, pz:int) -> tuple[int, np.ndarray]:
    """
    - pz: -1, 1
    """
    s_list = []
    for s in range(1<<L):
        if reflectbits(invertbits(s, L), L) < s:
            continue
        if pz==-1 and reflectbits(invertbits(s, L), L) == s:
            continue
        s_list.append(s)
    return len(s_list), np.array(s_list)


config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8))(i8,i8)")
def representative_pzblock(s:int, L:int) -> tuple[int, int]:
    t = reflectbits(invertbits(s, L), L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit
def recover_pzblock(state, L, pz, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = reflectbits(invertbits(t,L), L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += pz * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += pz * coef / np.sqrt(2)
    return res


############################################
# kblock
############################################
config.CACHE_DIR = numba_cache_dir
@njit("i8(i8,i8,i8)")
def checkstate_kblock(s:int, L:int, k:int) -> int:
    t = s
    for i in range(1,L+1):
        t = cyclebits(t,1,L)
        if t < s:
            return -1
        elif t == s:
            if k % (L//i) != 0:
                return -1
            return i
    return -1

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8[:],i8[:]))(i8,i8)")
def construct_kblock_basis(L:int, k:int) -> tuple[int, np.ndarray, np.ndarray]:
    """
    - k: 0,1,...,N-1
    """
    a = 0
    s_list = []
    R_list = []
    for s in range(1<<L):
        R = checkstate_kblock(s, L, k)
        if R >= 0:
            s_list.append(s)
            R_list.append(R)
            a = a + 1
    return a, np.array(s_list), np.array(R_list)

config.CACHE_DIR = numba_cache_dir

@njit('Tuple((i8,i8))(i8,i8)')
def representative_kblock(s:int, L:int) -> tuple[int,int]:
    """
    Finding the representative r of a state-integer s
    """
    r = s; t = s; l = 0
    for i in range(1,L):
        t = cyclebits(t,1,L)
        if t < r:
            r = t; l = i
    return r, l

config.CACHE_DIR = numba_cache_dir
@njit("c16[:,:](c16[:,:],i8,i8,i8[:],i8[:])")
def recover_kblock(state, L, k, s_list, R_list):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=np.complex128)
    for j in range(N):
        for a in range(M):
            coef = state[a,j]
            if coef == 0:
                continue
            sqrtRa = np.sqrt(R_list[a])
            r = s_list[a]
            t = r
            for i in range(L):
                res[t, j] += coef * np.exp( - 2j * np.pi * k * i / L) / sqrtRa
                t = cyclebits(t,1,L)
                if t == r:
                    break
    return res

############################################
# kblock_pblock
############################################
config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8))(i8,i8,i8)")
def checkstate_kblock_pblock(s:int, L:int, k:int) -> int:
    R = -1
    m = -1
    
    t = s
    for i in range(1,L+1):
        t = cyclebits(t, 1, L)
        if t < s:
            return R, m
        elif t == s:
            if k % (L//i) != 0:
                return R, m
            R = i
            break
    
    t = reflectbits(s, L)
    for i in range(R):
        if t < s:
            return -1, m
        elif t == s:
            m = i
            return R, m
        t = cyclebits(t, 1, L)
    return R, m

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:], i8[:], i8[:]))(i8,i8,i8)")
def construct_kblock_pblock_basis(L:int, k:int, p:int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    - Nup: 0,1,...,N
    - k: 0,1,...,N//2
    - p: -1, 1
    """
    s_list = []
    R_list = []
    m_list = []
    sigma_list = [1] if k == 0 or 2*k == L else [-1, 1]
    
    for s in range(1 << L):
        R, m = checkstate_kblock_pblock(s, L, k)
        for sigma in sigma_list:
            if m != -1:
                if np.abs(1 + sigma * p * np.cos(k*m*2*np.pi/L)) < 1e-6:
                    continue
                if sigma == -1 and np.abs(1 - sigma * p * np.cos(k*m*2*np.pi/L)) > 1e-6:
                    continue
            if R > 0:
                s_list.append(s)
                R_list.append(sigma * R)
                m_list.append(m)
        
    return len(s_list), np.array(s_list), np.array(R_list), np.array(m_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8, i8))(i8,i8)")
def representative_kblock_pblock(s:int, N:int) -> tuple[int, int, int]:
    r = s; t = s; l = 0
    
    for i in range(1,N):
        t = cyclebits(t, -1, N)
        if t < r:
            r = t
            l = i
    
    t = reflectbits(s, N)
    q = 0
    for i in range(N):
        if t < r:
            r = t
            l = i
            q = 1
        t = cyclebits(t, -1, N)
    return r, -l, q

################################################
# Nup pblock
################################################
config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8,i8)")
def construct_Nup_pblock_basis(L:int, Nup:int, p:int) -> tuple[int, np.ndarray]:
    """
    - Nup: 0,1,...,L
    - p: -1, 1
    """
    s_list = []
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << (L - Nup)) - 1
    
    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        if reflectbits(s, L) < s:
            if s == 0: break
            s = next_combination(s)
            continue
        if p==-1 and reflectbits(s, L) == s:
            if s == 0: break
            s = next_combination(s)
            continue
        s_list.append(s)
        if s == 0: break
        s = next_combination(s)
    return len(s_list), np.array(s_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8))(i8,i8)")
def representative_Nup_pblock(s, L):
    t = reflectbits(s, L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit
def recover_Nup_pblock(state, L, p, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = reflectbits(t,L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += p * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += p * coef / np.sqrt(2)
    return res

###############################################
# Nup zblock
###############################################

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8,i8)")
def construct_Nup_zblock_basis(L:int, Nup:int, z:int) -> tuple[int, np.ndarray]:
    """
    - Nup: 0,1,...,L//2
    - z: -1, 1
    """
    # 初始值：前Nup个位为1，其他位为0
    s_list = []
    lis = [(1 << Nup) - 1] if L == 2*Nup else [(1 << (L - Nup)) - 1, (1 << Nup) - 1]
    for s in lis:
        # 在 N 个位置内生成具有 Nup 个1的所有可能组合
        while s < (1 << L):
            if invertbits(s, L) < s:
                if s == 0: break
                s = next_combination(s)
                continue
            if z==-1 and invertbits(s, L) == s:
                if s == 0: break
                s = next_combination(s)
                continue
            s_list.append(s)
            if s == 0: break
            s = next_combination(s)
    s_list = np.array(s_list, dtype=np.int64)
    return len(s_list), np.sort(s_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8))(i8,i8)")
def representative_Nup_zblock(s:int, L:int) -> tuple[int, int]:
    t = invertbits(s, L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit
def recover_Nup_zblock(state, L, z, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = invertbits(t,L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += z * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += z * coef / np.sqrt(2)
    return res


###############################################
# Nup pzblock
###############################################

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:]))(i8,i8,i8)")
def construct_Nup_pzblock_basis(L:int, Nup:int, pz:int) -> tuple[int, np.ndarray]:
    """
    - Nup: 0,1,...,L//2
    - pz: -1, 1
    """
    # 初始值：前Nup个位为1，其他位为0
    s_list = []
    lis = [(1 << Nup) - 1] if L == 2*Nup else [(1 << (L - Nup)) - 1, (1 << Nup) - 1]
    for s in lis:
        # 在 N 个位置内生成具有 Nup 个1的所有可能组合
        while s < (1 << L):
            if reflectbits(invertbits(s, L), L) < s:
                if s == 0: break
                s = next_combination(s)
                continue
            if pz==-1 and reflectbits(invertbits(s, L), L) == s:
                if s == 0: break
                s = next_combination(s)
                continue
            s_list.append(s)
            if s == 0: break
            s = next_combination(s)
    s_list = np.array(s_list, dtype=np.int64)
    return len(s_list), np.sort(s_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8))(i8,i8)")
def representative_Nup_pzblock(s:int, L:int) -> tuple[int, int]:
    t = reflectbits(invertbits(s, L), L)
    if t < s:
        return t, 1
    else:
        return s, 0

config.CACHE_DIR = numba_cache_dir
@njit('f8[:](i8,i8,i8)')
def Nup_pzblock_state(r:int, L:int, pz:int) -> np.ndarray:
    vec = np.zeros(1<<L, dtype=np.float64)
    t = r
    for _ in range(L):
        vec[t] += pz
        t = reflectbits(invertbits(t,L),L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('f8[:](i8,i8,i8,i8,i8[:])')
def Nup_pzblock_state_in_Nup(r, L, pz, M, s_list):
    vec = np.zeros(M, dtype=np.float64)
    t = r
    for i in range(L):
        vec[findstate(s_list, t)] += pz
        t = reflectbits(invertbits(t,L),L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8)')
def projective_Nup_pzblock(s_list:list, M:int, L:int, pz:int) -> np.ndarray:
    proj = np.zeros((1<<L, M), dtype=np.float64)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[t,j] += pz
            t = reflectbits(invertbits(t,L),L)
            if t == r:
                break
    # 归一化每一列
    for j in range(M):
        norm = np.linalg.norm(proj[:, j])
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8,i8,i8[:])')
def projective_Nup_pzblock_in_Nup(s_list:list, M:int, L:int, pz:int, M_Nup:int, s_list_Nup:list) -> np.ndarray:
    proj = np.zeros((M_Nup, M), dtype=np.float64)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[findstate(s_list_Nup, t),j] += pz
            t = reflectbits(invertbits(t,L),L)
            if t == r:
                break
    # 归一化每一列
    for j in range(M):
        norm = np.linalg.norm(proj[:, j])
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit
def recover_Nup_pzblock(state, L, pz, s_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for i in range(N):
        for a in range(M):
            coef = state[a,i]
            if coef == 0:
                continue
            t = s_list[a]
            tp = reflectbits(invertbits(t,L),L)
            if tp == t:
                res[t, i] += coef
            elif tp < t:
                res[t, i] += pz * coef / np.sqrt(2)
                res[tp, i] += coef / np.sqrt(2)
            else:
                res[t, i] += coef / np.sqrt(2)
                res[tp, i] += pz * coef / np.sqrt(2)
    return res


##############################################
# Nup kblock
##############################################
config.CACHE_DIR = numba_cache_dir
@njit("i8(i8,i8,i8)")
def checkstate_Nup_kblock(s:int, L:int, k:int) -> int:
    t = s
    for i in range(1,L+1):
        t = cyclebits(t,1,L)
        if t < s:
            return -1
        elif t == s:
            if k % (L//i) != 0:
                return -1
            return i
    return -1

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8[:],i8[:]))(i8,i8,i8)")
def construct_Nup_kblock_basis(L:int, Nup:int, k:int) -> tuple[int, np.ndarray, np.ndarray]:
    """
    - Nup: 0,1,...,N
    - k: 0,1,...,N-1
    """
    a = 0
    s_list = []
    R_list = []
    
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << (L - Nup)) - 1
    
    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        R = checkstate_Nup_kblock(s, L, k)
        if R >= 0:
            s_list.append(s)
            R_list.append(R)
            a = a + 1
        if s == 0:
            break
        s = next_combination(s)
        
    return a, np.array(s_list), np.array(R_list)

config.CACHE_DIR = numba_cache_dir
@njit('Tuple((i8,i8))(i8,i8)')
def representative_Nup_kblock(s:int, N:int) -> tuple[int,int]:
    """
    Finding the representative r of a state-integer s
    """
    r = s; t = s; l = 0
    for i in range(1,N):
        t = cyclebits(t,1,N)
        if t < r:
            r = t; l = i
    return r, l

config.CACHE_DIR = numba_cache_dir
@njit('c16[:](i8,i8,i8)')
def Nup_kblock_state(r:int, L:int, k:int) -> np.ndarray:
    vec = np.zeros(1<<L, dtype=np.complex128)
    t = r
    for i in range(L):
        vec[t] += np.exp(-2j*np.pi*k*i/L)
        t = cyclebits(t,1,L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('c16[:](i8,i8,i8,i8,i8[:])')
def Nup_kblock_state_in_Nup(r, L, k, M, s_list):
    vec = np.zeros(M, dtype=np.complex128)
    t = r
    for i in range(L):
        vec[findstate(s_list, t)] += np.exp(-2j*np.pi*k*i/L)
        t = cyclebits(t,1,L)
        if t == r:
            break
    vec /= np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('c16[:,:](i8[:],i8,i8,i8)')
def projective_Nup_kblock(s_list:list, M:int, L:int, k:int) -> np.ndarray:
    proj = np.zeros((1<<L, M), dtype=np.complex128)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[t,j] += np.exp(-2j*np.pi*k*i/L)
            t = cyclebits(t,1,L)
            if t == r:
                break
    # 归一化每一列
    for j in range(M):
        norm = np.sqrt(np.sum(np.abs(proj[:, j])**2))
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit('c16[:,:](i8[:],i8,i8,i8,i8,i8[:])')
def Nup_kblock_projective_in_Nup(s_list:list, M:int, L:int, k:int, M_Nup:int, s_list_Nup:list) -> np.ndarray:
    proj = np.zeros((M_Nup, M), dtype=np.complex128)
    for j in range(M):
        r = s_list[j]
        t = r
        for i in range(L):
            proj[findstate(s_list_Nup, t),j] += np.exp(-2j*np.pi*k*i/L)
            t = cyclebits(t,1,L)
            if t == r:
                break
    # 归一化每一列
    for j in range(M):
        norm = np.sqrt(np.sum(np.abs(proj[:, j])**2))
        if norm != 0:
            proj[:, j] /= norm
    return proj


config.CACHE_DIR = numba_cache_dir
@njit("c16[:,:](c16[:,:],i8,i8,i8[:],i8[:])")
def recover_Nup_kblock(state, L, k, s_list, R_list):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=np.complex128)
    for j in range(N):
        for a in range(M):
            coef = state[a,j]
            if coef == 0:
                continue
            sqrtRa = np.sqrt(R_list[a])
            r = s_list[a]
            t = r
            for i in range(L):
                res[t, j] += coef * np.exp( - 2j * np.pi * k * i / L) / sqrtRa
                t = cyclebits(t,1,L)
                if t == r:
                    break
    return res



###############################################
# Nup kblock pblock
###############################################
config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8))(i8,i8,i8)")
def checkstate_Nup_kblock_pblock(s:int, L:int, k:int) -> int:
    R = -1
    m = -1
    
    t = s
    for i in range(1,L+1):
        t = cyclebits(t, 1, L)
        if t < s:
            return R, m
        elif t == s:
            if k % (L//i) != 0:
                return R, m
            R = i
            break
    
    t = reflectbits(s, L)
    for i in range(R):
        if t < s:
            return -1, m
        elif t == s:
            m = i
            return R, m
        t = cyclebits(t, 1, L)
    return R, m

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8[:], i8[:], i8[:]))(i8,i8,i8,i8)")
def construct_Nup_kblock_pblock_basis(L:int, Nup:int, k:int, p:int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    - Nup: 0,1,...,N
    - k: 0,1,...,N//2
    - p: -1, 1
    """
    s_list = []
    R_list = []
    m_list = []
    sigma_list = [1] if k == 0 or 2*k == L else [-1, 1]
    
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << (L - Nup)) - 1
    
    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        R, m = checkstate_Nup_kblock_pblock(s, L, k)
        for sigma in sigma_list:
            if m != -1:
                if np.abs(1 + sigma * p * np.cos(k*m*2*np.pi/L)) < 1e-6:
                    continue
                if sigma == -1 and np.abs(1 - sigma * p * np.cos(k*m*2*np.pi/L)) > 1e-6:
                    continue
            if R > 0:
                s_list.append(s)
                R_list.append(sigma * R)
                m_list.append(m)
        if s == 0:
            break
        s = next_combination(s)
        
    return len(s_list), np.array(s_list), np.array(R_list), np.array(m_list)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8, i8))(i8,i8)")
def representative_Nup_kblock_pblock(s:int, N:int) -> tuple[int, int, int]:
    r = s; t = s; l = 0
    
    for i in range(1,N):
        t = cyclebits(t, -1, N)
        if t < r:
            r = t
            l = i
    
    t = reflectbits(s, N)
    q = 0
    for i in range(N):
        if t < r:
            r = t
            l = i
            q = 1
        t = cyclebits(t, -1, N)
    return r, -l, q

config.CACHE_DIR = numba_cache_dir
@njit("f8[:](i8,i8,i8,i8,i8)")
def Nup_kblock_pblock_state(s:int, L:int, k:int, p:int, sigma:int) -> np.ndarray:
    vec = np.zeros(1<<L, dtype=np.float64)
    for r in range(L):
        i = cyclebits(s, r, L)
        vec[i] += np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L)
        j = reflectbits(cyclebits(s, r, L), L)
        vec[j] += p * (np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L))
    vec = vec / np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit("f8[:](i8,i8,i8,i8,i8,i8,i8[:])")
def Nup_kblock_pblock_state_in_Nup(s:int, L:int, k:int, p:int, sigma:int, M:int, s_list:list) -> np.ndarray:
    vec = np.zeros(M, dtype=np.float64)
    for r in range(L):
        i = cyclebits(s, r, L)
        vec[findstate(s_list, i)] += np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L)
        j = reflectbits(cyclebits(s, r, L), L)
        vec[findstate(s_list, j)] += p * (np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L))
    vec = vec / np.linalg.norm(vec)
    return vec

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8,i8,i8[:])')
def projective_Nup_kblock_pblock(s_list:list, M:int, L:int, k:int, p:int, R_list:list) -> np.ndarray:
    proj = np.zeros((1<<L, M), dtype=np.float64)
    for t in range(M):
        s = s_list[t]
        sigma = np.sign(R_list[t])
        for r in range(L):
            i = cyclebits(s, r, L)
            proj[i, t] += np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L)
            j = reflectbits(cyclebits(s, r, L), L)
            proj[j, t] += p * (np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L))
    # 归一化每一列
    for j in range(M):
        norm = np.linalg.norm(proj[:, j])
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit('f8[:,:](i8[:],i8,i8,i8,i8,i8[:],i8,i8[:])')
def projective_Nup_kblock_pblock_in_Nup(s_list:list, M:int, L:int, k:int, p:int, R_list:list, M_Nup:int, s_list_Nup:list) -> np.ndarray:
    proj = np.zeros((M_Nup, M), dtype=np.float64)
    for t in range(M):
        s = s_list[t]
        sigma = np.sign(R_list[t])
        for r in range(L):
            i = cyclebits(s, r, L)
            proj[findstate(s_list_Nup, i), t] += np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L)
            j = reflectbits(cyclebits(s, r, L), L)
            proj[findstate(s_list_Nup, j), t] += p * (np.sin(k*r * 2*np.pi/L) if sigma == -1 else np.cos(k*r * 2*np.pi/L))
    # 归一化每一列
    for j in range(M):
        norm = np.linalg.norm(proj[:, j])
        if norm != 0:
            proj[:, j] /= norm
    return proj

config.CACHE_DIR = numba_cache_dir
@njit
def recover_Nup_kblock_pblock(state, L, k, p, s_list, R_list, m_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for ii in range(N):
        for t in range(M):
            coef = state[t,ii]
            if coef == 0:
                continue
                
            s = s_list[t]
            mu_a = m_list[t]
            TPa = mu_a != -1
            
            sigma = np.sign(R_list[t])
            R = np.abs(R_list[t])
            norm = L**2 / R * (1 + sigma * p * np.cos(2*np.pi*k*mu_a/L)) if TPa else L**2 / R
            if k == 0 or 2*k == L:
                norm *= 2
            if sigma == -1:
                for r in range(L):
                    tmp = np.sin(k*r * 2*np.pi/L) * coef
                    i = cyclebits(s, r, L)
                    res[i, ii] += tmp / (np.sqrt(norm))
                    i = reflectbits(cyclebits(s, r, L), L)
                    res[i, ii] += p * tmp  / (np.sqrt(norm))
            else:
                for r in range(L):
                    tmp = np.cos(k*r * 2*np.pi/L) * coef
                    i = cyclebits(s, r, L)
                    res[i, ii] += tmp / (np.sqrt(norm))
                    i = reflectbits(cyclebits(s, r, L), L)
                    res[i, ii] += p * tmp  / (np.sqrt(norm))
            
    return res

###############################################
# Nup kblock pblock zblock
###############################################
config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8, i8, i8))(i8,i8,i8)")
def checkstate_Nup_kblock_pblock_zblock(s:int, L:int, k:int) -> tuple[int, int, int, int]:
    R = -1  # 循环周期
    m = -1  # 反射或者反转后的循环
    n = -1  # 反射+反转后的循环
    c = 1   # 代表性质，1，2，3，4，5
    
    t = s
    for i in range(1,L+1):
        t = cyclebits(t, 1, L)
        if t < s:  # not a representative
            return R, m, n, c
        elif t == s:  # may be representative
            if k % (L//i) != 0:  # not representative for this k block
                return R, m, n, c
            R = i  # representative for k block
            break  # not enough, need to check case after reflection
    
    t = reflectbits(s, L)
    for i in range(R):  # only need to check up to R-1
        if t < s:  # not a representative, cause there is a smaller one
            return -1 ,m, n, c
        elif t == s:  # may be representative
            m = i
            c = 2
            break  # not enough, need to check case after inversion
        t = cyclebits(t, 1, L)
    
    t = invertbits(s, L)
    for i in range(R):
        if t < s:  # not a representative, cause there is a smaller one
            return -1 ,m, n, c
        elif t == s:  # may be representative
            if m == -1:
                m = i
                c = 3
            else:
                n = i
                c = 5
            break  # not enough, need to check case after inversion and reflection
        t = cyclebits(t, 1, L)
    
    t = reflectbits(invertbits(s, L), L)
    for i in range(R):
        if t < s:
            return -1 ,m, c, n
        elif t == s:
            if m == -1:
                m = i
                c = 4
            else:
                if n == -1:
                    raise Exception("Should not happen")
            break
        t = cyclebits(t, 1, L)
    
    return R, m, n, c

config.CACHE_DIR = numba_cache_dir
@njit("f8(i8,i8,i8,i8,i8,i8,i8,i8)")
def get_nmc(L, k, p, z, sigma, m, n, c):
    kpi = k * 2 *np.pi / L
    if c == 2:
        nmc = 1 + sigma * p * np.cos(m*kpi)
    elif c == 3:
        nmc = 1 + z * np.cos(m*kpi)
    elif c == 4:
        nmc = 1 + sigma * p * z * np.cos(m*kpi)
    elif c == 5:
        nmc = (1 + sigma * p * np.cos(m*kpi))*(1 + z * np.cos(n*kpi))
    else:
        nmc = 1.
    return nmc

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8,i8[:], i8[:], i8[:], i8[:]))(i8,i8,i8,i8)")
def construct_Nup_kblock_pblock_zblock_basis(L:int, k:int, p:int, z:int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    - N: 偶数, 只能考虑半满空间：Nup = N/2
    - k: 0,1,...,N//2
    - p: -1, 1
    - z: -1, 1
    """
    assert L % 2 == 0, "N must be even"
    s_list = []
    R_list = []
    m_list = []
    c_list = []
    sigma_list = [1] if k == 0 or 2*k == L else [-1, 1]
    
    # 初始值：前Nup个位为1，其他位为0
    s = (1 << L//2) - 1
    
    # 在 N 个位置内生成具有 Nup 个1的所有可能组合
    while s < (1 << L):
        R, m, n, c = checkstate_Nup_kblock_pblock_zblock(s, L, k)
        for sigma in sigma_list:
            nmc = 1.
            if m != -1:
                nmc = get_nmc(L, k, p, z, sigma, m, n, c)
                if nmc == 0.:
                    continue
                if c == 2:
                    if sigma == -1 and np.abs(1 - sigma * p * np.cos(k*m*2*np.pi/L)) > 1e-6:
                        continue
                elif c == 4:
                    if sigma == -1 and np.abs(1 - sigma * p * z * np.cos(k*m*2*np.pi/L)) > 1e-6:
                        continue
                elif c == 5:
                    if sigma == -1 and np.abs((1 - sigma * p * np.cos(k*m*2*np.pi/L))*(1 + z * np.cos(k*n*2*np.pi/L))) > 1e-6:
                        continue
            if R > 0:
                s_list.append(s)
                R_list.append(sigma * R)
                m_list.append(m)
                c_list.append(c+n+1)
        if s == 0: break
        s = next_combination(s)
        
    return len(s_list), np.array(s_list, dtype=np.int64), np.array(R_list, dtype=np.int64), np.array(m_list, dtype=np.int64), np.array(c_list, dtype=np.int64)

config.CACHE_DIR = numba_cache_dir
@njit("Tuple((i8, i8, i8, i8))(i8,i8)")
def representative_Nup_kblock_pblock_zblock(s:int, L:int) -> tuple[int, int, int, int]:
    r = s; t = s
    
    l = 0
    for i in range(1,L):
        t = cyclebits(t, -1, L)
        if t < r:
            r = t
            l = i
    
    t = reflectbits(s, L)
    q = 0
    for i in range(L):
        if t < r:
            r = t
            l = i
            q = 1
        t = cyclebits(t, -1, L)
    
    t = invertbits(s, L)
    g = 0
    for i in range(L):
        if t < r:
            r = t
            l = i
            g = 1
            q = 0
        t = cyclebits(t, -1, L)
    
    t = invertbits(reflectbits(s, L), L)
    for i in range(L):
        if t < r:
            r = t
            l = i
            g = 1
            q = 1
        t = cyclebits(t, -1, L)
    
    return r, -l, q, g

config.CACHE_DIR = numba_cache_dir
@njit
def recover_Nup_kblock_pblock_zblock(state, L, k, p, z, s_list, R_list, m_list, c_list, dtype):
    M, N = state.shape
    res = np.zeros((1<<L, N), dtype=dtype)
    for ii in range(N):
        for t in range(M):
            coef = state[t,ii]
            if coef == 0:
                continue

            s = s_list[t]
            sigma = np.sign(R_list[t]) 
            mu = m_list[t]
            
            c_p = c_list[t]
            if c_p > 4:
                c = 5
                n = c_p - 6
            else:
                c = c_p
                n = -1

            sigma = np.sign(R_list[t])
            R = np.abs(R_list[t])
            norm = 2 * L**2 / R * get_nmc(L, k, p, z, sigma, mu, n, c)
            
            if k == 0 or 2*k == L:
                norm *= 2
            
            if sigma == -1:
                for r in range(L):
                    tmp = np.sin(k*r * 2*np.pi/L) / np.sqrt(norm)
                    i = cyclebits(s, r, L)
                    res[i, t] += tmp
                    i = reflectbits(cyclebits(s, r, L), L)
                    res[i, t] += p * tmp
                    i = invertbits(cyclebits(s, r, L), L)
                    res[i, t] += z * tmp
                    i = reflectbits(invertbits(cyclebits(s, r, L), L), L)
                    res[i, t] += p*z* tmp
            else:
                for r in range(L):
                    tmp = np.cos(k*r * 2*np.pi/L) / np.sqrt(norm)
                    i = cyclebits(s, r, L)
                    res[i, t] += tmp
                    i = reflectbits(cyclebits(s, r, L), L)
                    res[i, t] += p * tmp
                    i = invertbits(cyclebits(s, r, L), L)
                    res[i, t] += z * tmp
                    i = reflectbits(invertbits(cyclebits(s, r, L), L), L)
                    res[i, t] += p*z* tmp
    return res
