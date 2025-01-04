# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-11-20 02:20:47
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-28 16:53:23

import numpy as np
# from .usenumba.numba_settings import njit, numba, numba_cache_dir

__all__ = ['eigh_perturbation']

def eigh_perturbation(H0, H1, H2=None, eps=1e-10):
    r"""微扰计算
    
    用微扰方法计算：
    
    .. math::
        H = H^{(0)} + \lambda H^{(1)} + \lambda^2 H^{(2)} + \cdots
    
    的本征值和本征向量：
    
    .. math::
        E = \lambda E^{(0)} + \lambda^2 E^{(1)} + \lambda^3 E^{(2)} + \cdots \\\\
        U = \lambda U^{(0)} + \lambda^2 U^{(1)} + \lambda^3 U^{(2)} + \cdots
        
    也就是保证下面三个等式成立：
    
    .. math::
        H^{(0)} U^{(0)} &= U^{(0)} E^{(0)} \\\\
        H^{(0)} U^{(1)} + H^{(1)} U^{(0)} &= U^{(0)} E^{(1)} + U^{(1)} E^{(0)} \\\\
        H^{(0)} U^{(2)} + H^{(1)} U^{(1)} + H^{(2)} U^{(0)} 
        &= U^{(0)} E^{(2)} + U^{(1)} E^{(1)} + U^{(2)} E^{(0)}

    这个函数只是验证算法，对于具体问题需要具体优化。
    
    二级矩阵并不唯一（如果简并很多，那么一级矩阵也不唯一），但这个三个等式总是成立，误差总为 ``math:: O(\lambda^3)``
    
    Parameters
    ----------
    H0 : np.ndarray
        矩阵的零级部分
    H1 : np.ndarray
        矩阵的第一级部分
    H2 : np.ndarray
        矩阵的第二级部分
    eps : np.ndarray, optional
        本征值小于这个值就认为是简并的
        （具体问题需要取决于能级间距）, by default 1e-10
    
    Returns
    -------
    E0 : np.ndarray
        零级本征值
    E1 : np.ndarray
        一级本征值
    E2 : np.ndarray
        二级本征值
    U0 : np.ndarray
        零级本征向量
    U1 : np.ndarray
        一级本征向量
    U2 : np.ndarray
        二级本征向量
        
    Examples
    --------
    >>> import quante as qt
    >>> L = 6
    >>> basis = qt.generate.basis.spin_basis(L=L)
    >>> ham0 = qt.generate.operas.heisenberg_operator(L=L, j=(1,0,0))
    >>> H0 = ham0.to_matrix(basis)
    >>> ham1 = qt.generate.operas.heisenberg_operator(L=L, j=(0,0,1))
    >>> H1 = ham1.to_matrix(basis)
    >>> ham2 = qt.generate.operas.heisenberg_operator(L=L, j=(0,0,1))
    >>> H2 = ham2.to_matrix(basis)
    >>> E0, E1, E2, U0, U1, U2 = eigh_perturbation(H0, H1, H2)
    >>> check_perturbation(H0, H1, H2, E0, E1, E2, U0, U1, U2)
    eq1 = 4.366911925232853e-15, eq2 = 6.746548682934744e-15, eq3 = 8.644665959144473e-15
    """
    # 首先需要求解零级部分
    E0, U0 = np.linalg.eigh(H0)
    
    # 首先找到 E0 中简并能量的位置
    # 如果确认无简并可以直接通过 
    # E1 = np.sum(U0.conj().T * H1 @ U0, axis=0)
    # E2 = np.sum(U0.conj().T * H2 @ U0, axis=0)
    # 分解得到一级修正 E1 和 二级修正的近似 E2
    E0_degen_pos = start_end_indx(E0, eps)
    # 通过 H1 优化 U0 中的简并态，顺便得到一级近似能量
    E1 = _optimize_U0_first_order(E0, U0, H1, E0_degen_pos)
    # 通过 H2 优化 U0 中的简并态
    _optimize_U0_second_order(E0, E1, U0, H1, H2, E0_degen_pos, eps)
    
    # 计算在 U0 这组基矢下 H1, H2 的表示
    h1 = U0.conj().T @ H1 @ U0
    if H2 is not None:
        h2 = U0.conj().T @ H2 @ U0  # 当 H2 为 None 时，不计算 H2 相关的修正
    else:
        h2 = None
    
    # 本征态的一级近似
    U1 = _first_order_eigvecs(U0, E0, h1, E1, h2)
    
    # 本征能量的二级近似
    E2 = np.sum(U0.conj() * (H1 @ U1 - E1 * U1), axis=0)
    if h2 is not None:
        E2 += h2.diagonal()
    
    # 本征态的二级近似
    U2 = _second_order_eigvecs(E0, U0, H1, E1, U1, h2)
    
    return E0, E1, E2, U0, U1, U2

def start_end_indx(lis, eps):
    out = np.zeros(len(lis)+1, dtype=int)
    indx = 1
    for pos in range(1,len(lis)):
        if np.abs(lis[pos] - lis[pos-1]) > eps:
            out[indx] = pos
            indx += 1
    out[indx] = len(lis)
    return out[:indx+1]


def _optimize_U0_first_order(E0, U0, H1, E0_degen_pos):
    # 通过优化 U0 最终会得到能量的一级近似 E1
    E1 = np.zeros_like(E0)
    
    # 计算在 U0 这组基矢下 H1 的表示
    h1 = U0.conj().T @ H1 @ U0  #!! 主要的时间消耗
    
    # 逐个处理每个简并
    for i in range(len(E0_degen_pos)-1):
        start, end = E0_degen_pos[i], E0_degen_pos[i+1]
        
        # 如果 E0[i] 简并数是 1，那么直接计算 E1, E2 即可
        if end - start == 1:
            E1[start] = h1[start, start]
            continue
        
        # 如果 E0[i] 有更多简并，那么需要对角化方法找到投影空间
        # 的最优组合
        M1 = h1[start:end, start:end]
        M1_eigvals, M1_eigvecs = np.linalg.eigh(M1)
        
        # 对角化得到的能级就是一级修正
        E1[start:end] = M1_eigvals
        
        # 通过投影空间的本征向量最优组合，更新 U0
        U0[:, start:end] = U0[:, start:end] @ M1_eigvecs
        # todo 在 for 循环中的矩阵乘法效率低，如何优化？
        
    return E1


def _optimize_U0_second_order(E0, E1, U0, H1, H2, E0_degen_pos, eps):
    # 更新在 U0 这组基矢下 H1, H2 的表示
    h1 = U0.conj().T @ H1 @ U0
    if H2 is not None:
        h2 = U0.conj().T @ H2 @ U0  # 当 H2 为 None 时，不计算 H2 相关的修正
    
    # 逐个处理每个简并
    for i in range(len(E0_degen_pos)-1):
        E0_start, E0_end = E0_degen_pos[i], E0_degen_pos[i+1]
        
        # 如果 E0[i] 不简并，那么直接计算 E2 即可
        if E0_end - E0_start == 1:
            # E2[E0_start] = h2[E0_start, E0_start]
            continue
        
        # 如果 E0[i] 有更多简并，那么考察 E1[i] 的简并情况
        # 首先计算 E1[E0_start:E0_end] 中简并的位置，然后
        # 再通过加上 E0_start 映射到原来的 E0 索引上
        E1_degen_pos = E0_start + start_end_indx(E1[E0_start:E0_end], eps)
        
        # 在 E0[i] 的简并中逐个处理 E1[j] 的简并
        for j in range(len(E1_degen_pos) - 1):
            E1_start, E1_end = E1_degen_pos[j], E1_degen_pos[j+1]
            
            # 如果 E1[j] 简并数是 1，那么同样直接计算 E2 即可
            if E1_end - E1_start == 1:
                # E2[j] = h2[j, j]
                continue
                
            # 对于 E1[j] 有更多简并的情况，需要对角化方法找到投影空间
            # 的最优组合，根据微扰理论，H1 同样会对二级修正产生影响，
            # 这里需要对角化的不只有 M2，还需要加上：
            # 
            #                            H^{(1)} |m⟩⟨m| H^{(1)}
            #            ∑             -------------------------
            #  E^{(0)}_m ≠ E^{(0)}_n     E^{(0)}_m - E^{(0)}_n
            # 
            
            # 分母：
            E0diff = E0[E1_start] - E0
            # 需要 E^{(0)}_m ≠ E^{(0)}_n 避免分母发散，所以找到所有
            # E^{(0)}_m, E^{(0)}_n 不同的位置
            E0start, E0end = find_start_end(E0, E1_start, 0, len(E0), eps=eps)
            # 需要求和的包括 (0, E1_start) 和 (E1_end, len(E0)) 两个部分
            vec1 = h1[E1_start:E1_end, :E0start]
            vec2 = h1[E1_start:E1_end, E0end:]
            M2 = (vec1/E0diff[:E0start]) @ vec1.conj().T + (vec2/E0diff[E0end:]) @ vec2.conj().T
            
            # 还要补上 H2 相关的修正
            M2 += h2[E1_start:E1_end, E1_start:E1_end]
            
            # 通过对角化找到最优组合
            _, M2_eigvecs = np.linalg.eigh(M2)
            # 更新 U0
            U0[:, E1_start:E1_end] = U0[:, E1_start:E1_end] @ M2_eigvecs


def _first_order_eigvecs(U0, E0, h1, E1, h2=None, eps=1e-10):
    # 计算一级修正中的第一项
    # 
    #                          ⟨ E^{(0)}_m | H^{(1)} | E^{(0)}_n ⟩
    #            ∑           --------------------------------------  | E^{(0)}_m ⟩
    #  E^{(0)}_m ≠ E^{(0)}_n         E^{(0)}_n - E^{(0)}_m
    # 
    tmp1 = np.zeros_like(h1)
    for n in range(len(E0)):
        E0diff = E0[n] - E0
        E0_noteq_indx = np.abs(E0diff) > 1e-10
        tmp1[E0_noteq_indx, n] = h1[E0_noteq_indx, n] / (E0diff[E0_noteq_indx])
    U1 = U0 @ tmp1

    # 计算一级修正中的第二项
    # 
    #            ∑            ⟨E^{(0)}_m|H^{(1)}|E^{(0)}_l⟩ ⟨E^{(0)}_l|H^{(1)}|E^{(0)}_n⟩
    #  E^{(1)}_m ≠ E^{(1)}_n  ---------------------------------------------------------- |E^{(0)}_m⟩
    #  E^{(0)}_l ≠ E^{(0)}_n       (E^{(1)}_n - E^{(1)}_m) (E^{(0)}_n - E^{(0)}_l)
    # 
    # todo 如何提高下面的计算效率？
    tmp2 = np.zeros_like(U0)
    for n in range(len(E0)):
        # 计算求和范围
        E0start, E0end = find_start_end(E0, n, left=0, right=len(E0), eps=eps)
        if E0start == E0end:
            continue
        E1start, E1end = find_start_end(E1, n, left=E0start, right=E0end, eps=eps)
        if E1start == E1end:
            continue
        
        # 分母
        E0diff = E0[n] - E0
        E1diff = E1[n] - E1
        
        # 分别处理 E0start:E1start, E1end:E0end
        idx = ((E0start, E1start), (E1end, E0end))
        for a, b in idx:
            tmp2[a:b, n] = h1[a:b, :E0start] @ tmp1[:E0start, n] + h1[a:b, E0end:] @ tmp1[E0end:, n]
            # 顺带加上 H2 相关的修正
            #                          ⟨ E^{(0)}_m | H^{(2)} | E^{(0)}_n ⟩
            #            ∑           --------------------------------------  | E^{(0)}_m ⟩
            #  E^{(1)}_m ≠ E^{(1)}_n         E^{(1)}_n - E^{(1)}_m
            if h2 is not None:
                tmp2[a:b, n] += h2[a:b, n]
            tmp2[a:b, n] /= E1diff[a:b]
            
    U1 += U0 @ tmp2
    
    return U1

# numba.config.CACHE_DIR = numba_cache_dir
# @njit(cache=True)
def find_start_end(v, n, left, right, eps):
    v0 = v[n]
    start = n
    while start > left-1 and np.abs(v[start] - v0) < eps:
        start -= 1
    end = n
    while end < right and np.abs(v[end] - v0) < eps:
        end += 1
    return start+1, end


def _second_order_eigvecs(E0, U0, H1, E1, U1, h2=None):
    # 计算本征态的二级修正:
    # 
    #                          ⟨ E^{(0)}_m |H^{(2)} | E^{(0)}_n ⟩
    #            ∑           --------------------------------------  | E^{(0)}_m ⟩
    #  E^{(0)}_m ≠ E^{(0)}_n        E^{(0)}_n - E^{(0)}_m
    
    # 
    #                          ⟨ E^{(0)}_m | H^{(1)} | E^{(1)}_n ⟩
    #            ∑           --------------------------------------  | E^{(0)}_m ⟩
    #  E^{(0)}_m ≠ E^{(0)}_n         E^{(0)}_n - E^{(0)}_m
    
    # 
    #                          E^{(1}}_n ⟨E^{(0)}_m|E^{(1)}_n⟩
    #            ∑           --------------------------------------  | E^{(0)}_m ⟩
    #  E^{(0)}_m ≠ E^{(0)}_n         E^{(0)}_n - E^{(0)}_m
    # 
    # 分子部分
    tmp = U0.conj().T @ H1 @ U1 - E1.reshape(1,-1) * (U0.conj().T @ U1)
    # 分母部分
    for n in range(len(E1)):
        E0diff = E0[n] - E0
        E0_noteq_indx = np.abs(E0diff) > 1e-10
        if h2 is not None:
            tmp[E0_noteq_indx, n] += h2[E0_noteq_indx, n]
        tmp[E0_noteq_indx, n] /= (E0diff[E0_noteq_indx])
    return U0 @ tmp

def check_perturbation(H0, H1, H2, E0, E1, E2, U0, U1, U2):
    # 验证微扰计算的结果是否正确
    eq1 = np.linalg.norm(H0 @ U0 - U0 @ np.diag(E0))
    eq2 = np.linalg.norm(H0 @ U1 + H1 @ U0 - U0 @ np.diag(E1) - U1 @ np.diag(E0))
    eq3 = np.linalg.norm(H0 @ U2 + H1 @ U1 + H2 @ U0 - U0 @ np.diag(E2) - U1 @ np.diag(E1) - U2 @ np.diag(E0))
    print(f"eq1 = {eq1}, eq2 = {eq2}, eq3 = {eq3}")
    return eq1, eq2, eq3
