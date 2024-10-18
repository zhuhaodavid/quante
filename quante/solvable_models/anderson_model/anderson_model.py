# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-08-15 10:30:40
# @Last Modified by:   hzhu
# @Last Modified time: 2024-09-11 15:24:44

import numpy as np

__all__ = [
    "anderson_matrix",
    "anderson_kmat",
    "anderson_eigstate",
    "anderson_energies",
    "plot_anderson_band"
]

def anderson_matrix(T:np.ndarray, W:np.ndarray) -> np.ndarray:
    """
    生成安德森模型的矩阵
    
    Example:
    >>> q, l = 2, 1000
    >>> T = np.random.randn(q)
    >>> W = np.random.randn(l)
    >>> for i in range(len(W)):
    >>>     if i != 1:
    >>>         W[i] *= 0
    >>> for i in range(1,l//2+1):
    >>>     W[-i] = W[i]
    >>> anderson_matrix(T, W)

    Parameters
    ----------
    T : np.ndarray
        onsite 能量
    W : np.ndarray
        hopping 能量

    Returns
    -------
    np.ndarray
        矩阵
    """
    l: int = len(W)
    q: int = len(T)
    assert l % q == 0, "len(W) 需要是 len(T) 的倍数"
    
    isherm: bool = True
    for i in range(1, l//2+1):
        isherm = W[-i] == W[i]
    assert isherm, "W 的第 i 个元素需要等于第 l-i 个元素"
    
    from .anderson_model_numba import get_hammat
    return get_hammat(T, W, q, l)


def anderson_kmat(T:np.ndarray, W:np.ndarray, k:int) -> np.ndarray:
    """
    画出 anderson 模型第 k 个矩阵
    
    Example
    >>> q, l = 2, 1000
    >>> T = np.random.randn(q)
    >>> W = np.random.randn(l)
    >>> for i in range(len(W)):
    >>>     if i != 1:
    >>>         W[i] *= 0
    >>> for i in range(1,l//2+1):
    >>>     W[-i] = W[i]
    >>> anderson_kmat(T, W, 0)

    Parameters
    ----------
    T : np.ndarray
        onsite 能量
    W : np.ndarray
        hopping 能量

    Returns
    -------
    np.ndarray
        矩阵
    """
    assert isinstance(k, int)
    l: int = len(W)
    q: int = len(T)
    assert l % q == 0, "len(W) 需要是 len(T) 的倍数"
    
    isherm: bool = True
    for i in range(1, l//2+1):
        isherm = W[-i] == W[i]
    assert isherm, "W 的第 i 个元素需要等于第 l-i 个元素"
    
    from .anderson_model_numba import mat_k
    return mat_k(T, W, k*2*np.pi/l)


def anderson_eigstate(T:int, W:int, vec:np.ndarray, k:int):
    """
    获得第 k 个矩阵，vec 对应的本征态
    
    Example:
    >>> q, l = 2, 1000
    >>> T = np.random.randn(q)
    >>> W = np.random.randn(l)
    >>> for i in range(len(W)):
    >>>     if i != 1:
    >>>         W[i] *= 0
    >>> for i in range(1,l//2+1):
    >>>     W[-i] = W[i]
    >>> k = 0
    >>> mat = anderson_kmat(T, W, k)
    >>> eig, vec = nla.eigh(mat)
    >>> us = anderson_eigstate(l, q, vec[:,0], k)
    >>> hammat = anderson_matrix(T, W)
    >>> println(np.allclose(hammat @ us, eig[0] * us))
    """
    assert isinstance(k, int)
    l: int = len(W)
    q: int = len(T)
    reps: int = l // q
    vectilde = np.fft.fft(vec)
    phis = np.tile(A=vectilde, reps=reps)
    angle = 2*np.pi/l*k
    us = [np.exp(-1j*(angle*i))*phi for i, phi in enumerate(phis)]
    return np.array(us)


def anderson_energies(T:np.ndarray, W:np.ndarray) -> np.ndarray:
    """
    返回 anderson 模型的所有本征能量（矩阵，每一行为一个能带）
    
    Example
    >>> q, l = 1000, 1000
    >>> T = np.random.randn(q)
    >>> W = np.random.randn(l)
    >>> for i in range(len(W)):
    >>>     if i != 1:
    >>>         W[i] *= 0
    >>> for i in range(1,l//2+1):
    >>>     W[-i] = W[i]
    >>> _, engs = anderson_energies(T, W)
    
    Parameters
    ----------
    T : np.ndarray
        onsite 能量
    W : np.ndarray
        hopping 能量

    Returns
    -------
    np.ndarray
        本征值
    """
    l: int = len(W)
    q: int = len(T)
    assert l % q == 0, "len(W) 需要是 len(T) 的倍数"
    reps = l // q
    
    isherm: bool = True
    for i in range(1, l//2+1):
        isherm = W[-i] == W[i]
    assert isherm, "W 的第 i 个元素需要等于第 l-i 个元素"
    
    from .anderson_model_numba import engs_main
    ks: np.ndarray = np.arange(0, reps) *  2*np.pi/l
    engs_k: np.ndarray = engs_main(ks=ks, T=T, W=W)
    
    return ks, engs_k


def plot_anderson_band(T:np.ndarray, W:np.ndarray) -> None:
    """
    画出 anderson 模型的能带
    
    Example
    >>> q, l = 2, 1000
    >>> T = np.random.randn(q)
    >>> W = np.random.randn(l)
    >>> for i in range(len(W)):
    >>>     if i != 1:
    >>>         W[i] *= 0
    >>> for i in range(1,l//2+1):
    >>>     W[-i] = W[i]
    >>> plot_anderson_band(T, W)

    Parameters
    ----------
    T : np.ndarray
        onsite 能量
    W : np.ndarray
        hopping 能量
    """
    ks, engs_k = anderson_energies(T, W)
    
    l: int = len(W)
    q: int = len(T)
    reps: int = l // q
    
    import matplotlib.pyplot as plt
    plt.plot(ks * l / reps, engs_k, '.', markersize=1.5)
    
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    plt.xlabel("$k$")
    plt.ylabel("$E$")
    
