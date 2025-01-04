# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-08-15 10:30:40
# @Last Modified by:   hzhu
# @Last Modified time: 2024-11-09 19:33:03

import numpy as np

__all__ = [
    "anderson_matrix",
    "anderson_kmat",
    "anderson_eigstate",
    "anderson_energies",
    "plot_anderson_band"
]

def anderson_matrix(T:np.ndarray, W:np.ndarray) -> np.ndarray:
    r"""生成安德森模型的矩阵
    
    .. math:: 
        H = \sum_{i}^{} T_{i} c_{i}^{\dagger} c_{i} + \sum_{ir}^{} W_{r} c_{i}^{\dagger} c_{i + r}
        
    也可以写作：
    
    .. math:: 
        H = \sum_{i}^{} T_{i} \ket{i}\hspace{-1mm}\bra{i} + \sum_{ir}^{} W_{r} \ket{i}\hspace{-1mm}\bra{i + r} 
    
    以及：
    
    .. math:: 
        \begin{bmatrix}
            T_0 + W_0 & W_1 & W_2 &  \cdots  & W_{L - 1}  \\
            W_{ - 1} & T_1 + W_0 & W_1 & \cdots & W_{L - 2}  \\
            W_{ - 2} & W_{ - 1} & T_2 + W_0 & \cdots & W_{L - 3}  \\
            \cdots  & \cdots  & \cdots  & \cdots & \cdots   \\
            W_{ -(L - 1)}  & W_{ - (L - 2)}  & W_{ -(L - 3)}  & \cdots & T_{L - 1} + W_0  \\
        \end{bmatrix}
    
    Parameters
    ----------
    T : np.ndarray
        onsite 能量
    
    W : np.ndarray
        hopping 能量

    Examples
    --------
    >>> q, l = 2, 1000
    >>> T = np.random.randn(q)
    >>> W = np.random.randn(l)
    >>> for i in range(len(W)):
    >>>     if i != 1:
    >>>         W[i] *= 0
    >>> for i in range(1,l//2+1):
    >>>     W[-i] = W[i]
    >>> anderson_matrix(T, W)

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
    
    Examples
    --------
    >>> q, l = 2, 1000
    >>> T = np.random.randn(q)
    >>> W = np.random.randn(l)
    >>> for i in range(len(W)):
    >>>     if i != 1:
    >>>         W[i] *= 0
    >>> for i in range(1,l//2+1):
    >>>     W[-i] = W[i]
    >>> anderson_kmat(T, W, 0)
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


def anderson_eigstate(l:int, q:int, vec:np.ndarray, k:int):
    """
    获得第 k 个矩阵，vec 对应的本征态
    
    Examples
    --------
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
    reps: int = l // q
    vectilde = np.fft.fft(vec)
    phis = np.tile(A=vectilde, reps=reps)
    angle = 2*np.pi/l*k
    us = [np.exp(-1j*(angle*i))*phi for i, phi in enumerate(phis)]
    return np.array(us)


def anderson_energies(T:np.ndarray, W:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 anderson 模型的所有本征能量（矩阵，每一行为一个能带）
    
    Parameters
    ----------
    T : np.ndarray
        onsite 能量
        
    W : np.ndarray
        hopping 能量

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        能量，每一行为一个能带
    
    Examples
    --------
    >>> q, l = 1000, 1000
    >>> T = np.random.randn(q)
    >>> W = np.random.randn(l)
    >>> for i in range(len(W)):
    >>>     if i != 1:
    >>>         W[i] *= 0
    >>> for i in range(1,l//2+1):
    >>>     W[-i] = W[i]
    >>> _, engs = anderson_energies(T, W)
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

    Parameters
    ----------
    T : np.ndarray
        onsite 能量
        
    W : np.ndarray
        hopping 能量
    
    Examples
    --------
    >>> q, l = 2, 1000
    >>> T = np.random.randn(q)
    >>> W = np.random.randn(l)
    >>> for i in range(len(W)):
    >>>     if i != 1:
    >>>         W[i] *= 0
    >>> for i in range(1,l//2+1):
    >>>     W[-i] = W[i]
    >>> plot_anderson_band(T, W)
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
    
