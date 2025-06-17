# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 17:50:10
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 09:47:21

import numpy as _np

from typing import Union
from .general import onehot

 # single spin states
def up(dtype=float) -> _np.ndarray:
    return onehot(0, 2, dtype=dtype)

def down(dtype=float) -> _np.ndarray:
    return onehot(1, 2, dtype=dtype)

def plus(dtype=float) -> _np.ndarray:
    return _np.ones((2, 1), dtype=dtype) / _np.sqrt(2.)

def minus(dtype=float) -> _np.ndarray:
    return _np.array([[1.], [-1.]], dtype=dtype) / _np.sqrt(2)

def yplus(dtype=complex) -> _np.ndarray:
    return _np.array([[1.], [1.j]], dtype=dtype) / _np.sqrt(2)

def yminus(dtype=complex) -> _np.ndarray:
    return _np.array([[1.], [-1.j]], dtype=dtype) / _np.sqrt(2)

# two spin states
def bell(s, dtype=float):
    r"""One of the four bell-states.

    If n = 2**-0.5, they are:

        0. ``'psi-'`` : ``n * ( |01> - |10> )``
        1. ``'psi+'`` : ``n * ( |01> + |10> )``
        2. ``'phi-'`` : ``n * ( |00> - |11> )``
        3. ``'phi+'`` : ``n * ( |00> + |11> )``

    They can be enumerated in this order.

    Parameters
    ----------
    s : str or int
        String of number of state corresponding to above.

    Returns
    -------
    p : immutable vector
        The bell-state ``s``.
    """
    keymap = {"psi-": "psi-", 0: "psi-",
            "psi+": "psi+", 1: "psi+",
            "phi-": "phi-", 2: "phi-",
            "phi+": "phi+", 3: "phi+",}
    c = 2.**-.5
    statemap = {"psi-": lambda: [[0], [c], [-c], [0]],
                "phi+": lambda: [[c], [0], [0], [c]],
                "phi-": lambda: [[c], [0], [0], [-c]],
                "psi+": lambda: [[0], [c], [c], [0]]}
    data = statemap[keymap[s]]()
    return _np.array(data, dtype=dtype)

def singlet(dtype=float):
    """Alias for the 'psi-' bell-state."""
    return bell("psi-", dtype=dtype)

   
def bloch(theta: float, phi: float, j: Union[int, float]) -> _np.ndarray:
    r"""
    角动量相干态，生成一个纯态，其x,y,z测量值在theta,phi方向上。
    
    计算方法是：
    
    .. math:: 
        \left| \theta \phi \right> = \exp \left[ \mathrm{i} \theta (J_{x}\sin \phi - J_{y}\cos \phi) \right] \left| j \right>

    简化得到向量元为：
    
    .. math:: 
        \braket{ jm \vert \theta \phi } = (1 + \gamma \gamma ^* )^{ - j} \gamma^{j - m} \sqrt{\begin{pmatrix} 2j \\ j - m \end{pmatrix}}
    
    Examples
    --------
    >>> j = 100
    >>> theta = np.arccos(np.random.uniform(-1, 1))
    >>> phi = np.random.uniform(0, 2*np.pi)
    >>> co_state = gn.bloch(theta, phi, j=j)
    >>> qt.generate.state.plot_bloch_state(co_state, j=j)
    
    角动量相干态的其他性质参见 Qauntum Signatures of Chaos P268-P269
    """
    from scipy.special import gammaln
    from ..matrix.pauli import _check_spin_number # type: ignore
    j = _check_spin_number(j)
    co_state = _np.zeros(int(2*j+1), dtype=complex)
    if theta == 0:
        co_state[0] = 1.
    elif theta == _np.pi:
        co_state[-1] = 1.
    else:
        gamma = _np.exp(1j*phi) * _np.tan(theta/2)
        logcoef0 = -j * _np.log(1+gamma*gamma.conj())
        for m in _np.arange(-j,j+1,1):
            # 通过使用 gammaln 来避免 comb 的发散问题
            logcomb = gammaln(2*j + 1) - gammaln(j - m + 1) - gammaln(j + m + 1)
            logcoef = logcoef0 + (j-m)*_np.log(gamma) + logcomb/2
            co_state[int(j-m)] = _np.exp(logcoef)
    return co_state


def plot_bloch_state(co_state, j):
    """
    在 Bloch 球中绘制角动量相干态
    """
    from ..matrix import pauli_matrix
    x = pauli_matrix('x', S=j)
    y = pauli_matrix('y', S=j)
    z = pauli_matrix('z', S=j)

    vx, vy, vz = _np.real_if_close([co_state.conj() @ i @ co_state for i in [x,y,z]])
    theta = _np.arccos(vz/j)
    phi = _np.arctan2(vy, vx)
    
    # 量子测量
    from ...linalg.decomp.eig_modified import eigh
    vx, Ux = eigh(x)
    vy, Uy = eigh(y)
    vz = z.diagonal().real

    sample_num = 200

    probs_x = _np.abs(Ux.conj().T @ co_state)**2
    x_meas = _np.random.choice(vx/j, p=probs_x, size=sample_num)
    probs_y = _np.abs(Uy.conj().T @ co_state)**2
    y_meas = _np.random.choice(vy/j, p=probs_y, size=sample_num)
    probs_z = _np.abs(co_state)**2
    z_meas = _np.random.choice(vz/j, p=probs_z, size=sample_num)

    # 均值
    mean_x = _np.mean(x_meas)
    mean_y = _np.mean(y_meas)
    mean_z = _np.mean(z_meas)

    print(f"mean_x = {mean_x}, mean_y = {mean_y}, mean_z = {mean_z}")
    print(f"norm = {_np.sqrt(mean_x**2 + mean_y**2 + mean_z**2)}")

    # 方差
    dx = x_meas - mean_x
    dy = y_meas - mean_y
    dz = z_meas - mean_z

    var_x = _np.sum(dx**2) / sample_num
    var_y = _np.sum(dy**2) / sample_num
    var_z = _np.sum(dz**2) / sample_num

    var = var_x + var_y + var_z
    print(f"var_x = {var_x}, var_y = {var_y}, var_z = {var_z}")

    print(f"var = {var}")

    # 可视化
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制bloch球
    u = _np.linspace(0, 2 * _np.pi, 100)
    v = _np.linspace(0, _np.pi, 100)
    X = _np.outer(_np.cos(u), _np.sin(v))
    Y = _np.outer(_np.sin(u), _np.sin(v))
    Z = _np.outer(_np.ones(_np.size(u)), _np.cos(v))
    ax.plot_wireframe(X, Y, Z, color='r', alpha=0.1) # type: ignore
    # ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # 绘制测量结果
    ax.scatter(x_meas, y_meas, z_meas, c='b', marker='.', alpha=0.2)

    # 绘制均值向量
    ax.quiver(0, 0, 0, mean_x, mean_y, mean_z, color='r')

    # 绘制 (theta, phi) 点
    ax.scatter(_np.cos(phi)*_np.sin(theta), _np.sin(phi)*_np.sin(theta), _np.cos(theta), c='g', marker='x', alpha=1)

    # 绘制球心在 (theta, phi) 点，半径为 sqrt(var) 的圆
    u = _np.linspace(0, 2 * _np.pi, 100)
    v = _np.linspace(0, _np.pi, 100)
    X = _np.sqrt(var)*_np.outer(_np.cos(u), _np.sin(v)) + _np.cos(phi)*_np.sin(theta)
    Y = _np.sqrt(var)*_np.outer(_np.sin(u), _np.sin(v)) + _np.sin(phi)*_np.sin(theta)
    Z = _np.sqrt(var)*_np.outer(_np.ones(_np.size(u)), _np.cos(v)) + _np.cos(theta)
    ax.plot_wireframe(X, Y, Z, color='g', alpha=0.1) # type: ignore

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') # type: ignore
    # 坐标轴比例相等
    ax.set_aspect('equal')

    # 不显示坐标轴
    ax.set_axis_off()

    plt.show()

