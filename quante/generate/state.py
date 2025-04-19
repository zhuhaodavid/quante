# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 16:50:25
# @Last Modified by:   hzhu
# @Last Modified time: 2025-04-19 15:52:07

"""
生成一些常用的态（`np.ndarray`）
- `basis_state`：通过索引生成态
- `product_state`: 通过字符串，如`0101`生成态
- `neel`: 生成 Neel 态
- `ghz`: 生成 GHZ 态
- `w`: 生成 W 态
- `random`: 生成随机态
- `bloch`: 生成 Bloch 球上的态
"""

import itertools
import numpy as _np
import scipy.sparse as _sparse
import math as _math

from ..linalg.operations import kron
from ..linalg.eig_modified import eigh

from typing import Union, Optional

def spin_up(dtype=float) -> _np.ndarray:
    return _np.array([1., 0.], dtype=dtype)

def spin_down(dtype=float) -> _np.ndarray:
    return _np.array([0., 1.], dtype=dtype)

def basis_state(i, dim, dtype=complex):
    r"""Constructs a unit vector ket:
    
    Parameters
    ----------
    i : int
        Which index should the single non-zero, unit entry.
    dim : int
        Total size of hilbert space.

    Examples
    --------
    >>> qt.generate.state.basis_state(1,4)
    [[0.+0.j]
     [1.+0.j]
     [0.+0.j]
     [0.+0.j]]
    """
    shape = (dim, 1)
    x = _np.zeros(shape, dtype=dtype)
    x[i] = 1.0
    return x


def state_from_string(coef:list[float], basis:list[str]):
    rows = [int(i,2) for i in basis]
    cols = _np.zeros_like(rows)
    return _sparse.coo_array((coef, (rows, cols))).tocsr()


def product_state(updns:list[str], dtype=float):
    """通过字符串生成态
    
    Parameters
    ----------
    binary : sequence of 0s and 1s
        The binary of the computation state.

    Examples
    --------
    >>> qt.generate.state.product_state(["up", "dn"])
    [[0.]
     [1.]
     [0.]
     [0.]]
    """
    tmp = ""
    for b in updns:
        if b == "up" or b == "1":
            tmp += "0"
        elif b == "dn" or b == "0":
            tmp += "1"
        else:
            raise ValueError(f"Invalid value {b} in binary string.")
    result = _np.zeros((2**len(updns), 1), dtype=dtype)
    result[int(tmp, 2)] = 1.0
    return result


def neel(L:int, down_first=False, dtype=float, basis=None):
    """
    生成 Neel 态
    
    L 是链长，down_first 为 True 时，第一个是 down，否则是 up
    
    Examples
    --------
    >>> qt.generate.state.neel(2, down_first=False)
    [[0.]
     [1.]
     [0.]
     [0.]]
    """
    updns = "01" * (L // 2) + (L % 2 == 1) * "0"
    if down_first:
        updns = "1" + updns[:-1]
    updnint = int(updns, 2)
    if basis is None:
        result = _np.zeros((2**len(updns), 1), dtype=dtype)
        result[updnint] = 1.0
    else:
        from .basis.symmetry.spin_half.Nup.defclass import SpinHalfBasisNup
        if isinstance(basis, SpinHalfBasisNup):
            indices = _np.where(basis.s_list == updnint)[0]
            result = _np.zeros((basis.Ns, 1), dtype=dtype)
            result[indices] = 1.0
        else:
            raise ValueError(f"Invalid basis type {type(basis)}")
    return result

def ghz(L):
    """
    生成 ghz 态，即所有自旋都是 up 或 down 的叠加态
    
    Examples
    --------
    >>> qt.generate.state.ghz(2)
    [[0.70710678]
     [0.        ]
     [0.        ]
     [0.70710678]]
    """
    return (basis_state(0, 2**L) +
           basis_state(2**L - 1, 2**L)) / 2.**.5


def w(L, dtype=complex):
    """生成 w 态，即只有一个 down 的态的叠加
    
    Examples
    --------
    >>> qt.generate.state.w(2)
    [[0.        +0.j]
     [0.70710678+0.j]
     [0.70710678+0.j]
     [0.        +0.j]]
    """
    shape = (2**L, 1)
    c = 1.0 / L**0.5
    
    x = _np.zeros(shape, dtype=dtype)
    for i in range(L):
        x[2**i, 0] = c
    return x
    # return sum(basis_state(2**i, 2**n, **kwargs) for i in range(n)) / n**0.5

def random(dim: int, n: int = 1, dtype: type = complex, seed: Optional[int] = None) -> _np.ndarray:
    """
    生成一个随机向量或矩阵，并归一化。

    参数:
    - dim (int): 向量或矩阵的维度。
    - n (int, 可选): 如果生成矩阵，则为矩阵的列数，默认为1。
    - dtype (type, 可选): 数据类型，默认为complex。
    - seed (int, 可选): 随机数生成器的种子，用于复现结果。
    
    Examples
    --------
    >>> qt.generate.state.random(4)
    [[ 0.08615608]
     [-0.63327314]
     [ 0.06310054]
     [ 0.766525  ]]
    """
    rng = _np.random.default_rng(seed)
    if issubclass(dtype, complex):
        ket = rng.standard_normal(size=(dim, n)) + 1j * rng.standard_normal(size=(dim, n))
        ket[:] /= _np.linalg.norm(ket, axis=0)
        return ket
    else:
        ket = rng.standard_normal(size=(dim, n))
        ket[:] /= _np.linalg.norm(ket, axis=0)
        return ket
    

def random_sparse(dim: int, n: int = 1, density: float = 1., dtype: type = complex, seed: Optional[int] = None) -> _sparse.sparray:
    """
    生成一个稀疏的随机向量或矩阵，并归一化。

    参数:
    -----
    - dim (int): 向量或矩阵的维度。
    
    - n (int, 可选): 如果生成矩阵，则为矩阵的列数，默认为1。
    
    - density (float, 可选): 稀疏矩阵的密度，范围在0到1之间。默认为1。
    
    - dtype (type, 可选): 数据类型，默认为complex。
    
    - seed (int, 可选): 随机数生成器的种子，用于复现结果。
    
    
    Examples
    --------
    >>> qt.generate.state.random_sparse(4)
      (0, 0)        0.350909962928404
      (1, 0)        0.30461411299623736
      (2, 0)        0.8644957509435779
      (3, 0)        -0.19162342414642677
    """
    rng = _np.random.default_rng(seed)
    ket = _sparse.coo_array(_sparse.random(dim, n, format="coo", density=density))
    if isinstance(dtype, complex):
        ket.data = rng.standard_normal((ket.nnz,)) + 1j * rng.standard_normal((ket.nnz,))
    else:
        ket.data = rng.standard_normal((ket.nnz,))
    ket = ket.asformat("csr")
    ket[:] /= _np.sum(ket.conj() * ket, axis=0)**0.5 # type: ignore
    return ket

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
    from .matrix import _check_spin_number # type: ignore
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
    from .matrix import pauli_matrix
    x = pauli_matrix('x', S=j)
    y = pauli_matrix('y', S=j)
    z = pauli_matrix('z', S=j)

    vx, vy, vz = _np.real_if_close([co_state.conj() @ i @ co_state for i in [x,y,z]])
    theta = _np.arccos(vz/j)
    phi = _np.arctan2(vy, vx)
    
    # 量子测量
    from ..linalg.eig_modified import eigh
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



def xplus(dtype=float) -> _np.ndarray:
    return _np.array([1., 1.], dtype=dtype) / _np.sqrt(2.)


def xminus(dtype=float) -> _np.ndarray:
    return _np.array([-1., -1.], dtype=dtype) / _np.sqrt(2)


def yplus(dtype=complex) -> _np.ndarray:
    return _np.array([-1., -1.j], dtype=dtype) / _np.sqrt(2)


def yminus(dtype=complex) -> _np.ndarray:
    return _np.array([-1., 1.j], dtype=dtype) / _np.sqrt(2)


def plus(dtype=float) -> _np.ndarray:
    return _np.array([1., 1.], dtype=dtype) / _np.sqrt(2.)


def minus(dtype=float) -> _np.ndarray:
    return _np.array([1., -1.], dtype=dtype) / _np.sqrt(2)


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
    kwargs :
        Supplied to ``qu`` called on state.

    Returns
    -------
    p : immutable vector
        The bell-state ``s``.
    """
    keymap = {"psi-": "psi-", 0: "psi-", "psim": "psi-",
            "psi+": "psi+", 1: "psi+", "psip": "psi+",
            "phi-": "phi-", 2: "phi-", "phim": "phi-",
            "phi+": "phi+", 3: "phi+", "phip": "phi+"}
    c = 2.**-.5
    statemap = {"psi-": lambda: [[0], [c], [-c], [0]],
                "phi+": lambda: [[c], [0], [0], [c]],
                "phi-": lambda: [[c], [0], [0], [-c]],
                "psi+": lambda: [[0], [c], [c], [0]]}
    data = statemap[keymap[s]]()
    return _np.array(data, dtype=dtype)


def singlet(dtype=float):
    """Alias for the 'psi-' bell-state.
    """
    return bell("psi-", dtype=dtype)


def thermal(ham, beta=None, precomp_func=False):
    """Generate a thermal state of a Hamiltonian.

    Parameters
    ----------
    ham : operator or (1d-array, 2d-array)
        Hamiltonian, either full or tuple of (evals, evecs).
    beta : float
        Inverse temperature of state.
    precomp_func : bool, optional
        If True, return a function that takes ``beta``
        only and is closed over the solved hamiltonian.

    Returns
    -------
    operator or callable
        Density operator of thermal state, or function to generate such given
        a temperature.
    """
    if isinstance(ham, (list, tuple)):  # solved already
        evals, evecs = ham
    else:
        evals, evecs = eigh(ham)
    evals -= evals.min()  # offset by min to avoid numeric problems

    def gen_state(b):
        el = _np.exp(-b * evals)
        el /= _np.sum(el)
        return (evecs * el) @ evecs.conjugate().transpose()
    
    return gen_state if precomp_func else gen_state(beta)

def singlet_pairs(n, **kwargs):
    return kron([bell('psi-', **kwargs)] * (n // 2))


def _levi_civita(perm):
    """Compute the generalised levi-civita coefficient for a permutation.

    Parameters
    ----------
    perm : sequence of int
        The permutation, a re-arrangement of ``range(n)``.

    Returns
    -------
    int
        Either -1, 0 or 1.
    """
    n = len(perm)
    if n != len(set(perm)):  # infer there are repeated elements
        return 0
    mat = _np.zeros((n, n), dtype=_np.int32)
    for i, j in zip(range(n), perm):
        mat[i, j] = 1
    return int(_np.linalg.det(mat))


def antisymmetric(*args):
    """Construct the anti-symmetric state which is the +- sum of all
    tensored permutations of states ``ps``.

    Parameters
    ----------
    ps :  sequence of states
        The states to combine.

    Returns
    -------
    vector or operator
        The permutation state, dimension same as ``kron(*ps)``.

    Examples
    --------
    A singlet is the ``perm_state`` of up and down.

    >>> states = [up(), down()]
    >>> pstate = perm_state(states)
    >>> expec(pstate, singlet())
    1.0
    """
    if not args:
        raise TypeError("Requires at least one input argument")
    if len(args) == 1 and isinstance(args[0], list):
        # this is the case when tensor is called on the form:
        # tensor([q1, q2, q3, ...])
        ps = args[0]
    else:
        # this is the case when tensor is called on the form:
        # tensor(q1, q2, q3, ...)
        ps = args
    n = len(ps)
    vec_perm = itertools.permutations(ps)
    ind_perm = itertools.permutations(range(n))

    def terms():
        for vec, ind in zip(vec_perm, ind_perm):
            yield _levi_civita(ind) * kron(*vec) # type: ignore

    return sum(terms()) / _math.factorial(n)**0.5

