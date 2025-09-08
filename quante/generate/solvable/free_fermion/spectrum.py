# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-07-14 15:28:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-08 19:47:03
from ....linalg import eigvalsh
import numpy as _np
import scipy as _sp
from typing import Optional, Union


__all__ = [
    "XY_gdenergy",
    "XY_energies",
    "XY_free_energy",
    "XY_internal_energy",
    "XY_specific_heat",
    "XY_magnetization",
]


#######################################
# XY 通过自由费米子解
######################################

def XY_gdenergy(
    L: Optional[float] = _np.inf,  # 默认为无穷大尺寸的系统
    jxx: Union[float, _np.ndarray] = 1.0, 
    jyy: Union[float, _np.ndarray] = 1.0, 
    jxy: Union[float, _np.ndarray] = 0.0, 
    jyx: Union[float, _np.ndarray] = 0.0, 
    hz: Union[float, _np.ndarray] = 0.0, 
    pauli: int = 0
) -> float:
    r"""计算 **开边界条件** 下的 xy 模型的 **基态能量**

    .. math:: 
        H = \sum_{i = 0}^{L - 2} 
                \left(
                j^{xx}_i s^{x}_{i} s^{x}_{i + 1} + 
                j^{yy}_i s^{y}_{i} s^{y}_{i + 1} + 
                j^{xy}_i s^{x}_{i} s^{y}_{i + 1} + 
                j^{yx}_i s^{y}_{i} s^{x}_{i + 1}
                \right)
            + \sum_{i = 0}^{L - 1} h^{z}_i s^{z}_{i}
    
    平移不变并且，:math:`L \to \infty`，时，系统的基态能量为：
    
    .. math:: 
        E_{g}/L  = 
        \begin{cases}
            \frac{ | \lambda | }{\pi} E(1 - \frac{\gamma^2}{\lambda^2 }) + \frac{ | \gamma |  }{\pi} E(1 - \frac{\lambda^2}{\gamma^2 } ), & \gamma \neq 0, \lambda \neq 0, h = 0;\\
            2 | \gamma | / \pi, & \lambda = h = 0;\\
            2 | \lambda | / \pi, & \gamma = h = 0;\\
            - 4h/\pi, & \lambda = \gamma = h
        \end{cases}
    
    其中
    
    .. math:: 
        \begin{cases}
            \lambda = \frac{1}{4} \left( J_{x} + J_{y} \right) + \frac{\mathrm{i}}{4}\left( J_{yx}-J_{xy} \right) \\
            \gamma = \frac{1}{4} \left( J_{x} - J_{y} \right) + \frac{\mathrm{i}}{4}\left( J_{yx} + J_{xy} \right) \\
            h = \frac{1}{2}h_{z}
        \end{cases}

    Parameters
    ----------
    L: int, optional
        系统的大小。如果为np.inf，则计算无穷大系统的能量。
    jxx: float or numpy.ndarray, optional
        x方向的交换常数。
    jyy: float or numpy.ndarray, optional
        y方向的交换常数。
    jxy: float or numpy.ndarray, optional
        交叉耦合常数。
    jyx: float or numpy.ndarray, optional
        另一种交叉耦合常数。
    hz: float or numpy.ndarray, optional
        外磁场的z分量。
    pauli: int, optional
        表示Pauli矩阵的选择，取值为-1、0或1。

    Returns
    -------
    float
        系统的基态能量值。
    
    Examples
    --------
    ed 计算：
    
    >>> import quante as qt
    >>> op = qt.generate.operas
    >>> import numpy as np
    >>> L = 10
    >>> jxx, jyy, jxy, jyx, hz = 1, 0.1, 0.3, 0.4, 0.5
    >>> ham = op.heisenberg_operator(L, j=(jxx,jyy,0), h=(0,0,hz)) + op.sum(jxy*op.xy(i,i+1) + jyx*op.yx(i,i+1) for i in range(L-1))
    >>> basis = qt.generate.basis.spin_basis(L)
    >>> mat = ham.to_matrix(basis, pauli=False)
    >>> np.linalg.eigvalsh(mat)[0]
    -3.270292325887006
    
    用这个函数
    
    >>> import quante as qt
    >>> op = qt.generate.operas
    >>> L = 10
    >>> jxx, jyy, jxy, jyx, hz = 1, 0.1, 0.3, 0.4, 0.5
    >>> qt.solvable_models.free_fermion.XY_gdenergy(L, jxx, jyy, jxy, jyx, hz, pauli=0)
    -3.270292325887006
    """
    # 检查L是否为无穷大，如果是则计算无穷大系统的能量
    if _np.isinf(L):
        return _XY_gdenergy_inf(jxx=jxx, jyy=jyy, jxy=jxy, jyx=jyx, hz=hz, pauli=pauli)
    else:
        # 对于有限尺寸系统，计算能量谱omega并返回其总和的负值
        omega = _XY_omega(L, jxx=jxx, jyy=jyy, jxy=jxy, jyx=jyx, hz=hz, pauli=pauli)
        return -_np.sum(omega)


def _XY_gdenergy_inf(
    jxx: float = 1.0, 
    jyy: float = 1.0, 
    jxy: float = 0.0, 
    jyx: float = 0.0, 
    hz: float = 0.0, 
    pauli: int = 0
    ):
    """计算无穷大尺寸下系统的基态能量"""
    assert _np.all([_np.isscalar(i) for i in [jxx, jyy, jxy, jyx, hz]])
    
    if pauli == -1 or pauli == 1:
        jxx, jyy, jxy, jyx, hz = jxx * 4, jyy * 4, jxy * 4, jyx * 4, hz * 2

    λ = (jxx + jyy) / 4 + 1j * (jyx - jxy) / 4
    γ = (jxx - jyy) / 4 + 1j * (jyx + jxy) / 4
    h = hz / 2
    
    if _np.isclose(λ, 0) and _np.isclose(h, 0):
        return -2 * _np.abs(γ) / _np.pi
    if _np.isclose(γ, 0) and _np.isclose(h, 0):
        return -2 * _np.abs(λ) / _np.pi
    if _np.isclose(h, 0):
        try:
            return (_np.abs(λ) / _np.pi * _sp.special.ellipe(1 - γ**2 / λ**2) + 
                    _np.abs(γ) / _np.pi * _sp.special.ellipe(1 - λ**2 / γ**2))
        except Exception:
            pass

    return _sp.integrate.quad(
        lambda x: -_np.sqrt(
            (h - _np.real(λ) * _np.cos(2 * _np.pi * x)) ** 2 +
            (_np.abs(γ) * _np.sin(2 * _np.pi * x)) ** 2
        ),0,1,)[0]
    
    

def XY_energies(
    L: int,
    jxx: Union[float, _np.ndarray] = 1.0, 
    jyy: Union[float, _np.ndarray] = 1.0, 
    jxy: Union[float, _np.ndarray] = 0.0, 
    jyx: Union[float, _np.ndarray] = 0.0, 
    hz: Union[float, _np.ndarray] = 0.0, 
    pauli: int = 0
) -> _np.ndarray:
    r"""计算 **开边界条件** 下的 xy 模型的 **能谱**

    .. math:: 
        H = \sum_{i = 0}^{L - 2} 
                \left(
                j^{xx}_i s^{x}_{i} s^{x}_{i + 1} + 
                j^{yy}_i s^{y}_{i} s^{y}_{i + 1} + 
                j^{xy}_i s^{x}_{i} s^{y}_{i + 1} + 
                j^{yx}_i s^{y}_{i} s^{x}_{i + 1}
                \right)
            + \sum_{i = 0}^{L - 1} h^{z}_i s^{z}_{i}

    Parameters
    ----------
    L : int
        系统的大小。
    jxx : Union[float, _np.ndarray], optional
        xx 方向的交换常数, by default 1.0
    jyy : Union[float, _np.ndarray], optional
        yy 方向的交换常数, by default 1.0
    jxy : Union[float, _np.ndarray], optional
        xy 方向的交叉耦合常数, by default 0.0
    jyx : Union[float, _np.ndarray], optional
        yx 方向的交叉耦合常数, by default 0.0
    hz : Union[float, _np.ndarray], optional
        z 方向的外磁场, by default 0.0
    pauli : int, optional
        表示 Pauli 矩阵的选择，取值为 -1、0 或 1, by default 0

    Returns
    -------
    float
        系统的能谱。
        
    Examples
    --------
    ed 方法计算能谱
    
    >>> import quante as qt
    >>> op = qt.generate.operas
    >>> import numpy as np
    >>> L = 10
    >>> jxx, jyy, jxy, jyx, hz = 1, 0.1, 0.3, 0.4, 0.5
    >>> ham = op.heisenberg_operator(L, j=(jxx,jyy,0), h=(0,0,hz)) + op.sum(jxy*op.xy(i,i+1) + jyx*op.yx(i,i+1) for i in range(L-1))
    >>> basis = qt.generate.basis.spin_basis(L)
    >>> mat = ham.to_matrix(basis, pauli=False)
    >>> np.linalg.eigvalsh(mat)
    array([-3.27029233, -3.21818576, -3.03602612, ...,  3.03602612,  3.21818576,  3.27029233])
    
    用这个函数：
    
    >>> import quante as qt
    >>> L = 10
    >>> jxx, jyy, jxy, jyx, hz = 1, 0.1, 0.3, 0.4, 0.5
    >>> qt.solvable_models.free_fermion.XY_energies(L, jxx, jyy, jxy, jyx, hz, pauli=0)
    array([-3.27029233, -3.21818576, -3.03602612, ...,  3.03602612,  3.21818576,  3.27029233])
    
    也可以是下面这种情形：
    
    >>> import numpy as np
    >>> import quante as qt
    >>> op = qt.generate.operas
    >>> L = 10
    >>> jxx, jyy, jxy, jyx, hz = np.random.rand(L-1), np.random.rand(L-1), np.random.rand(L-1), np.random.rand(L-1), np.random.rand(L)
    >>> ham = op.sum(
    >>>     jxx[i] * op.xx(i,i+1) 
    >>>     + jyy[i] * op.yy(i,i+1) 
    >>>     + jxy[i] * op.xy(i,i+1) 
    >>>     + jyx[i] * op.yx(i,i+1) for i in range(L-1)) + op.sum( hz[i] * op.z(i) for i in range(L))
    >>> basis = qt.generate.basis.spin_basis(L)
    >>> mat = ham.to_matrix(basis, pauli=False)
    >>> print(np.linalg.eigvalsh(mat))
    >>> print(qt.solvable_models.free_fermion.XY_energies(L, jxx, jyy, jxy, jyx, hz, pauli=0))
    [-3.62108382 -3.61534953 -3.53208427 ...  3.53208427  3.61534953  3.62108382]
    [-3.62108382 -3.61534953 -3.53208427 ...  3.53208427  3.61534953  3.62108382]
    
    todo： 如何返回本征态？
    """
    from .spectrum_numba import _get_full_sprem
    omega = _XY_omega(L, jxx=jxx, jyy=jyy, jxy=jxy, jyx=jyx, hz=hz, pauli=pauli)
    gdeng = -_np.sum(omega)
    return _np.sort(_get_full_sprem(gdeng, omega, L))


def _XY_omega(
    L: int,
    jxx: Union[float, _np.ndarray] = 1.0, 
    jyy: Union[float, _np.ndarray] = 1.0, 
    jxy: Union[float, _np.ndarray] = 0.0, 
    jyx: Union[float, _np.ndarray] = 0.0, 
    hz: Union[float, _np.ndarray] = 0.0, 
    pauli: int = 0
) -> _np.ndarray:
    """
    计算有限尺寸系统的激发谱。
    """
    jxxlist = _to_array(jxx, L-1)
    jyylist = _to_array(jyy, L-1)
    jxylist = _to_array(jxy, L-1)
    jyxlist = _to_array(jyx, L-1)
    hzlist = _to_array(hz, L)

    # 如果 pauli 为 -1 或 1，放大所有耦合常数和磁场强度
    if pauli == -1 or pauli == 1:
        jxxlist, jyylist, jxylist, jyxlist, hzlist = jxxlist * 4, jyylist * 4, jxylist * 4, jyxlist * 4, hzlist * 2

    # 检查是否满足简化计算激发谱的条件
    if _np.isclose(jxylist, 0.0).all() and _np.isclose(jyxlist, 0.0).all() and _np.isclose(jxxlist, jyylist).all() and _np.isclose(jxxlist, jxxlist[0]).all():
        λ = jxxlist[0] / 2
        h = hzlist / 2
        # 计算激发谱，使用离散正弦变换解
        omega = _np.abs([λ * _np.cos(_np.pi / (L + 1) * (k + 1)) - h for k in range(L)])
    else:
        # 一般情况下，通过哈密顿量矩阵计算
        H = _get_spin_BdG_mat(
            L, jxx=jxxlist, jyy=jyylist, jxy=jxylist, jyx=jyxlist, hz=hzlist, reduce=True
        )
        eigvalue = eigvalsh(H)  # 计算哈密顿量矩阵的特征值
        if len(eigvalue) == L:
            omega = _np.sqrt(_np.abs(eigvalue))  # 如果特征值数量等于系统大小
        else:
            omega = eigvalue[L:] # 提取正的特征值部分
    return _np.array(omega)


def _to_array(param: Union[float, _np.ndarray], size: int) -> _np.ndarray:
    """
    将标量参数转换为数组，以便统一处理
    """
    if _np.isscalar(param):
        return _np.full(size, param)
    assert len(param) == size, "Array length must match system size L or L-1."
    return _np.array(param)


def _get_spin_BdG_mat(
    L: int,
    jxx: _np.ndarray,
    jyy: _np.ndarray,
    hz: _np.ndarray,
    jxy: _np.ndarray,
    jyx: _np.ndarray,
    reduce: bool = False
) -> _np.ndarray:
    # 计算 BdG 哈密顿量的参数
    lbdalist = (jxx + jyy) / 8 + 1j / 8 * (jyx - jxy)
    gammalist = (jxx - jyy) / 8 + 1j / 8 * (jyx + jxy)
    hlist = hz / 2
    
    # 标记是否所有参数均为实数
    is_real_lbdalist = _np.isreal(lbdalist).all()
    is_real_gammalist = _np.isreal(gammalist).all()

    if is_real_lbdalist:
        lbdalist = _np.real(lbdalist)
    if is_real_gammalist:
        gammalist = _np.real(gammalist)
    is_real = is_real_lbdalist and is_real_gammalist

    # 构造 H22 矩阵（哈密顿量的对角块）
    dtype = _np.float64 if is_real else _np.complex128
    H22 = _np.zeros(shape=(L, L), dtype=dtype)
    H22 += - _np.diag(hlist)
    H22 += _np.diag(lbdalist, k=1)
    H22 += _np.diag(lbdalist.conj(), k=-1)
    
    # 构造 H21 矩阵（哈密顿量的对角块）
    dtype = _np.float64 if is_real else _np.complex128
    H21 = _np.zeros(shape=(L, L), dtype=dtype)
    H21 += _np.diag(gammalist, k=1)
    H21 += - _np.diag(gammalist, k=-1)
    
    # 如果所有参数均为实数且 reduce 为 True，则返回化简后的结果
    if is_real and reduce:
        return (H21 - H22) @ (H21 + H22)
    else:
        # 返回完整的 BdG 哈密顿量矩阵
        return _np.block([[-_np.conj(H22), -_np.conj(H21)],
                          [           H21,            H22]])

def XY_free_energy(
    L: Optional[float] = _np.inf,  # 默认为无穷大尺寸的系统
    jxx: Union[float, _np.ndarray] = 1.0, 
    jyy: Union[float, _np.ndarray] = 1.0, 
    jxy: Union[float, _np.ndarray] = 0.0, 
    jyx: Union[float, _np.ndarray] = 0.0, 
    hz: Union[float, _np.ndarray] = 0.0, 
    beta: float = 1.0,
    pauli: int = 0
) -> float:
    from .spectrum_numba import _logcosh
    if _np.isinf(L):
        assert _np.all([_np.isscalar(i) for i in [jxx, jyy, jxy, jyx, hz]])
        
        if pauli == -1 or pauli == 1:
            jxx, jyy, jxy, jyx, hz = jxx * 4, jyy * 4, jxy * 4, jyx * 4, hz * 2
        from scipy.integrate import quad

        lamb = (jxx + jyy) / 4 + 1j * (jyx - jxy) / 4
        gamma = (jxx - jyy) / 4 + 1j * (jyx + jxy) / 4
        h = hz / 2
        omega = lambda k: _np.sqrt(
            (h - _np.real(lamb) * _np.cos(k)) ** 2 + (_np.abs(gamma) * _np.sin(k)) ** 2
        ) + _np.imag(lamb) * _np.sin(k)
        integral = quad(
            lambda k: _logcosh(beta * omega(k)),
            -_np.pi,
            _np.pi,
        )[0]
        return -_np.log(2) / beta - integral / 2 / _np.pi / beta
    omega = _XY_omega(L, jxx=jxx, jyy=jyy, jxy=jxy, jyx=jyx, hz=hz, pauli=pauli)
    return -_np.log(2) / beta - _np.sum(_logcosh(beta * omega)) / beta / L


def XY_internal_energy(
    L: Optional[float] = _np.inf,  # 默认为无穷大尺寸的系统
    jxx: Union[float, _np.ndarray] = 1.0, 
    jyy: Union[float, _np.ndarray] = 1.0, 
    jxy: Union[float, _np.ndarray] = 0.0, 
    jyx: Union[float, _np.ndarray] = 0.0, 
    hz: Union[float, _np.ndarray] = 0.0, 
    beta: float = 1.0,
    pauli: int = 0
) -> float:
    """有限尺寸为开边界结果"""
    if _np.isinf(L):
        if pauli == -1 or pauli == 1:
            jxx, jyy, jxy, jyx, hz = jxx * 4, jyy * 4, jxy * 4, jyx * 4, hz * 2
        from scipy.integrate import quad

        lamb = (jxx + jyy) / 4 + 1j * (jyx - jxy) / 4
        gamma = (jxx - jyy) / 4 + 1j * (jyx + jxy) / 4
        h = hz / 2
        omega = lambda k: _np.sqrt(
            (h - _np.real(lamb) * _np.cos(k)) ** 2 + (_np.abs(gamma) * _np.sin(k)) ** 2
        ) + _np.imag(lamb) * _np.sin(k)
        integral = quad(
            lambda k: omega(k) * _np.tanh(beta * omega(k)),
            -_np.pi,
            _np.pi,
        )[0]
        return -integral / 2 / _np.pi
    omega = _XY_omega(L, jxx=jxx, jyy=jyy, jxy=jxy, jyx=jyx, hz=hz, pauli=pauli)
    return -_np.sum(_np.tanh(beta * omega)) / L


def XY_specific_heat(
    L: Optional[float] = _np.inf,  # 默认为无穷大尺寸的系统
    jxx: Union[float, _np.ndarray] = 1.0, 
    jyy: Union[float, _np.ndarray] = 1.0, 
    jxy: Union[float, _np.ndarray] = 0.0, 
    jyx: Union[float, _np.ndarray] = 0.0, 
    hz: Union[float, _np.ndarray] = 0.0, 
    beta: float = 1.0,
    pauli: int = 0
) -> float:
    """有限尺寸为开边界结果"""
    if _np.isinf(L):
        if pauli == -1 or pauli == 1:
            jxx, jyy, jxy, jyx, hz = jxx * 4, jyy * 4, jxy * 4, jyx * 4, hz * 2
        from scipy.integrate import quad

        lamb = (jxx + jyy) / 4 + 1j * (jyx - jxy) / 4
        gamma = (jxx - jyy) / 4 + 1j * (jyx + jxy) / 4
        h = hz / 2
        omega = lambda k: _np.sqrt(
            (h - _np.real(lamb) * _np.cos(k)) ** 2 + (_np.abs(gamma) * _np.sin(k)) ** 2
        ) + _np.imag(lamb) * _np.sin(k)
        integral = quad(
            lambda k: omega(k) ** 2 * (1 - _np.tanh(beta * omega(k)) ** 2),
            -_np.pi,
            _np.pi,
        )[0]
        return -(beta**2) * integral / 2 / _np.pi
    omega = _XY_omega(L, jxx=jxx, jyy=jyy, jxy=jxy, jyx=jyx, hz=hz, pauli=pauli)
    return -(beta**2) / L * _np.sum(omega**2 * (1 - _np.tanh(beta * omega) ** 2))


def XY_magnetization(
    L: Optional[float] = _np.inf,  # 默认为无穷大尺寸的系统
    jxx: Union[float, _np.ndarray] = 1.0, 
    jyy: Union[float, _np.ndarray] = 1.0, 
    jxy: Union[float, _np.ndarray] = 0.0, 
    jyx: Union[float, _np.ndarray] = 0.0, 
    hz: Union[float, _np.ndarray] = 0.0, 
    beta: float = 1.0,
    pauli: int = 0
) -> float:
    """有限尺寸为开边界结果"""
    if _np.isinf(L):
        if pauli == -1 or pauli == 1:
            jxx, jyy, jxy, jyx, hz = jxx * 4, jyy * 4, jxy * 4, jyx * 4, hz * 2
        from scipy.integrate import quad

        lamb = (jxx + jyy) / 4 + 1j * (jyx - jxy) / 4
        gamma = (jxx - jyy) / 4 + 1j * (jyx + jxy) / 4
        h = hz / 2
        omega = lambda k: _np.sqrt(
            (h - _np.real(lamb) * _np.cos(k)) ** 2 + (_np.abs(gamma) * _np.sin(k)) ** 2
        ) + _np.imag(lamb) * _np.sin(k)
        domega_dhz = lambda k: (h - _np.real(lamb) * _np.cos(k)) / omega(k) / 2
        integral = quad(
            lambda k: domega_dhz(k) * _np.tanh(beta * omega(k)),
            -_np.pi,
            _np.pi,
        )[0]
        return integral / 2 / _np.pi
    # 只有开边界 λ (xx + yy) 可以用离散正弦变换严格解
    if _np.isclose(jxy, 0.0) and _np.isclose(jyx, 0.0) and _np.isclose(jxx, jyy):
        if pauli == -1 or pauli == 1:
            jxx, jyy, jxy, jyx, hz = jxx * 4, jyy * 4, jxy * 4, jyx * 4, hz * 2
        λ = (jxx + jyy) / 4
        h = hz / 2
        omega = _np.abs([λ * _np.cos(_np.pi / (L + 1) * (k + 1)) - h for k in range(L)])
        return -_np.sum(_np.tanh(beta * omega) * beta) / beta / L / 2
    else:
        omega1 = _XY_omega(
            L, jxx=jxx, jyy=jyy, jxy=jxy, jyx=jyx, hz=hz + 0.00005, pauli=pauli
        )
        omega2 = _XY_omega(
            L, jxx=jxx, jyy=jyy, jxy=jxy, jyx=jyx, hz=hz - 0.00005, pauli=pauli
        )
        domega_dhz = (omega1 - omega2) / 0.0001
        return -_np.sum(_np.tanh(beta * omega1) * beta * domega_dhz) / beta / L
    

#todo: how to return eigenstates?
# def ising_pblock_spectrum(L, j, h, pauli):
#     jxxlist = _to_array(j, L-1)
#     hzlist = _to_array(h, L)

#     jyylist = _to_array(0., L-1)
#     jxylist = _to_array(0., L-1)
#     jyxlist = _to_array(0., L-1)

#     if pauli == -1 or pauli == 1:
#         jxxlist, jyylist, jxylist, jyxlist, hzlist = jxxlist * 4, jyylist * 4, jxylist * 4, jyxlist * 4, hzlist * 2

#     H = _get_spin_BdG_mat(
#         L, jxx=jxxlist, jyy=jyylist, jxy=jxylist, jyx=jyxlist, hz=hzlist, reduce=False
#     )
#     val, vec = _np.linalg.eigh(H)
#     assert (_np.diff(val) > 1e-9).all(), "degeneracy detected"

#     from ...operas import fermion as op
#     from ...basis import spin_basis
#     basis = spin_basis(L)
#     res = []
#     for i in range(L):
#         f = op.sum(vec[p,i] * op.fdag(p) for p in range(L)) + op.sum(vec[p+L,i] * op.f(p) for p in range(L))
#         f = f.jw_transfer()
#         res.append(f.to_matrix(basis))




