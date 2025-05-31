# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-08-23 14:26:26
# @Last Modified by:   hzhu
# @Last Modified time: 2025-05-31 16:58:53

#!! 不要在这里引用 quante 中的其他函数（可以在函数中引用）

import numpy as _np
import math
import warnings as _warnings
from scipy.sparse import issparse, dia_array, dia_matrix
from typing import Literal

__all__ = [
    "spectral_form_factor",
    "entanglement_spectrum",
    "entanglement_entropy",
    "entropy",
    "entropy_page",
    "cg_coef",
    "unfolding",
    "mean_level_spacing",
]

__all__ += [
    "plot_energy_density", 
    "plot_energy_hist", 
    "plot_level_spacing_distribution", 
]

__all__ += [
    "Ginibre_distribution",
    "Poisson_distribution",
    "WignerDyson_distribution",
]

def spectral_form_factor(engs:_np.ndarray, times:_np.ndarray | float):
    """谱形因子.
    
    计算谱形因子，谱形因子是一个描述能级统计性质的指标，它是能级分布的傅里叶变换。

    Parameters
    ----------
    engs : _np.ndarray
        能级矩阵，可以包含多个系统的能级，将计算平均值
    times : _np.ndarray
        时间
    
    Returns
    -------
    _np.ndarray
        谱形因子
    """
    if engs.ndim == 1:
        engs = engs.reshape(1, -1)
    
    try:
        import torch as tc
        from tqdm import tqdm
        assert tc.cuda.is_available()
        ts = (times.to('cuda') if isinstance(times, tc.Tensor) 
              else tc.tensor(times, device='cuda'))
        mat = (engs.to('cuda') if isinstance(engs, tc.Tensor)
               else tc.tensor(engs, device='cuda'))
        sff = tc.zeros(len(ts), device='cuda', dtype=tc.float64)
        for j in tqdm(range(len(ts)), ascii=True):
            sff[j] = tc.mean(tc.abs(tc.sum(tc.exp(1j*mat*ts[j]), dim=1))**2)
        return sff.cpu().numpy()
    #     sff = tc.zeros(len(ts), engs.shape[0], device='cuda', dtype=tc.float64)
    #     for j in tqdm(range(len(ts)), ascii=True):
    #         sff[j] = tc.abs(tc.sum(tc.exp(1j*mat*ts[j]), dim=1))**2
    #     return sff
    except Exception as e:
        pass

    if isinstance(times, float):
        from .linalg.usenumba.operations_numba import _spectral_form_factor_single
        return _spectral_form_factor_single(engs, times)
    
    if isinstance(times, list):
        times = _np.array(times)
    assert _np.isrealobj(engs), "engs must be real"
    from .linalg.usenumba.operations_numba import _spectral_form_factor
    return _spectral_form_factor(engs, times)


def entanglement_spectrum(
    state: _np.ndarray, 
    left_number: int, 
    basis = None
) -> _np.ndarray:
    """The entanglement spectrum of a pure state.
    
    Parameters
    ----------
    state : ndarray
        The pure state, can be 1D or 2D array.
        - 1D array: the single state
        - 2D array: the multiple states with shape `(dim, num_states)`
    left_number : int
        The number of spins on the left side of the bipartition.
    basis : SpinBasis, optional
        The basis of the state, by default None.

    Returns
    -------
    ndarray | float
        The entanglement spectrum of the state.
        
    Examples
    --------
    >>> L = 10
    >>> ham = qt.generate.operas.heisenberg_operator(L=10)
    >>> basis = qt.generate.basis.spin_basis(L=L, Nup=5, kblock=1)
    >>> hammat = ham.to_matrix(basis)
    >>> val, vec = qt.linalg.eigh(hammat, k=1)
    >>> print(qt.quantity.entanglement_spectrum(vec, L=L, left_number=L//2, basis=basis))
    [0.70710678 0.70710678 0.         0.         0.         0.         0.         0.        ]
    """
    if basis is not None:
        if state.ndim == 1:
            state = state.reshape(-1,1)
        state = basis.recover(state)
        L = basis.L
    else:
        D = state.shape[0] if state.ndim == 1 else state.shape[1]
        L = int(math.log2(D))
        assert D == 1 << L, "The dimension of the state is not 2^L"
    matrix = state.T.reshape(-1,1<<left_number,1<<L-left_number)
    return _np.linalg.svd(matrix, compute_uv=False) # type: ignore

def entanglement_entropy(
    states: _np.ndarray, 
    left_number: int, 
    basis = None
) -> _np.ndarray | float:
    """The entanglement entropy of pure states.
    
    Parameters
    ----------
    states : ndarray
        The pure states, can be 1D or 2D array.
        - 1D array: the single state
        - 2D array: the multiple states with shape `(dim, num_states)`
    L : int
        The number of spins.
    left_number : int
        The number of spins on the left side of the bipartition.
    basis : SpinBasis, optional
        The basis of the state, by default None.

    Returns
    -------
    ndarray | float
        The entanglement entropy of the states.
        
    Examples
    --------
    >>> L = 10
    >>> ham = qt.generate.operas.heisenberg_operator(L=10)
    >>> basis = qt.generate.basis.spin_basis(L=L, Nup=5, kblock=1)
    >>> hammat = ham.to_matrix(basis)
    >>> val, vec = qt.linalg.eigh(hammat, k=1)
    >>> print(qt.quantity.entanglement_entropy(vec, L=L, left_number=L//2, basis=basis))
    0.6931471805599453
    """
    ee = entanglement_spectrum(states, left_number, basis)
    # ee.shape = (items, spectrum)
    ee = _np.where(ee > 0, ee, 1)  # Replace zeros with 1 to make log(1)=0
    res = (-2) * _np.sum(ee**2 * _np.log(ee), axis=1)
    if res.size == 1:
        return res[0]
    return res

def entropy(a, rank=None, base=_np.e) -> _np.float64:
    """计算 von Neumann 熵.
    
    如果 `a` 是密度矩阵，那么计算：
    
    .. math::
        -\\operatorname{tr} a \\log a
        
    通过 `rank` 可以指定计算的本征值个数。
    
    如果 `a` 是本征值，那么计算：
    
    .. math::
        -\\sum_{i=1}^n a_i \\log a_i
    
    其中 :math:`n` 是 `a` 的维度。

    Examples
    --------
    >>> L = 6
    >>> mat = ed.rdmat_rho(2**L, sparse=True, density=0.5)
    >>> etp = qla.entropy(mat)
    >>> print(etp)
    >>> 
    >>> vals = qla.eigvals(mat)
    >>> etp = qla.entropy(vals)
    >>> print(etp)
    
    可以直接输入本征值
    """
    if _np.ndim(a) == 1:
        evals = a
    else:
        from .linalg.eig_modified import eigvals
        if rank is None:
            evals = eigvals(a)
        else:  # know that not all eigenvalues needed
            evals = eigvals(a, k=rank, which="LM")

    evals = evals[evals > 0.0]
    return _np.real_if_close([_np.sum(-evals * _np.log2(evals)) / _np.log2(base)])[0]


def entropy_page(Dim_sub:int, Dim_tot:int) -> float:
    """计算 Page 熵。
    
    Page 熵指 Hilbert 空间中一个随机态的熵。
    
    Parameters
    ----------
    Dim_sub : int
        子空间维数
    Dim_tot : int
        空间维数

    Returns
    -------
    float
        Page 熵
    
    Examples
    --------
    计算 12 个自旋，二分为 6 个自旋的 Page 熵：
    
    >>> import quante.quantity as qq
    >>> L = 12
    >>> vals = qq.entropy_page(2**(L//2), 2**L)
    >>> vals
    3.6590254932605575
    
    随机这样一个态，计算它的二分纠缠熵：
    
    >>> import quante as qt
    >>> vec = qt.generate.state.random(dim=2**12)
    >>> rho = qt.linalg.partial_trace(vec, [2]*L, range(L//2))
    >>> qt.quantity.entropy(rho)
    3.6520327465312925
    
    可以看到是非常接近的。
    
    References
    ----------
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.71.1291
    """
    # * Ensure m is the smaller subsystem
    if Dim_sub <= Dim_tot//Dim_sub:
        m, n = Dim_sub, Dim_tot//Dim_sub
    else:
        m, n = Dim_tot//Dim_sub, Dim_sub
        
    s = 0.
    for k in range(n+1, Dim_tot+1):
        s += 1 / k
    return s - (m-1)/(2*n)



def cg_coef(j1:float, j2:float, j3:float, m1:float, m2:float, m3:float) -> float:
    """Clebsch-Gordon coefficient 系数.
    
    (j1,m1) 与 (j2,m2) 耦合成为 (j3,m3) 的系数
    
    Parameters
    ----------
    j1 : float
        1 的总角动量
    j2 : float
        2 的总角动量
    j3 : float
        3 的总角动量
    m1 : float
        1 角动量的 z 分量
    m2 : float
        2 角动量的 z 分量
    m3 : float
        3 角动量的 z 分量

    Returns
    -------
    cg_coeff : float
        相应的 cg 系数
    
    Examples
    --------
    计算 [1] 中第一个数: (1/2,1/2) + (1/2,1/2) => (1,1)
    
    >>> import quante.quantity as qq
    >>> qq.cg_coef(1/2, 1/2, 1, 1/2, 1/2, 1)
    1.0
    
    References
    ----------
    [1]. https://github.com/BrandonHenke/phy803/blob/main/clebrpp.pdf
    """
    from .linalg.usenumba.operations_numba import clebsch
    return clebsch(j1, j2, j3, m1, m2, m3)


def mean_level_spacing(E,verbose=True):
    """Clebsch-Gordon coefficient 系数.
    
    三种系综的值如下：

    .. code-block:: text
        +----------+-------------+-------------+-------------+
        |  Possion |     GOE     |     GUE     |     GSE     |
        +----------+-------------+-------------+-------------+
        |  0.38629 |  0.5307(1)  |  0.5996(1)  |  0.6744(1)  |
        +----------+-------------+-------------+-------------+
    
    Parameters
    ----------
    E : list or _np.ndarray
        能级列表，必须是升序且没有重复的能级
    verbose : bool, optional
        是否打印警告信息, by default True

    Returns
    -------
    float
        平均能级间距
    
    Examples
    --------
    计算 xxz 模型的平均能级间距：
    
    >>> import quante as qt
    >>> import numpy as np
    >>> ham = qt.generate.operas.heisenberg_operator(L=12, j=(1,1,0.5))
    >>> basis = qt.generate.basis.spin_basis(L=12, Nup=6, zblock=1)
    >>> mat = ham.to_matrix(basis)
    >>> engs = np.linalg.eigvalsh(mat)
    >>> qt.quantity.mean_level_spacing(engs)
    0.3693119203415571
    
    References
    ----------
    https://arxiv.org/pdf/1212.5611.pdf
	"""
    if not isinstance(E,_np.ndarray):
        E = _np.asarray(E)

    if _np.any(_np.sort(E)!=E):
        raise TypeError("Expecting a sorted list of ascending, nondegenerate eigenenergies 'E'.")

	# check for degeneracies    
    if len(_np.unique(E)) != len(E):
        if verbose:
            _warnings.warn("Degeneracies found in spectrum 'E'!")
        return _np.nan
    else:
	    # compute consecutive E-differences
        sn = _np.diff(E)
		
		# calculate the ratios of consecutive spacings
        aux = _np.zeros((len(E)-1,2),dtype=_np.float64)

        aux[:,0] = sn
        aux[:,1] = _np.roll(sn,-1)

        return _np.mean(_np.divide( aux.min(1), aux.max(1) )[0:-1] )

def unfolding_diff(eng, discard=0.2, polynomial_of_degree=15, n=30):
    """unfolding energy spectrum.

    Parameters
    ----------
    val : ndarray
        the energy spectrum
    discard : float, optional
        percentage of the spectrum to discard from both ends, by default 0.2
    polynomial_of_degree : int, optional
        (for real spectrum) degree of the polynomial for unfolding, by default 15
    n : int, optional
        (for complex spectrum) number of neighbors to consider, by default 30

    Returns
    -------
    ndarray
        unfolded energy spectrum
    """
    if _np.iscomplexobj(eng):
        # this is a complex spectrum
        # the algorithm is based on the paper: 
        # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.108.043301
        from scipy.spatial import cKDTree
        eigval_2d = _np.column_stack((eng.real, eng.imag))
        tree = cKDTree(eigval_2d)
        dists, _ = tree.query(eigval_2d, k=n+1)  # k=31 to include the point itself as the 0th neighbor
        rhobar = n/(_np.pi*dists[:, n]**2)
        return dists[:, 1] * rhobar**0.5
    else:
        # cdf
        E_list, NE_list = [], []  # unit step function Theta: less or equal
        for i in range(len(eng)-1):
            E_list.append(eng[i])
            NE_list.append(i)
            E_list.append(eng[i])
            NE_list.append(i+1)
        # unfolding
        Fit = _np.polynomial.Polynomial.fit(
            E_list, NE_list, polynomial_of_degree)  # polynomial fitting - degree 15
        # discard the spectrum located at the edges
        val_discard = eng[int(len(eng)*discard):-int(len(eng)*discard)]
        eps = Fit(_np.array(val_discard))  # unfolded energy
        # mean level density = mean level spacing = 1
        assert (eps[-1]-eps[0])/(len(eps)-1)-1. < 1.e-2
        return _np.diff(eps)

def plot_level_spacing_distribution(
    eng: _np.ndarray, 
    ax = None, 
    bins = None, 
    unfolded:bool = False, 
    st_dists: Literal['poisson', 'poisson-c',
            'wigner-dyson-1', 'wigner-dyson-2','wigner-dyson-4', 
            'ginibre'] | None = None
):
    """
    Plot the level spacing distribution of energy spectrum.
    
    Parameters
    ----------
    eng : _np.ndarray
        The energy spectrum, can be real or complex.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on, by default None, which creates a new figure.
    bins : int or None, optional
        The number of bins for the histogram, by default None, which calculates the optimal number of bins.
    unfolded : bool, optional
        Whether to unfold the energy spectrum, by default False.
        If True, the `unfolding_diff` function will be applied to the energy spectrum.
    st_dists : str or list of str, optional
        The standard distributions to plot, by default None.
        Options are:
        - 'poisson' or 'poisson-c': Poisson distribution (real or complex)
        - 'wigner-dyson-1', 'wigner-dyson-2', 'wigner-dyson-4': Wigner-Dyson distributions
        - 'ginibre': Ginibre distribution for complex eigenvalues

    Example
    -------
    >>> eng = np.linalg.eigvalsh(hammat)
    >>> plot_level_spacing_distribution(eng)
    """
    # ensure the eng is unfolded
    if not unfolded:
        eng = unfolding_diff(eng)
    assert _np.isrealobj(eng), "eng must be real, unfolding_diff is needed for complex spectrum"

    # set plot axes
    if ax is None:
        import matplotlib.pyplot as _plt
        _, ax = _plt.subplots()
    
    # determine bins
    eps_spc = _np.array(eng)/_np.mean(eng)
    if bins is None:
        h = 1.05*_np.std(eps_spc) * eps_spc.size**(-1/5)
        bins = _np.arange(eps_spc.min(), eps_spc.max()+h, h)
    else:
        bins = _np.linspace(0, 4+0.1, bins)
    
    # plot the histogram
    ax.hist(eps_spc, bins=bins, density=True, color='lightgray', ec="gray") # type: ignore

    # plot the standard distributions
    if st_dists is not None:
        if isinstance(st_dists, str):
            st_dists = [st_dists]
        xs = _np.linspace(0, bins[-1], 1000)
        for dist in st_dists or []:
            if dist.lower() in ["poisson-r", "poisson"]:
                ys = Poisson_distribution(xs)
                ax.plot(xs, ys, label="Poisson", color="red")
            elif dist.lower() == "poisson-c":
                ys = Poisson_distribution(xs, complex_plane=True)
                ax.plot(xs, ys, label="Poisson", color="red")
            elif dist.lower() == "wigner-dyson-1":
                ys = WignerDyson_distribution(xs, beta=1)
                ax.plot(xs, ys, label="Wigner-Dyson-1", color="blue")
            elif dist.lower() == "wigner-dyson-2":
                ys = WignerDyson_distribution(xs, beta=2)
                ax.plot(xs, ys, label="Wigner-Dyson-2", color="blue")
            elif dist.lower() == "wigner-dyson-4":
                ys = WignerDyson_distribution(xs, beta=4)
                ax.plot(xs, ys, label="Wigner-Dyson-4", color="blue")
            elif dist.lower() == "ginibre":
                ys = Ginibre_distribution(xs)
                ax.plot(xs, ys, label="Ginibre", color="blue")
            else:
                raise ValueError(f"Unknown distribution: {dist}")
    return ax

def Poisson_distribution(s, complex_plane=False):
    """
    Poisson distribution for real eigenvalues.
    Vectorized for numpy arrays.
    """
    if complex_plane:
        return _np.pi * s / 2 * _np.exp(-_np.pi * s**2 / 4)
    else:
        # 1D Poisson distribution
        return _np.exp(-s)

def WignerDyson_distribution(s, beta=1):
    """
    Wigner-Dyson distribution for real eigenvalues.
    Vectorized for numpy arrays.
    """
    if beta == 1:
        return _np.pi * s / 2 * _np.exp(-_np.pi * s**2 / 4)
    elif beta == 2:
        return 32 / _np.pi**2 * s**2 * _np.exp(-4 * s**2 / _np.pi)
    elif beta == 4:
        return 2**18 / _np.pi**3/3**6 * s**4 * _np.exp(-64 * s**2 / _np.pi/9)
    else:
        raise NotImplementedError

def _e(m, s):
    # s: array or scalar
    s = _np.asarray(s)
    res = _np.zeros_like(s, dtype=float)
    for l in range(m+1):
        res += s**(2*l) / math.factorial(l)
    return res

def _p(s, M):
    # s: array or scalar
    s = _np.asarray(s)
    term1 = _np.ones_like(s, dtype=float)
    for m in range(1, M):
        term1 *= _e(m, s) * _np.exp(-s**2)
    term2 = _np.zeros_like(s, dtype=float)
    for m in range(1, M):
        term2 += (2 * s**(2*m+1)) / (math.factorial(m) * _e(m, s))
    return term1 * term2

def Ginibre_distribution(s, M=100):
    """
    Ginibre distribution for complex eigenvalues.
    Vectorized for numpy arrays.
    """
    c = 1.1429
    return c * _p(c * s, M)

def plot_energy_density(vals, ax, bandwidth=1.):
    """利用 KDE 方法快速画出连续的能级密度分布曲线，调节 bandwidth 会改变结果
    """
    import matplotlib.pyplot as _plt
    from scipy.stats import norm
    color1 = "C0"  # 统计图的颜色
    color2 = "|C3"  # 能级的线和颜色
    x_d = _np.linspace(vals.min(), vals.max(), 500)
    density = sum(norm(xi, bandwidth).pdf(x_d) for xi in vals) / len(vals) # type: ignore
    ax.fill_between(x_d, density, color=color1, alpha=0.8)
    ax.plot(vals, _np.full_like(vals, -0.003), color2, markeredgewidth=0.1, markersize=10)
    return ax

def plot_energy_hist(vals, ax, bins=None):
    import matplotlib.pyplot as _plt
    color2 = "|C5"  # 能级的线和颜色
    if bins is None:
        # 这是对高斯分布最优的选择，其它分布也应当保证 N**(-1/5)
        h = 1.05*_np.std(vals) * vals.size**(-1/5)
        bins = _np.arange(vals.min(), vals.max()+h, h)
    ax.hist(vals, bins=bins, density=True, color="0.7")
    ax.plot(vals, _np.full_like(vals, -0.0035), color2, markeredgewidth=0.1, markersize=10)
    from scipy.stats import norm
    x_d = _np.linspace(vals.min(), vals.max(), 100)
    y_norm = [norm(_np.mean(vals), _np.std(vals)).pdf(xi) for xi in x_d] # type: ignore
    ax.plot(x_d, y_norm, "k--")
    return ax



