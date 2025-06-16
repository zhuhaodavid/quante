# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:38:18
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 19:17:07

import numpy as _np
import math
import warnings as _warnings
from typing import Literal

__all__ = [
    "plot_energy_density",
    "plot_energy_hist",
    "spectral_form_factor",
    "mean_level_spacing",
    "unfolding_diff",
    "plot_level_spacing_distribution",
    "plot_level_spacings_ratio",
    "Ginibre_distribution",
    "Poisson_distribution",
    "WignerDyson_distribution",
]

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
        from .nbfuc.level_statistics_nb import _spectral_form_factor_single
        return _spectral_form_factor_single(engs, times)
    
    if isinstance(times, list):
        times = _np.array(times)
    assert _np.isrealobj(engs), "engs must be real"
    from .nbfuc.level_statistics_nb import _spectral_form_factor
    return _spectral_form_factor(engs, times)


def mean_level_spacing(val,verbose=True):
    """mean level spacing as a measure of level repulsion.
    
    Parameters
    ----------
    val : list or ndarray
        energy spectrum or single eigenvalue spectrum.
    verbose : bool, optional
        if True, print warnings for degeneracies, by default True.

    Notes
    -----
    typical values for the energy level spacing in different ensembles 
    (Possion and the threefold ensembles) are
    .. code-block:: text
        +----------+----------+-------------+-------------+-------------+
        |  class   |  Possion |   GUE/A     |   GOE/AI    |   GSE/AII   |
        +----------+----------+-------------+-------------+-------------+
        | symmetry |    -     |   None      | Time-re(+1) | Time-re(-1) |
        +----------+----------+-------------+-------------+-------------+
        |  energy  |  0.38629 |  0.5996(1)  |  0.5307(1)  |  0.6744(1)  |
        +----------+----------+-------------+-------------+-------------+
        | sv(herm) |  0.38629 |  0.422245   |  0.423589   |  0.411438   |
        +----------+----------+-------------+-------------+-------------+
        | sv(non-h)|  0.38629 |  0.6026     |  0.5358     |  0.6761     |
        +----------+----------+-------------+-------------+-------------+
       
    Returns
    -------
    float
        The mean level spacing, or NaN if degeneracies are found.
    
    Example
    -------
    >>> mat = np.random.randn(1000, 1000)
    >>> mat += mat.conj().T  # make it Hermitian
    >>> eng = np.linalg.eigvalsh(mat)
    >>> qt.measure.mean_level_spacing(eng) 
    np.float64(0.5321503482669373)

    References
    ----------

    [1]. https://doi.org/10.1103/PhysRevLett.110.084101

    [2]. https://doi.org/10.1103/PRXQuantum.4.040312 
	"""
    val = _np.sort(val)  # ensure the input is sorted

	# check for degeneracies    
    if len(_np.unique(val)) != len(val):
        if verbose:
            _warnings.warn("Degeneracies found in spectrum 'E'!")
        return _np.nan
    else:
	    # compute consecutive E-differences
        sn = _np.diff(val)
		
		# calculate the ratios of consecutive spacings
        aux = _np.zeros((len(val)-1,2),dtype=_np.float64)

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
    unfolding:bool = True, 
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
    unfolding : bool, optional
        If True, the energy spectrum will be unfolded before plotting, by default True.
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
    if unfolding:
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

def plot_level_spacings_ratio(val, ax, bins=None):
    """ r 的分布 """
    s_list = _np.diff(val)
    r_list = []
    for i in range(len(s_list)-1):
        if s_list[i] < 1e-10 or s_list[i+1] < 1e-10:
            r_list.append(0)
        else:
            r = min(s_list[i]/s_list[i+1], s_list[i+1]/s_list[i])
            r_list.append(r)
    if bins is None:
        # 这是对高斯分布最优的选择，其它分布也应当保证 N**(-1/5)
        h = 1.05 * _np.std(r_list) * len(r_list)**(-1/5)
        bins = _np.arange(_np.min(r_list), _np.max(r_list)+h, h)
    else:
        bins = _np.linspace(0, 1+0.1, bins)
    ax.hist(r_list, bins=bins, density=True, color='lightgray', ec="gray") # type: ignore
    # 设置横纵坐标的名称以及对应字体格式
    return ax