# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-08-23 14:26:26
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 18:51:30

#!! 不要在这里引用 quante 中的其他函数（可以在函数中引用）

import numpy as _np
import warnings as _warnings


__all__ = [
    "entropy",
    "entropy_page",
    "entanglement_spectrum",
    "cg_coef",
    "mean_level_spacing",
]

__all__ += [
    "plot_energy_density", 
    "plot_energy_hist", 
    "plot_level_spacing_distribution", 
    ]

def entanglement_spectrum(state:_np.ndarray, L:int, left_number:int, basis=None) -> _np.float64:
    """计算纯态纠缠熵
    basis 表示 state 所处的对称性空间
    
    示例:
    --------
    >>> L = 10
    >>> ham = qt.generate.operas.heisenberg_operator(L=10)
    >>> basis = qt.generate.basis.spin_basis(L=L, Nup=5, kblock=1)
    >>> hammat = ham.to_matrix(basis)
    >>> val, vec = qt.linalg.eigh(hammat, k=1, which="SA")
    >>> print(qt.quantity.entanglement_spectrum(vec, L=L, left_number=L//2, basis=basis))
    [0.70710678 0.70710678 0.         0.         0.         0.         0.         0.        ]
    """
    if basis is not None:
        fullstate = basis.recover(state.reshape(-1,1))
    else:
        fullstate = state.reshape(-1,1)
    assert fullstate.shape[0] == 1<<L
    matrix = fullstate.reshape(1<<left_number, -1)
    from quante.linalg.svd_robust import svd
    return svd(matrix, compute_uv=False)


def entropy(a, rank=None, base=_np.e) -> _np.float64:
    """
    计算 von Neumann 熵 (- tr a log a) ，rank 表示使用 eigs

    示例:
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
    return _np.real_if_close(_np.sum(-evals * _np.log2(evals)) / _np.log2(base))


def entropy_page(Dim_sub:int, Dim_tot:int) -> _np.float64:
    """
    Calculate the page entropy, i.e. 
       expected entropy for a subsytem of a random state in Hilbert space.
    
    示例:
    --------
    >>> L = 12
    >>> vals = qla.entropy_page(2**(L//2), 2**L)
    >>> print(vals)
    >>> vec = qla.rdket(2**L)
    >>> rho = qla.partial_trace(vec, [2]*L, range(L//2))
    >>> print(qla.entropy(rho))
    
    来自文献：10.1103/PhysRevLett.71.1291
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



def cg_coef(j1:int, j2:int, j3:int, m1:int, m2:int, m3:int) -> float:
    """Calculates the Clebsch-Gordon coefficient
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).
    
    参考：https://github.com/BrandonHenke/phy803/blob/main/clebrpp.pdf

    Parameters
    ----------
    j1 : float
        Total angular momentum 1.

    j2 : float
        Total angular momentum 2.

    j3 : float
        Total angular momentum 3.

    m1 : float
        z-component of angular momentum 1.

    m2 : float
        z-component of angular momentum 2.

    m3 : float
        z-component of angular momentum 3.

    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.

    """
    from .linalg.usenumba.numba_settings import clebsch
    return clebsch(j1, j2, j3, m1, m2, m3)


def mean_level_spacing(E,verbose=True):
	r"""Calculates the mean-level spacing of an energy spectrum.

	See mean level spacing, :math:`\\langle\\tilde r_\mathrm{W}\\rangle`, in 
	`arXiv:1212.5611 <https://arxiv.org/pdf/1212.5611.pdf>`_ for more details.

	For Wigner-Dyson statistics, we have :math:`\\langle\\tilde r_\mathrm{W}\\rangle\\approx 0.53`, while
	for Poisson statistics: :math:`\\langle\\tilde r_\mathrm{W}\\rangle\\approx 0.38`.

	Examples
	--------

	The following example shows how to calculate the mean level spacing :math:`\\langle\\tilde r_\mathrm{W}\\rangle` for the
	spectrum of the ergodic Hamiltonian :math:`H_1=\\sum_jJ S^z_{j+1}S^z + hS^x_j + g S^z_j`.

	Parameters
	-----------
	E : numpy.ndarray
			Ordered list of ascending, NONdegenerate energies. If `E` contains a repeating value, the function returns `nan`.
	verbose : bool, optional
		Toggles warning message about degeneracies of the spectrum `E`.

	Returns
	-------- 
	float
		mean-level spacing.
	nan
		if spectrum `E` has degeneracies.

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


#############################
# 下面是换能级分布的函数
#############################

def discard_len(val, discard):
    assert _np.linalg.norm(val - _np.sort(val)) < 1.0e-10
    L = len(val)
    discard = discard / 2
    return val[int(L * discard) : -int(L * discard)]


def staircase_function(val):
    assert _np.linalg.norm(val - _np.sort(val)) < 1.0e-10
    E_list, NE_list = [], []
    for i in range(len(val) - 1):
        E_list.append(val[i])
        NE_list.append(i)
        E_list.append(val[i])
        NE_list.append(i + 1)
    return E_list, NE_list


def smooth_function(E_list, NE_list, polynomial_degree):
    return _np.polynomial.Polynomial.fit(E_list, NE_list, polynomial_degree)


def get_new_level(val, discard, polynomial_degree):
    val = discard_len(val, discard=discard)
    E_list, NE_list = staircase_function(val)
    Nsm = smooth_function(E_list, NE_list, polynomial_degree)
    return Nsm(val)


def indicator_eta(eps, bin):
    # mean level density = mean level spacing = 1
    assert (eps[-1] - eps[0]) / (len(eps) - 1) - 1.0 < 1.0e-2
    eps_spc = _np.diff(eps)
    """Interable-Chaos transition"""
    Ps, s_order = _np.histogram(
        eps_spc, bins=list(_np.linspace(0, 4, bin + 1)) + [eps.max()], density=True
    )
    s0 = 0.4729
    s0_lab = [
        i for i, j in enumerate(s_order) if abs(j - s0) < 1.0e-3
    ]  # find the lab of s0 in s_order
    assert len(s0_lab) != 0  # enlarging the bins
    s0_lab = s0_lab[0] + 1  # 横坐标截断 s0_lab 的个数，需要 +1.
    diff_s_s0 = _np.diff(s_order[:s0_lab])
    Ps_s0 = Ps[: s0_lab - 1]
    integral_Ps = _np.sum(Ps_s0 * diff_s_s0)  # 求矩阵面积，Ps_s0*d(s_s0)

    def Pp_s(s):
        return _np.exp(-s)

    def PWD_s(s):
        return _np.pi * s / 2 * _np.exp(-_np.pi * s**2 / 4)

    from scipy import integrate

    integral_Pp_s, _ = integrate.quad(Pp_s, 0, s0)  # 分母减数
    integral_PWD_s, _ = integrate.quad(PWD_s, 0, s0)  # 分母被减数
    eta = (integral_Ps - integral_PWD_s) / (integral_Pp_s - integral_PWD_s)
    return eta


def plot_energy_density(vals, ax=None, bandwidth=1.):
    """利用 KDE 方法快速画出连续的能级密度分布曲线，调节 bandwidth 会改变结果
    """
    import matplotlib.pyplot as _plt
    try:
        vals = vals.cpu().real
    except:
        pass
    from scipy.stats import norm
    tag = False
    color1 = "C0"  # 统计图的颜色
    color2 = "|C3"  # 能级的线和颜色
    if ax is None:
        ax = _plt.subplot(111)
        tag = True
    x_d = _np.linspace(vals.min(), vals.max(), 500)
    density = sum(norm(xi, bandwidth).pdf(x_d) for xi in vals) / len(vals)
    ax.fill_between(x_d, density, color=color1, alpha=0.8)
    ax.plot(vals, _np.full_like(vals, -0.003), color2, markeredgewidth=0.1, markersize=10)
    
    if tag:
        _plt.show()
    return ax


def plot_energy_hist(vals, ax=None, bins=None):
    import matplotlib.pyplot as _plt
    try:
        vals = (vals.to("cpu")).numpy().real
    except:
        pass
    tag = False
    color1 = "C0"  # 统计图的颜色
    color2 = "|C5"  # 能级的线和颜色
    if ax is None:
        ax = _plt.subplot(111)
        tag = True
    if bins is None:
        # 这是对高斯分布最优的选择，其它分布也应当保证 N**(-1/5)
        h = 1.05*_np.std(vals) * vals.size**(-1/5)
        bins = _np.arange(vals.min(), vals.max()+h, h)
    ax.hist(vals, bins=bins, density=True, color="0.7")
    ax.plot(vals, _np.full_like(vals, -0.0035), color2, markeredgewidth=0.1, markersize=10)
    
    from scipy.stats import norm
    x_d = _np.linspace(vals.min(), vals.max(), 100)
    y_norm = [norm(_np.mean(vals), _np.std(vals)).pdf(xi) for xi in x_d]
    ax.plot(x_d, y_norm, "k--")
    if tag:
        _plt.show()
    return ax

 
def unfolding(val, discard=0.2, polynomial_of_degree=15):
    # cdf
    E_list, NE_list = [], []  # unit step function Theta: less or equal
    for i in range(len(val)-1):
        E_list.append(val[i])
        NE_list.append(i)
        E_list.append(val[i])
        NE_list.append(i+1)
    # unfolding
    Fit = _np.polynomial.Polynomial.fit(
        E_list, NE_list, polynomial_of_degree)  # polynomial fitting - degree 15
    # discard the spectrum located at the edges
    val_discard = val[int(len(val)*discard):-int(len(val)*discard)]
    eps = Fit(_np.array(val_discard))  # unfolded energy
    # mean level density = mean level spacing = 1
    assert (eps[-1]-eps[0])/(len(eps)-1)-1. < 1.e-2
    return eps


def plot_level_spacing_distribution(eps, ax=None, bins=None, beta=1):
    """ 能级间距分布 """
    eps = unfolding(eps)
    spacing = _np.diff(eps)
    _plot_level_spacing_distribution(spacing, ax=ax, bins=bins, beta=beta)

def _plot_level_spacing_distribution(spacing, ax=None, bins=None, beta=1):
    """ 能级间距分布 """
    import matplotlib.pyplot as _plt
    tag = False
    if ax is None:
        ax = _plt.subplot(111)
        tag = True
    eps_spc = _np.array(spacing)/_np.mean(spacing)
    # stats
    if bins is None:
        # 这是对高斯分布最优的选择，其它分布也应当保证 N**(-1/5)
        h = 1.05*_np.std(eps_spc) * eps_spc.size**(-1/5)
        bins = _np.arange(eps_spc.min(), eps_spc.max()+h, h)
    else:
        bins = _np.linspace(0, 4+0.1, bins)
    ax.hist(eps_spc, bins=bins, density=True, color='lightgray', ec="gray")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    # comparing
    sn = _np.linspace(0, 4, 100)
    Ps_poisson = _np.exp(-sn)
    if beta==1:
        Ps_WD = _np.pi * sn / 2 * _np.exp(-_np.pi * sn ** 2 / 4)
    elif beta==2:
        Ps_WD = 32 / _np.pi**2 * sn**2 * _np.exp(- 4 * sn ** 2 / _np.pi)
    elif beta==4:
        Ps_WD = 2**18 / _np.pi**3/3**6 * sn**4 * _np.exp(- 64 * sn ** 2 / _np.pi/9)
    else:
        raise NotImplementedError
    ax.plot(sn, Ps_poisson, color="red", label='poisson')
    ax.plot(sn, Ps_WD, color="blue", label='WD')
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$P$")
    ax.legend()
    if tag:
        _plt.show()
    return ax

def level_spacing_indicator_eta(eps):  # ? # todo
    """ eta 指数 可积：1 不可积：0 """
    eps_spc = _np.diff(eps)
    # stats
    s0 = 0.4729
    eps_spc.sort()
    integral_Ps = _np.count_nonzero(eps_spc < s0) / eps_spc.size
    integral_Pp_s = 0.37680761269947016
    integral_PWD_s = 0.16108178372342252
    eta = (integral_Ps - integral_PWD_s) / (integral_Pp_s - integral_PWD_s)
    return eta


def peak_position(eps, polynomial_of_degree=15):
    """ 峰值位置 可积：0 不可积：0.8 """
    eps_spc = _np.diff(eps)
    eps_spc.sort()
    e_list, Ne_list = [], []  # unit step function Theta: less or equal
    for i in range(len(eps_spc)-1):
        e_list.append(eps_spc[i])
        Ne_list.append(i)
        e_list.append(eps_spc[i])
        Ne_list.append(i+1)
    e_list = _np.array(e_list)
    Ne_list = _np.array(Ne_list)/Ne_list[-1]
    fit = _np.polynomial.Polynomial.fit(
        e_list, Ne_list, polynomial_of_degree).deriv()  # polynomial fitting - degree 15
    rt = fit.deriv().roots()
    rt = rt[(abs(rt.imag) < 1e-10)*(rt.real >= 0)*(rt.real <= 1)].real
    rt = _np.append(rt, [eps_spc[0]])
    Pk = rt[_np.argmax(fit(rt))]
    return Pk


def plot_level_number_variance(eps, l_list, ax=None):
    """ Sigma """
    import matplotlib.pyplot as _plt
    tag = False
    if ax is None:
        ax = _plt.subplot(111)
        tag = True
    Sigma_l = []
    for l in l_list:
        if l == 0:
            Sigma_l.append(0)
            continue
        N_eps, _ = _np.histogram(eps, _np.arange(
            eps[0], eps[-1]+0.1, l), density=False)
        Sigma_l.append(_np.var(N_eps))
    ax.plot(l_list, Sigma_l)
    ax.plot(l_list, l_list, 'b-.')
    y_d = [2*(_np.log(2*_np.pi*l)+_np.euler_gamma+1-_np.pi**2/8) /
           _np.pi**2 if l != 0 else 0 for l in l_list]
    ax.plot(l_list, y_d, 'r--')
    ax.set_ylim([0, 2])
    if tag:
        _plt.show()
    return l_list, ax


def level_spacing_indicator_beta(eps, bandwidth=0.05):
    """ beta 指数 可积：0 不可积：1 """
    # fit eps with poly
    import math

    from scipy import optimize
    from scipy.stats import norm
    eps_spc = _np.diff(eps)
    def b(beta): return math.gamma((beta + 2)/(beta + 1))**(beta + 1)
    def PB(s, beta): return b(beta) * (beta+1) * \
        s**beta * _np.exp(- b(beta) * s**(beta+1))
    s_order = _np.linspace(0, _np.max(eps_spc), 1000)[1:]
    Ps = sum(norm(xi, bandwidth).pdf(s_order)
             for xi in eps_spc)/len(eps_spc)  # 有参数可调
    beta = optimize.curve_fit(PB, s_order, Ps)[0][0]
    return beta


def plot_level_spacings_ratio(val, ax=None, bins=None):
    """ r 的分布 """
    import matplotlib.pyplot as _plt
    tag = False
    if ax is None:
        ax = _plt.subplot(111)
        tag = True
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
    ax.hist(r_list, bins=bins, density=True, color='lightgray', ec="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    """RMT result comparing"""
    r = _np.linspace(0, 1, 100)
    P_poisson = 2 / (r + 1)**2
    Z1 = 8/27
    P_GOE = 1/Z1 * 2 * (r + r**2) / (1 + r + r**2)**(5/2)
    ax.plot(r, P_poisson, color="red", label='poisson')
    ax.plot(r, P_GOE, color="blue", label='WD')
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    ax.set_xlabel(r"$r$", font)
    ax.set_ylabel(r"$P(r)$", font)
    ax.legend(prop={'size': 12}, loc='upper right')
    if tag:
        _plt.show()
    return ax
