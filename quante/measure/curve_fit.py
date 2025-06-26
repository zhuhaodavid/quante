# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-17 10:13:29
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-26 17:39:12

import numpy as _np

def plot_gaussian(mean, std, ax=None, xrange=None, **kwargs):
    """plot a Gaussian distribution

    Parameters
    ----------
    mean : float
        the mean of the Gaussian distribution
    std : float
        the standard deviation of the Gaussian distribution
    ax : axes, optional
        the axes to plot on, by default None
    xrange : list, optional
        the range of x values to plot, by default None (mean ± 4 * std)
    **kwargs : dict, optional
        additional keyword arguments for the plot function
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from quante.measure.curve_fit import plot_gaussian
    >>> mean = 0
    >>> std = 1
    >>> fig, ax = plt.subplots()
    >>> plot_gaussian(mean, std, ax=ax, color='blue', linestyle='-', linewidth=2)
    >>> plt.show()
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
    if xrange is None:
        xrange = [mean - 4 * std, mean + 4 * std]
    xs = _np.linspace(*xrange, 100)
    from scipy.stats import norm
    ys = norm.pdf(xs, loc=mean, scale=std)
    ax.plot(xs, ys, **kwargs)


def plot_hist(top, bins, ax=None, **kwargs):
    """Plot a histogram with the top values connected by lines

    The `top` and `bins` parameters should be the output of a histogram function, 
    such as `numpy.histogram`.

    Parameters
    ----------
    top : numpy.ndarray
        The top values of the histogram to be connected
    bins : numpy.ndarray
        The bin edges of the histogram
    ax : matplotlib.axes.Axes, optional
        The axes to plot on, by default None (creates a new figure and axes)
    **kwargs : dict, optional
        Additional keyword arguments for the plot function
    
    Examples
    --------
    >>> import numpy as np
    >>> from quante.measure.curve_fit import plot_hist
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.normal(0, 1, 1000)
    >>> hist, bins = np.histogram(data, bins=30, density=True)
    >>> fig, ax = plt.subplots()
    >>> plot_hist(hist, bins, ax=ax, color='blue', linestyle='-', linewidth=2)
    >>> plt.show()
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
    x = _np.zeros(4 * len(bins) - 3)
    y = _np.zeros(4 * len(bins) - 3)
    x[0:2*len(bins)-1:2], x[1:2*len(bins)-1:2] = bins, bins[:-1]
    x[2*len(bins)-1:] = x[1:2*len(bins)-1][::-1]
    y[1:2*len(bins)-1:2] = y[2:2*len(bins):2] = top
    y[0] = y[-1]
    split = 2 * len(bins)
    ax.plot(x[:split], y[:split], **kwargs)


def hist_gaussian(data, ax=None, bins=None, **kwargs):
    """Plot a histogram of the data and overlay a Gaussian distribution

    Parameters
    ----------
    data : numpy.ndarray
        The data to plot the histogram for
    ax : matplotlib.axes.Axes, optional
        The axes to plot on, by default None (creates a new figure and axes)
    bins : list or int, optional
        The bins for the histogram, by default None (uses Freedman-Diaconis rule)
    **kwargs : dict, optional
        Additional keyword arguments for the histogram function

    Returns
    -------
    tuple
        A tuple containing the histogram values, bin edges, mean, and standard deviation of the data
        This can be used in plot_gaussian and plot_hist functions.
    
    Examples
    --------
    >>> import numpy as np
    >>> from quante.measure.curve_fit import hist_gaussian
    >>> data = np.random.normal(0, 1, 1000)
    >>> hist, bin_edges, mean, std = hist_gaussian(data)
    >>> print(f"Mean: {mean}, Std: {std}")
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
    
    if bins is None:
        h = 1.05*_np.std(data) * data.size**(-1/5)
        bins = _np.arange(data.min(), data.max()+h, h)
    
    hist, bin_edges, _ = ax.hist(data, bins=bins, density=True, histtype='step', **kwargs)
    mean, std = _np.mean(data), _np.std(data)
    plot_gaussian(mean, std, ax=ax, color='k', linestyle='--', linewidth=1.5)  
    return hist, bin_edges, mean, std


def fit(xs: list, ys: list, polynomial_degree: int=1) -> _np.polynomial.Polynomial:
    """fit the data with a polynomial curve

    Parameters
    ----------
    xs : list
        x coordinates of the data points
    ys : list
        y coordinates of the data points
    polynomial_degree : int
        degree of the polynomial to fit, default is 1 (linear fit)
   
    Returns
    -------
    _np.polynomial.Polynomial
        A polynomial object representing the fitted curve. 
        The coefficients can be accessed via the `coef` method.

        ..math::
            f(x) = c_0 + c_1 * x + c_2 * x^2 + ... + c_n * x^n
        
   
    Examples
    --------
    >>> import numpy as np
    >>> from quante.measure.curve_fit import fit
    >>> import matplotlib.pyplot as plt
    >>> xs = np.linspace(1, 10, 10)
    >>> ys = 3 * xs + 7 + 0.1 * np.random.normal(0, 1, xs.shape)
    >>> f = fit(xs, ys)
    >>> print(f.c)
    [6.94646096 3.01413511]
    >>> nxs = np.linspace(1, 10, 100)
    >>> nys = f(nxs)
    >>> plt.plot(xs, ys, 'o', label='data')
    >>> plt.plot(nxs, nys, label='fit')
    """
    Fit = _np.polynomial.Polynomial.fit(xs, ys, polynomial_degree).convert()
    return Fit


def interp(x, y, x0, kind="linear"):
    """插值
    Args:
        x (list): x
        y (list): y
        x0 (list): x0
        kind (string, optional): 插值类型。Defaults to 'linear'.
    Returns:
        list: f(x0)
    """
    from scipy.interpolate import interp1d

    return interp1d(x, y, kind=kind, bounds_error=False)(x0)


def find_boundary(x, y, zdata, a, clf=None, axes=None):
    """找到 (x, y, z) 图中的二分类边界，一边 z < a

    默认使用 Gassian 过程分类：clf = GaussianProcessClassifier(1.0 * RBF(1.0))

    Parameters
    ----------
    x : numpy.ndarray
        二维数组，x 方向数据
    y : numpy.ndarray
        二维数组，y 方向数据
    zdata : numpy.ndarray
        二维数组，z 方向数据
    a : real
        分界线
    clf : classifier, optional
        分类器。Defaults to None.
    axes : list, optional
        [x0, x1, y0, y1]. Defaults to None.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        边界的横纵坐标
    
    
    常用的分类器还有：
    
    - 支持向量机线性分类
        clf = sklearn.svm.SVC(kernel="linear", C=0.025)
    
    - 支持向量机分类
        clf = sklearn.svm.SVC(gamma=2, C=1)
    
    - 决策树分类
        clf = sklearn.tree.DecisionTreeClassifier(max_depth=5)
    
    - MLPC 分类
        clf = sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000)
    
    - 高斯朴素贝叶斯分类
        clf = sklearn.naive_bayes.GaussianNB()
    
    - 随机森林分类
        clf = sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    
    - AdaBoost 分类
        clf = sklearn.ensemble.AdaBoostClassifier()
    
    - 二次判别分析算法
        clf = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()

    """
    import matplotlib.pyplot as _plt
    from sklearn.metrics import accuracy_score

    assert _np.all([isinstance(i, _np.ndarray) for i in [x, y, zdata]])
    assert x.ndim == 1 and y.ndim == 1 and zdata.ndim == 2
    assert zdata.shape == (y.size, x.size)
    pts = _np.array([[i, j] for i in x for j in y])
    vls = _np.array([zdata[j, i] for i in range(len(x)) for j in range(len(y))]) < a
    if clf is None:
        from sklearn.svm import SVC

        clf = SVC(gamma="scale", C=10)
    clf.fit(pts, vls)
    pre_y = clf.predict(pts)
    accuracy = accuracy_score(vls, pre_y)
    print(f"classify accuracy: {accuracy:0.2f}")
    if axes is None:
        axes = [x.min(), x.max(), y.min(), y.max()]
    x0s = _np.linspace(axes[0], axes[1], 2000)
    x1s = _np.linspace(axes[2], axes[3], 2000)
    x0, x1 = _np.meshgrid(x0s, x1s)
    X = _np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X)
    y_pred = y_pred.reshape(x0.shape)
    fig = _plt.figure("temp")
    cs = fig.add_subplot(111).contour(x0, x1, y_pred, levels=[0.5])
    _plt.close("temp")
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    return v[:, 0], v[:, 1]

