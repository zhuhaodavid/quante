# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:15:34
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-30 18:11:15

import numpy as _np
from typing import Literal

__all__ = [
    "plt_style_use",
    "DynamicPlot",
]


def plt_style_use(stylename:Literal['quante', 'default', 'science'] = "quante", svg: bool = True) -> None:
    """设置 pyplot 风格样式。
    
    Parameters
    ----------
    stylename: str, optional
        风格样式,常用的有 "quante", "default" 和 "science". 默认为 "quante"
    svg: bool, optional
        是否使用 SVG 格式. 默认为 True.
    
    References
    ----------
    https://matplotlib.org/stable/tutorials/introductory/customizing.html
    https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html#sphx-glr-gallery-style-sheets-style-sheets-reference-py
    """
    import matplotlib.pyplot as _plt
    
    try:
        if svg:
            from IPython.display import set_matplotlib_formats
            set_matplotlib_formats("svg")
        else:
            from IPython.display import set_matplotlib_formats
            set_matplotlib_formats("png")
    except:
        pass
    
    if stylename == "quante":
        try:
            import matplotlib as _mpl
            _mpl.font_manager.findfont("Times New Roman", fallback_to_default=False) # type: ignore
            font = 'Times New Roman'
        except:
            font = 'sans-serif'
        defaultconfig = {
            "pdf.fonttype": 42,
            "figure.dpi": 100,
            "font.size": 12,
            "axes.labelsize": 14,
            "mathtext.fontset": "stix",
            "font.family": font,  # 'sans-serif', "Times New Roman"
            # 'dejavusans','dejavuserif', 'cm', 'stix','stixsans' or 'custom'
            "font.serif": ["SimSun"],
            # "figure.autolayout": True,
            "xtick.direction": "in",  # x tick 方向
            "ytick.direction": "in",  # y tick 方向
            # grid
            "axes.grid": "False",
            "grid.alpha": 0.4,  # 透明度
            "grid.linewidth": 1.0,  # 粗细
            # "svg.image_inline": True
            "legend.frameon":       False,
            "legend.fontsize":      13,
            "savefig.bbox" : "tight",
            "text.usetex" : False,
        }
        _plt.style.use("default")
        _plt.rcParams.update(defaultconfig)
    elif stylename == "science":
        # from https://github.com/garrettj403/SciencePlots/blob/master/scienceplots/styles/science.mplstyle
        scienceconfig = {
            # Set default figure size
            # "figure.figsize" : (3.5, 2.625),
            "figure.figsize" : (4, 2.9),
            
            # Set x axis
            "xtick.direction": "in",
            # "xtick.major.size" : 3,
            # "xtick.major.width" : 0.5,
            # "xtick.minor.size" : 1.5,
            # "xtick.minor.width" : 0.5,
            # "xtick.minor.visible" : True,
            "xtick.top" : False,

            # Set y axis
            "ytick.direction" : "in",
            # "ytick.major.size" : 3,
            # "ytick.major.width" : 0.5,
            # "ytick.minor.size" : 1.5,
            # "ytick.minor.width" : 0.5,
            # "ytick.minor.visible" : True,
            "ytick.right" : False,
            
            # Set line widths
            "axes.linewidth" : 0.5,
            "grid.linewidth" : 0.5,
            "lines.linewidth" : 1.,

            # Remove legend frame
            "legend.frameon" : False,

            # Always save as 'tight'
            "savefig.bbox" : "tight",
            "savefig.pad_inches" : 0.05,

            # Use serif fonts
            # font.serif : Times
            "font.family" : "serif",
            "mathtext.fontset" : "dejavuserif",

            # Use LaTeX for math formatting
            "text.usetex" : True,
            "text.latex.preamble" : "\\usepackage{amsmath} \\usepackage{amssymb}"
        }
        _plt.style.use("default")
        _plt.rcParams.update(scienceconfig)
    else:
        _plt.style.use(stylename)

class DynamicPlot:
    def __init__(self, tlist, ax, *args, **kwargs):
        """Initialize a dynamic plot object.

        This class is designed to create a dynamic plot that updates as new data is appended.
        It supports different types of plots (line, parametric, and density) based on the data provided.

        Parameters
        ----------
        tlist : list or array-like
            A list of time points or x-coordinates for the plot.
        ax : matplotlib.axes.Axes, optional
            A matplotlib Axes object where the plot will be drawn. If None, a new figure and axes will be created.
        *args : tuple
            Additional positional arguments to pass to the plot function.
        **kwargs : dict
            Additional keyword arguments to pass to the plot function.
        """
        # save package
        import matplotlib.pyplot as plt
        self.pkg = plt

        # check ax
        if ax is None:
            fig, ax = plt.subplots()
        if not isinstance(ax, plt.Axes):
            raise TypeError("The 'ax' parameter must be a matplotlib Axes object or None.")
        self.ax = ax

        # check if in ipython
        in_ipython = False
        try:
            from IPython import get_ipython
            in_ipython = get_ipython() is not None
        except ImportError:
            in_ipython = False
        if in_ipython:
            from IPython.display import clear_output, display
            self.clear_output = clear_output
            self.display = display
        self.in_ipython = in_ipython
        
        # save data
        self.tlist = tlist
        self.data = None

        # save plot parameters
        self.ptype = None
        self.args = args
        self.kwargs = kwargs

        # initialize dp parameters
        self.i = 0
        self.xlim = None
        self.ylim = None
        self.clim = None
        self.legend = None
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()
        
    def set(self, 
            xlim=None, 
            ylim=None, 
            clim=None, 
            legend=None, 
            ptype=None
        ):
        """Set the parameters for the dynamic plot.

        This method allows you to set the limits for the x and y axes, color limits for density plots,
        whether to display a legend, and the type of plot to be used.

        Parameters
        ----------
        xlim : tuple, optional
            A tuple specifying the limits for the x-axis, by default None
        ylim : tuple, optional
            A tuple specifying the limits for the y-axis, by default None
        clim : tuple, optional
            A tuple specifying the color limits for density plots, by default None
        legend : bool, optional
            Whether to display a legend on the plot, by default None
        ptype : Literal['line', 'para', 'dens'], optional
            The type of plot to be used. Can be 'line' for line plots, 'para' for parametric plots, or 'dens' for density plots.

        Returns
        -------
        self : DynamicPlot
            Returns the instance of the DynamicPlot class with updated parameters.
        """
        self.xlim = xlim
        self.ylim = ylim
        self.clim = clim
        self.legend = legend
        self.ptype = ptype
        return self

    def append(self, res_t):
        """Append new data to the plot and update the display.

        This method updates the plot with new data at each time step. It initializes the plot if it hasn't been done yet.
        The type of plot (line, parametric, or density) is determined based on the shape of the data provided.

        Parameters
        ----------
        res_t : list or array-like
            The new data to append to the plot. It can be 
            - a single value (for line plots), 
            - a pair of values (for parametric plots),
            - a list of values (for density plots).

        Returns
        -------
        data : numpy.ndarray
            The updated data array after appending the new data.
        """
        ax = self.ax
        plt = self.pkg
        i = self.i

        if self.data is None:
            res_t = self._init_plot(res_t)
            
        if self.ptype == "line":
            self.data[i] = res_t    
            self.plot.set_xdata(self.tlist[:i+1])
            self.plot.set_ydata(self.data[:i+1])
            if self.xlim is None:
                ax.set_xlim(min(self.tlist[:i+1]), max(self.tlist[:i+1]))
            if self.ylim is None:
                ax.set_ylim(min(self.data[:i+1]), max(self.data[:i+1]))
        elif self.ptype == "para":
            self.data[0, i] = res_t[0]
            self.data[1, i] = res_t[1]
            self.plot.set_xdata(self.data[0,:i+1])
            self.plot.set_ydata(self.data[1,:i+1])
            if self.xlim is None:
                ax.set_xlim(min(self.data[0,:i+1]), max(self.data[0,:i+1]))
            if self.ylim is None:
                ax.set_ylim(min(self.data[1,:i+1]), max(self.data[1,:i+1]))
        elif self.ptype == "dens":
            self.data[:, i] = _np.asarray(res_t)
            self.plot.set_data(self.data.T)
            if self.clim is None:
                valid = self.data[:, :i+1]
                vmin, vmax = _np.nanmin(valid), _np.nanmax(valid)
                if vmin != vmax:
                    self.plot.set_clim(vmin, vmax)           
        
        if self.in_ipython:
            self.clear_output(wait=True)
            self.display(plt.gcf())
        else:
            plt.pause(0.1)
        self.i += 1
        
        if i == len(self.tlist) - 1:
            if self.in_ipython:
                self.clear_output(wait=True)
            else:
                self.pkg.show()
        return self.data


    def _init_plot(self, res_t):
        ax = self.ax
        plt = self.pkg

        res_t = _np.asarray(res_t)

        if self.ptype is None:
            # determine plot type according to the data type
            if res_t.size == 1:
                self.ptype = "line"
            elif res_t.size == 2:
                self.ptype = "para"
            else:
                self.ptype = "dens"
        
        if self.ptype == "line":
            if self.xlim is None:
                self.xlim = (self.tlist[0], self.tlist[-1])
            self.data = _np.full(len(self.tlist), _np.nan, dtype=_np.float64)
            self.plot, = ax.plot(self.tlist, self.data, *self.args, **self.kwargs)
            if self.legend:
                ax.legend()
        elif self.ptype == "para":
            self.data = _np.full((2, len(self.tlist)), _np.nan, dtype=_np.float64)
            self.plot, = ax.plot(self.data[0,:], self.data[1,:], *self.args, **self.kwargs)
            if self.legend:
                ax.legend()
        elif self.ptype == "dens":
            n = len(res_t)
            self.data = _np.full((n, len(self.tlist)), _np.nan, dtype=_np.float64)
            self.plot = ax.imshow(self.data.T, *self.args, aspect='auto', origin='lower', **self.kwargs, extent=(0, n, self.tlist[0], self.tlist[-1]))
            if self.legend:
                plt.colorbar(self.plot, ax=ax)
            if self.clim is not None:
                self.plot.set_clim(*self.clim)           
        else:
            raise ValueError("Unknown plot type.")
        
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
            
        return res_t

