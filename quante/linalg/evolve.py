# # -*- coding: utf-8 -*-
# # @Author: hzhu
# # @Date:   2023-10-22 17:13:49
# # @Last Modified by:   hzhu
# # @Last Modified time: 2025-05-16 22:42:06

from scipy import sparse as sps
from scipy.special import jv
from scipy.sparse.linalg import LinearOperator, expm_multiply, spsolve, eigsh, svds
from scipy.integrate import solve_ivp

import numpy as _np
import warnings as _warnings
from typing import Callable, Union, Literal, TYPE_CHECKING
from functools import lru_cache
from typing import Literal
from tqdm import tqdm

if TYPE_CHECKING:  # 类型检查时，导入 torch
    import torch as _tc

__all__ = [
    "evolve_and_measure",
    "Liouvillian"
    "EvolveEngine",
    "expm_multiply",
    "get_time_evolution_states_ED",
    "chebyshev_evolve",
]
 
def expm_multiply(
    mat:Union[_np.ndarray, Callable[[_np.ndarray], _np.ndarray]],
    psi0:_np.ndarray,
    *, 
    ttype:Literal['real-time', 'imag-time']='imag-time', 
    start:None|float=None, 
    stop:None|float=None, 
    num:None|int=None, 
    endpoint:None|bool=None, 
    traceA:None|float=None, 
    herm:bool=False, 
    device:str|None=None
) -> _np.ndarray:
    """calculate `exp(mat) @ psi0` or `exp(- 1j * mat) @ psi0`

    Parameters
    ----------
    mat : Union[_np.ndarray, Callable[[_np.ndarray], _np.ndarray]]
        can be a matrix or a function
        - if mat is a matrix, it should be a numpy.ndarray or scipy.sparse matrix. By 
        setting ttype='real-time', and `herm=True`, the efficiency can be improved.
        - if mat is a function, it should be a function that takes a numpy.ndarray 
        as input and returns a numpy.ndarray. In this case, the right multiplication
        should be provided by the `herm` parameter. If `herm=True`, it means that the
        function is hermitian, then `rmatvec = matvec`. If it is not hermitian, then
        the adjoint operator should be passed in with `herm`, `rmatvec = herm`. In this
        case, the traceA (the trace of matvec) should be passed in, otherwise it will
        be estimated internally, which will affect the accuracy of the result.
    psi0 : _np.ndarray
        the initial state vector
    ttype : Literal['real-time', 'imag-time'], optional
        evolve type, by default 'real-time'
        - `ttype='real-time'`: real-time evolution using `exp(- 1j * H * t)`
        - `ttype='imag-time'`: imaginary-time evolution using `exp(H * t)`
    start : None | float, optional
        the start time, by default None
    stop : None | float, optional
        the stop time, by default None
    num : None | int, optional
        the number of time points, by default None
    endpoint : None | bool, optional
        whether to include the stop time, by default None
    traceA : None | float, optional
        the trace of matvec, by default None
        if mat is a function, this parameter is required
        if mat is a matrix, this parameter is optional
    herm : bool, optional
        whether mat is hermitian, by default False
        - if mat is a function, this parameter is required. If `herm=True`,
        it means that the function is hermitian, then `rmatvec = matvec`.
        If it is not hermitian, then the adjoint operator should be passed in
        with `herm`, `rmatvec = herm`.
        - if mat is a matrix, this parameter is optional
    device : str | None, optional
        the device to use, by default None
        - if device is None, use numpy-CPU
        - if device is 'cuda', use torch-GPU
        - if device is 'cpu', use torch-CPU

    Returns
    -------
    _np.ndarray
        the final state vector after evolution
        the shape of the output array can be 1, 2 or 3.
        - if calculating the action of expm on a single vector at a single time point,
          `ndim` will be 1.
        - if calculating the action of expm on a vector at multiple time points, or
          calculating the action of expm on a matrix at a single time point, `ndim` will be 2.
        - if calculating the action of expm on a matrix with multiple columns at multiple
          time points, `ndim` will be 3.

    Notes
    ------
    When `num` is small, the effect is similar to stepwise iteration.
    When `num` is large, different operators will be used. When `num` takes certain
    specific values `n*s+1` (where `n` is an integer and `s` is an integer depending
    on the matrix modulus), the efficiency will be slightly reduced, and a warning
    will be given, but it will not affect the result. The efficiency can be improved
    by slightly changing `num`.

       
    References
    ----------
    - scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html
    
    - Awad H. Al-Mohy and Nicholas J. Higham (2011)
           "Computing the Action of the Matrix Exponential,
           with an Application to Exponential Integrators."
           SIAM Journal on Scientific Computing,
           33 (2). pp. 488-511. ISSN 1064-8275
           http://eprints.ma.man.ac.uk/1591/

    - Nicholas J. Higham and Awad H. Al-Mohy (2010)
           "Computing Matrix Functions."
           Acta Numerica,
           19. 159-208. ISSN 0962-4929
           http://eprints.ma.man.ac.uk/1451/
           
    Examples
    --------
    >>> import quante as qt
    >>> L = 10
    >>> mat = qt.generate.matrix.heisenberg_matrix(L)
    >>> state = qt.generate.state.random(mat.shape[0], seed=42)
    >>> res = qt.linalg.expm_multiply(mat, state)
    >>> res.shape
    (100, 1024, 1)
    
    For more detailed examples, please refer to the `evolve.ipynb` file in the example folder.
    """
   
    if ttype == 'real-time':
        scale = -1j
    elif ttype == 'imag-time':
        scale = 1.0
    else:
        raise ValueError("ttype should be 'real-time' or 'imag-time'")
    
    if device is not None:
        assert isinstance(mat, (_np.ndarray, sps.spmatrix, sps.sparray)), "cuda only support numpy.ndarray or scipy.sparse matrix"
        
        try:
            import cupy  # 如果有 cupa,直接在 gpu 中计算 trace norm
            traceA, norm1A = traceA, None
            hasshifted = False
        except ImportError:
            # 否则用 scipy 计算 trace norm
            if isinstance(mat, _np.ndarray):
                traceA = _np.trace(mat)
                mat = mat - (_np.trace(mat) / mat.shape[0]) * _np.eye(mat.shape[0])
                norm1A = _np.linalg.norm(mat, ord=1)
            else:
                traceA = mat.trace()
                mat = mat - (mat.trace() / mat.shape[0]) * sps.eye(mat.shape[0], format='csr')
                norm1A = max(abs(mat).sum(axis=0).flat)
            hasshifted = True
            
        
        from ..torch_utils.linalg.sparse import to_csr
        from ..torch_utils.linalg.expm_multiply import expm_multiply
        import torch as tc
        
        dtype = tc.complex128 if scale == -1j or _np.iscomplexobj(mat) or _np.iscomplexobj(psi0) else tc.float64
        
        res = expm_multiply(tc.tensor(mat, device=device) if isinstance(mat, _np.ndarray) else to_csr(mat, device=device), 
                            tc.tensor(psi0, device=device, dtype=dtype),
                            scale, start=start, stop=stop, num=num, endpoint=endpoint, 
                            traceA=traceA, herm=herm, norm1A=norm1A, hasshifted=hasshifted).cpu()
        
        tc.cuda.empty_cache()
        
        return res.numpy()
    
    # 如果 matvec 给的是函数
    if callable(mat):
        # 构造 scipy 的线性算符
        dim = psi0.shape[0]
        dtype = psi0.dtype
        if herm is True:
            if scale == 1.0:
                lo = LinearOperator((dim,dim), matvec=mat, rmatvec=mat, dtype=dtype) # type: ignore
            elif scale == -1j:
                lo = LinearOperator((dim,dim), matvec=lambda v: (-1j) * mat(v), rmatvec=lambda v: (1j) * mat(v), dtype=dtype) # type: ignore
        elif callable(herm):
            assert scale == 1.0
            lo = LinearOperator((dim,dim), matvec=mat, rmatvec=herm, dtype=dtype) # type: ignore
        else:
            raise ValueError("herm should be 1 for hermitian or -1 for antihermitian or callable")
    else:
        assert isinstance(mat, (_np.ndarray, sps.spmatrix, sps.sparray)), "only support numpy.ndarray or scipy.sparse matrix"
        dtype = _np.complex128 if scale == -1j or _np.iscomplexobj(mat) or _np.iscomplexobj(psi0) else _np.float64
        psi0 = psi0.astype(dtype)
        lo = mat
    
    # 主要的工作:
    from .usenumba.expm_multiply_numba import _expm_multiply_numba
    return _expm_multiply_numba(lo, psi0, scale=scale, start=start, stop=stop, num=num, endpoint=endpoint, traceA=traceA)


class EvolveEngine:
    """            
    This class make use of the `expm_multiply` function to evolve the state vector.
    The precalculation will be done in the constructor here, which will speed up the
    calculation of the time evolution.
    """
    def __init__(
        self,
        ham:sps.csr_array|LinearOperator,
        init_state:_np.ndarray,
        ts:_np.ndarray,
        *,
        normalize:bool=False,
        ttype:Literal['real-time', 'imag-time']='real-time',
        traceA:float|None=None,
        dtype:_np.dtype|None=None,
        herm:bool|Callable=False,
        method:Literal['cpu_mul', 'gpu_mul-cuda:0', 
                    'linear_operator','RK45', 'RK23', 'DOP853', 'Radau', 
                    'BDF', 'LSODA']='cpu_mul',
        ivp_kwargs:dict={}
    ):
        """calculate the time evolution of the state vector

        Parameters
        ----------
        ham : sps.csr_array | LinearOperator
            the Hamiltonian matrix
        init_state : _np.ndarray
            the initial state vector
        ts : _np.ndarray
            the time list
        normalize : bool, optional
            if True, normalize the state after each evolution, by default False
        ttype : str, optional
            - `ttype='real-time'`: real-time evolution using `exp(-1j * H * t)`
            - `ttype='imag-time'`: imaginary-time evolution using `exp(H * t)`
        traceA : float | None, optional
            the trace of the matrix, by default None
            if matrix is a LinearOperator and the method is `cpu_mul`, this parameter is required
        dtype : _np.dtype | None, optional
            the data type of the matrix, by default None
            only need to be set when only float is involved
        herm : bool | Callable, optional
            whether the matrix is hermitian, by default False
            - if matrix is a LinearOperator, this parameter is required
            - if matrix is a matrix, this parameter is optional. `herm=True` will accelerate a little bit
        method : str, optional
            - `method='cpu_mul'`: use the CPU method to calculate the time evolution. (when matrix is
            a LinearOperator, the traceA should be passed in)
            - `method='gpu_mul-cuda:0'`: use the GPU method to calculate the time evolution. (LinearOperator
            is not supported in this case)
            - `method='RK45'` ...: use the RK45 ... method to calculate the time evolution, for more
            information, please refer to the `scipy.integrate.solve_ivp` documentation. check,
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        ivp_kwargs : dict, optional
            the parameters for the `scipy.integrate.solve_ivp` function, by default {}
        """
        device = method[8:] if method[:3] == 'gpu' else 'cpu'
        if device == 'cpu':
            # for cpu, the hamiltonian can be a LinearOperator or a csr sparse matrix
            if isinstance(ham, LinearOperator):
                self.csr_mt = ham
            else:
                try:
                    self.csr_mt = ham.tocsr()
                except:
                    self.csr_mt = sps.csr_array(ham)
        
            # the initial state should be a column vector
            dtype = _np.complex128 if dtype is None else dtype
            self.psi = init_state.reshape(-1, 1).astype(dtype)

        else:
            # first move the data to the device
            assert sps.issparse(ham), "ham should be sparse array"
            from ..torch_utils.linalg.sparse import to_csr
            from ..torch_utils.utils import totc
            import torch as tc
            dtype = tc.complex128 if dtype is None else dtype
            self.csr_mt = to_csr(ham.tocsr(), device=device)
            self.psi = totc(init_state, device=device, dtype=dtype).reshape(-1, 1)
            self.tc = tc

        self.tlist = ts
        self.dts = _np.insert(_np.diff(ts), 0, ts[0])
        self.evolved_time = 0
        self.cur_step = 0
        self.normalize = normalize
        self.scale = -1j if ttype == 'real-time' else 1.0
        self.traceA = traceA
        self.device = device
        self.herm = herm
        self.method = method
        self.ivp_kwargs = ivp_kwargs
    
    @lru_cache(maxsize=None)
    def get_evolve_engine(self, dt):
        if self.device == 'cpu':
            from .usenumba.expm_multiply_numba import _evolve_engine
            return _evolve_engine(self.csr_mt, scale=self.scale, t=dt, traceA=self.traceA, herm=self.herm)
        else:
            from ..torch_utils.linalg.expm_multiply import evolve_engine
            return evolve_engine(dt * self.csr_mt, scale=self.scale, herm=self.herm)

    def run(self):
        """
        calculate the time evolution of the state vector
        """
        try:
            dt = self.dts[self.cur_step]
        except:
            _warnings.warn(
                f"t {self.evolved_time} has been reached, dt = {self.dts[-1]} will be used"
            )
            dt = self.dts[-1]
        self.cur_step += 1
        if dt != 0:
            # ============= main =============
            if self.method[4:7] == 'mul':
                ee = self.get_evolve_engine(round(dt,14))
                self.psi = ee(self.psi)
            else:
                if self.scale == 1.:
                    matmul = lambda t, state: self.csr_mt @ state
                else:
                    matmul = lambda t, state: -1j * (self.csr_mt @ state)
                sol = solve_ivp(
                    matmul, (0, dt), self.psi.flatten(), t_eval=[dt],
                    method=self.method, rtol=1e-9, atol=1e-12, **self.ivp_kwargs
                )
                self.psi = sol.y
            # ============= end =============
            if self.normalize:
                if self.device == 'cpu':
                    self.psi /= _np.linalg.norm(self.psi, ord=2)
                else:
                    self.psi /= self.tc.linalg.norm(self.psi, ord=2)
            self.evolved_time += dt
        return self.psi
    
    def pre_obs(self, obs):
        if obs is None:
            return lambda t, state: state
        elif isinstance(obs, list):
            if self.device == 'cpu':
                _obs = lambda t, state: _np.real_if_close([state.conj().reshape(-1) @ (obsmat @ state).reshape(-1) for obsmat in obs])
            else:
                from ..torch_utils.linalg.sparse import to_csr
                import torch as _tc
                obsmatlist = [to_csr(o, device=self.device, dtype=_tc.complex128) for o in obs]
                _obs = lambda t, state: [(state.conj().reshape(1,-1) @ (obsmat @ state).reshape(-1,1)).item() 
                                         for obsmat in obsmatlist]
            return _obs
        elif callable(obs):
            return obs
        else:
            raise ValueError("obs should be a list of sparse matrices or a function")
        
    def measure(self, obs:list[sps.csr_array]|Callable[[float, _np.ndarray], _np.ndarray]):
        """calculate the expectation value of the observable
        
        Parameters
        ----------
        obs : list[sps.csr_array] | Callable[[float, _np.ndarray], _np.ndarray]
            the observable matrix or a function that takes the time and state as input

        Returns
        -------
        _np.ndarray
            the expectation value of the observable
        """
        obs = self.pre_obs(obs)
        res = []
        for t in tqdm(self.tlist, ascii=True):
            state = self.run()
            res.append(obs(t, state))
        try:
            return _np.real_if_close(res)    
        except:
            return res

    def plot_measure(self, obs:list[sps.csr_array]|Callable[[float, _np.ndarray], _np.ndarray], ax=None, **kwargs):
        """dynamic plot the expectation value of the observable

        Parameters
        ----------
        obs : list[sps.csr_array] | Callable[[float, _np.ndarray], _np.ndarray]
            the observable matrix or a function that takes the time and state as input
        ax : matplotlib.axes.Axes, optional
            the axes to plot on, by default None

        Returns
        -------
        _np.ndarray
            the expectation value of the observable
        """
        import matplotlib.pyplot as plt
        obs = self.pre_obs(obs)
        res = None

        # 判断是否在 IPython 环境
        in_ipython = False
        try:
            from IPython import get_ipython
            in_ipython = get_ipython() is not None
        except ImportError:
            in_ipython = False

        if in_ipython:
            from IPython.display import clear_output, display
        
        if ax is None:
            fig, ax = plt.subplots()
        
        for i, t in enumerate(self.tlist):
            state = self.run()
            res_t = _np.real_if_close(obs(t, state))
            if res is None:
                n = len(res_t)
                res = _np.full((n, len(self.tlist)), _np.nan, dtype=_np.float64)
                if n == 1:
                    line, = ax.plot(self.tlist, res.reshape(-1), **kwargs)
                    ax.set_xlim(self.tlist[0], self.tlist[-1])
                else:
                    img = ax.imshow(res.T, aspect='auto', origin='lower', **kwargs, extent=(0, n, self.tlist[0], self.tlist[-1]))
                    plt.colorbar(img, ax=ax)
            res[:, i] = res_t
            if n == 1:
                line.set_ydata(res.reshape(-1))
                # 自动调整 y 轴范围
                valid = res[0, :i+1]
                if valid.size > 0:
                    ymin, ymax = _np.nanmin(valid), _np.nanmax(valid)
                    if ymin != ymax:
                        ax.set_ylim(ymin, ymax)
            else:
                img.set_data(res.T)
                # 自动调整色标范围
                valid = res[:, :i+1]
                vmin, vmax = _np.nanmin(valid), _np.nanmax(valid)
                if vmin != vmax:
                    img.set_clim(vmin, vmax)
            if in_ipython:
                clear_output(wait=True)
                display(plt.gcf())
            else:
                plt.pause(0.1)
        if in_ipython:
            clear_output(wait=True)
        else:
            plt.show()
        return res


# ====================
#   ED Time Evolution 
# ======================

# -> CPU
def _in_CPU(
    initial_state: _np.ndarray,
    eigenvalues: _np.ndarray,
    eigenstates: _np.ndarray,
    times:_np.ndarray,
    herm:bool=True,
    scale=-1j
) -> _np.ndarray:
    # 确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    if _np.iscomplexobj(eigenstates) or _np.iscomplexobj(initial_state):
        eigenstates = eigenstates.astype(_np.complex128)
        initial_state = initial_state.astype(_np.complex128)

    # U† |psi> 
    if herm:
        udagger_psi = (eigenstates.T.conj() @ initial_state).reshape(1, -1)
    else:
        udagger_psi = (_np.linalg.solve(eigenstates, initial_state)).reshape(1, -1)

    # U exp() |U†psi>
    if not _np.iscomplexobj(eigenstates) and scale == -1j:
        times_E = _np.broadcast_to(times, (len(eigenvalues), len(times))).T * eigenvalues
        real_part = _np.cos(times_E) * udagger_psi
        imag_part = _np.sin(times_E) * udagger_psi
        return eigenstates @ real_part.T - 1j * (eigenstates @ imag_part.T)
    else:
        times_E = _np.broadcast_to(scale*times, (len(eigenvalues), len(times))).T * eigenvalues
        exptimeEpsi = _np.exp(times_E) * udagger_psi
        return eigenstates @ exptimeEpsi.T

# -> GPU
def _in_GPU(
    initial_state: '_tc.Tensor',
    eigenvalues: '_tc.Tensor',
    eigenstates: '_tc.Tensor',
    times: '_tc.Tensor',
    device,
    herm,
    scale=-1j
) -> _np.ndarray:
    """
    在 GPU 上计算初始态在不同时刻的时间演化态。
    """
    import torch as _tc

    # 将数据从 numpy 数组转换为 GPU 上的 torch.Tensor。
    initial_state = _tc.from_numpy(initial_state).to(device)
    eigenvalues = _tc.from_numpy(eigenvalues).to(device)
    eigenstates = _tc.from_numpy(eigenstates).to(device)
    times = _tc.from_numpy(times).to(device)

    # 确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    if eigenstates.dtype == _tc.complex128 or initial_state.dtype == _tc.complex128:
        eigenstates = eigenstates.to(_tc.complex128)
        initial_state = initial_state.to(_tc.complex128)
 
    # U† |psi>
    if herm:
        udagger_psi = (eigenstates.T.conj() @ initial_state).reshape(1, -1)
    else:
        udagger_psi = (_tc.linalg.solve(eigenstates, initial_state)).reshape(1, -1)

    # U exp() |U†psi>
    if not _tc.is_complex(eigenstates) and scale == -1j:
        times_E = times.unsqueeze(1) * eigenvalues
        real_part = _tc.cos(times_E) * udagger_psi
        imag_part = _tc.sin(times_E) * udagger_psi
        res = eigenstates @ real_part.T - 1j * (eigenstates @ imag_part.T)
    else:
        times_E = (scale*times).unsqueeze(1) * eigenvalues
        exp_timeE_psi = _tc.exp(times_E) * udagger_psi
        time_evolution_states = eigenstates @ exp_timeE_psi.T
        res = time_evolution_states
    
    return res.cpu().numpy()  # 将结果从 GPU 转回 CPU，并转换为 numpy 数组

def get_time_evolution_states_ED(
    initial_state: _np.ndarray,
    eigenvalues: _np.ndarray,
    eigenstates: _np.ndarray,
    times: _np.ndarray, 
    *, 
    failback_to_CPU: bool = False, 
    device_name='cpu',
    herm=True,
    ttype='real-time'
) -> _np.ndarray:
    """
    基于严格对角化的时间演化
    
    Args:
        initial_state (_np.ndarray): 初始量子态
        eigenvalues (_np.ndarray): 哈密顿量本征值
        eigenstates (_np.ndarray): 哈密顿量本征态
        times (_np.ndarray): 时间列表
    
    Returns:
        _np.ndarray: 时间演化量子态矩阵
    """
    initial_state = _np.squeeze(initial_state)
    scale = -1j if ttype=='real-time' else 1.
    try:
        import torch as tc
        device = tc.device(device_name)
        time_states = _in_GPU(initial_state, eigenvalues, eigenstates, times, device, herm, scale)
    except Exception as e:
        if not failback_to_CPU:
            raise e
        time_states = _in_CPU(initial_state, eigenvalues, eigenstates, times, herm, scale)
    return time_states


# =============================================
# evolve and measure
# ==============================================
def evolve_and_measure(
    matrix:sps.csr_array,
    inistate:_np.ndarray,
    tlist:_np.ndarray,
    *,
    measure:list[sps.csr_array] | Callable[[float, _np.ndarray], _np.ndarray] | None = None,
    normalize:bool = False,
    ttype:Literal['real-time', 'imag-time'] = 'real-time',
    method:Literal['eig', 'cpu_mul', 'gpu_mul-cuda:0', 
                    'linear_operator','RK45', 'RK23', 'DOP853', 'Radau', 
                    'BDF', 'LSODA']='cpu_mul',
    traceA = None,
    dtype = None,
    herm = False,
    ivp_kwargs = {}
):
    """calculate the measurement values at different time points

    This function is a wrapper for the `EvolveEngine().measure()` but also 
    integrate the `eig` method.
    
    Parameters
    ----------
    matrix : sps.csr_array | sps.LinearOperator
        the Hamiltonian matrix
    inistate : numpy.ndarray
        the initial state vector
    tlist : numpy.ndarray
        the time list
    measure : list[sps.csr_array] | Callable[[numpy.ndarray], numpy.ndarray] | None, optional
        which observable to measure, by default None
        - `list of sparse matrices`: calculate the measurement values of each
            measurement operator at different time points
        - `function`: calculate the measurement values of the function at
            different time points, reflected in the second and subsequent indices of the
            return value
        - `None`: return the time evolution state at different time points
    normalize : bool, optional
        if True, normalize the state after each evolution, by default False
    ttype : str, optional, by default 'real-time'
        - `type='real-time'`: real-time evolution using `exp(-1j * H * t)`
        - `type='imag-time'`: imaginary-time evolution using `exp(H * t)`
    method : str, optional, by default 'auto'
        - `method='eig'`: use the exact diagonalization method to calculate the time evolution
        - `method='cpu_mul'`: use the CPU method to calculate the time evolution. (when matrix is
        a LinearOperator, the traceA should be passed in)
        - `method='gpu_mul-cuda:0'`: use the GPU method to calculate the time evolution. (LinearOperator
        is not supported in this case)
        - `method='RK45'` ...: use the RK45 ... method to calculate the time evolution, for more
        information, please refer to the `scipy.integrate.solve_ivp` documentation. check,
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    traceA : float, optional
        the trace of the matrix, by default None
        if matrix is a LinearOperator and the method is `cpu_mul`, this parameter is required
    dtype : numpy.dtype, optional
        the data type of the matrix, by default None
        only need to be set when only float is involved
    herm : bool, optional
        whether the matrix is hermitian, by default False
        - if matrix is a LinearOperator, this parameter is required
        - if matrix is a matrix, this parameter is optional. `herm=True` will accelerate a little bit
    ivp_kwargs : dict, optional
        the parameters for the `scipy.integrate.solve_ivp` function, by default {}

    Returns
    -------
    numpy.ndarray
        return a multi-dimensional array, the first dimension is the time point, and the subsequent dimensions are determined by `measure`
    """
    if method == 'eig':
        assert sps.issparse(matrix), f"matrix should be sparse array not {type(matrix)}"
        mat = matrix.toarray()
        herm = _np.allclose(mat, mat.T.conj())
        eigfuc = _np.linalg.eigh if herm else _np.linalg.eig
        evalstate = get_time_evolution_states_ED(
                inistate, *eigfuc(mat), tlist, device_name='cpu', herm=herm, ttype=ttype
        )
        if normalize:
            evalstate /= _np.linalg.norm(evalstate, ord=2, axis=0)
        from .operations import observe_states
        if measure is None:
            return evalstate
        elif isinstance(measure, list):
            return _np.real_if_close([observe_states(evalstate, obs.toarray()) for obs in measure]).T
        else:
            res = [measure(t, evalstate[:, i]) for i,t in enumerate(tlist)]
            try:
                return _np.real_if_close(res)    
            except:
                return res

    assert sps.issparse(matrix) or isinstance(matrix, LinearOperator), f"matrix should be sparse array not {type(matrix)}"
    return EvolveEngine(
        matrix, inistate, ts=tlist, normalize=normalize, ttype=ttype, 
        traceA=traceA, dtype=dtype, herm=herm, method=method, ivp_kwargs=ivp_kwargs
    ).measure(measure)

# =============================================
# chebyshev
# ==============================================

def chebyshev_evolve(mat:_np.ndarray, initstate:_np.ndarray, t:float, max_eng:float, min_eng:float, N:int) -> _np.ndarray:
    """ Chebyshev evolution of a state under a Hamiltonian, `exp( - 1j H t) |initstate>`.
    This function uses Chebyshev polynomial expansion to evolve the state under the Hamiltonian mat.
    
    # (max_eng - min_eng) * t ~ O(1) works better

    # todo: 自动计算误差，通过误差推出循环：

    Parameters
    ----------
    mat : np.ndarray
        the Hamiltonian matrix
    initstate : np.ndarray
        the initial state vector
    t : float
        the time parameter for evolution
    max_eng : float
        maximum energy eigenvalue of the Hamiltonian
    min_eng : float
        minimum energy eigenvalue of the Hamiltonian
    N : int
        the number of Chebyshev polynomials to use

    Returns
    -------
    np.ndarray
        the final state vector after evolution
    
    Notes
    -----
    This is a Chebyshev polynomial expansion method for time evolution.
    If you need to speed up, consider using gpu torch for mat @ xxx.
    For larger scale calculations, consider using petsc, related c++ program see
    https://github.com/Phyzch/Chebyshev_method
    When choosing parameters, let (max_eng - min_eng) * t ~ O(1)
      
    Example
    -------
    >>> L, t, N = 5, 1., 10
    >>> mat = qt.generate.matrix.heisenberg_matrix(L=L)
    >>> initstate = np.random.randn(mat.shape[0])
    >>> initstate /= np.linalg.norm(initstate)
    >>> max_eng, min_eng = np.max(np.linalg.eigvalsh(mat)), np.min(np.linalg.eigvalsh(mat))
    >>> finalstate = chebyshev_evolve(mat, initstate, t, max_eng, min_eng, N)
    >>> np.linalg.norm(finalstate - qt.linalg.expm(mat, c=-t*1j) @ initstate)
    np.float64(1.5768894460867202e-08)
    """
    a = (max_eng + min_eng) / 2
    b = (max_eng - min_eng) / 2
    tmp_state0 = initstate.copy()
    tmp_state1 = (mat @ initstate - a * initstate)/b  #!! main time
    finalstate_cheb = jv(0, b*t) * tmp_state0 * _np.exp(-1j*a*t)
    finalstate_cheb += 2 * (-1j) * jv(1, b*t) * tmp_state1 * _np.exp(-1j*a*t)
    for k in range(2,N):
        tmp_state0 = (2/b) * (mat @ tmp_state1 - a * tmp_state1) - tmp_state0  #!! main time
        tmp_state1, tmp_state0 = tmp_state0, tmp_state1
        finalstate_cheb += 2 * (-1j)**k * jv(k, b*t) * tmp_state1 * _np.exp(-1j*a*t)
    return finalstate_cheb


# =============================================
# Quantum Master Equation
# ==============================================

class Liouvillian(LinearOperator):
    def __init__(self, ham:sps.csr_array|None, lindblad_ops:list[sps.csr_array]|None):
        r"""
        The Liouvillian is given by the following equation:
        
        .. math::
            \mathcal{L}(\rho) = -i [H, \rho] + \sum_{l} L_l \rho L_l^{\dagger} - \frac{1}{2} \sum_{l} (L_l^{\dagger} L_l \rho + \rho L_l^{\dagger} L_l)
        
        where :math:`H` is the Hamiltonian, :math:`L_l` are the Lindblad operators, and :math:`\rho` is the density matrix.

        Notes
        -----
        - The Liouvillian is a linear operator that acts on the density matrix.
        - The Hamiltonian and Lindblad operators should be given in the sparse matrix format.
        - The Lindbladian can be sparse or dense, but it is recommended to use sparse matrices for large systems and dense matrices for small systems.
        
        Parameters
        ----------
        ham : sps.csr_array | None
            The Hamiltonian of the system.
        lindblad_ops : list[sps.csr_array] | None
            The Lindblad operators of the system.
        """
        if ham is not None:
            assert sps.issparse(ham), "ham must be sparse matrix"
        if ham is None and lindblad_ops is None:
            raise ValueError("ham and lindblad_ops cannot be both None")
        
        self.ham = ham
        self.lindblad_ops = lindblad_ops
        self._ham_eff = None
        self._sum_jump = None  # use lazy loading to speed up the initialization
        self.Ns = ham.shape[0] if ham is not None else lindblad_ops[0].shape[0]
        self.dtype = _np.dtype(_np.complex128)
        self.shape = (self.Ns**2, self.Ns**2)
    
    @property
    def ham_eff(self):
        if self._ham_eff is None:
            if self.ham is None:
                self._ham_eff = - 1j * sum(lo.conj().T @ lo for lo in self.lindblad_ops)/2
            elif self.lindblad_ops is None:
                self._ham_eff = self.ham
            else:
                self._ham_eff = self.ham - 1j * sum(lo.conj().T @ lo for lo in self.lindblad_ops)/2
        return self._ham_eff

    @property
    def sum_jump(self):
        """将所有的 jump operator 进行求和，得到一个稀疏矩阵"""
        if self._sum_jump is None:
            if self.lindblad_ops is None:
                return None
            # self._sum_jump = sum(sps.kron(lo, lo.conj()) for lo in self.lindblad_ops)
            # 如果 lo 比较多且简单，那么下面的方法会更高效（占用内存会更多）
            from ..generate.basis.symmetry.basis_class_nb import coodiaglists2csr
            row_result = []
            col_result = []
            ele_result = [] 
            for lo in self.lindblad_ops:
                tmp = sps.kron(lo, lo.conj())
                row_result.append(tmp.row)
                col_result.append(tmp.col)
                ele_result.append(tmp.data)
            self._sum_jump = coodiaglists2csr(row_result=row_result, col_result=col_result, ele_result=ele_result, diag=None, n_row=self.Ns**2, index_type=_np.int32, dtype=_np.complex128)
        return self._sum_jump

    @property
    def trace(self):
        a = 2 * self.Ns * self.ham_eff.trace().imag
        if self.lindblad_ops is None:
            return a
        b = sum(abs(lo.trace())**2 for lo in self.lindblad_ops)
        res = a + b
        if isinstance(res, _np.ndarray):
            return res.item()
        else:
            return res

    def __call__(self, rho):
        drho_dt = -1j * (self.ham_eff @ rho - rho @ self.ham_eff.conj().T) 
        if self.lindblad_ops is None:
            return drho_dt
        for lo in self.lindblad_ops:
            drho_dt += lo @ rho @ lo.conj().T 
        return drho_dt

    def _matvec(self, rho):
        return self(rho.reshape(self.Ns, self.Ns)).flatten()

    def _rmatvec(self, rho):
        rho = rho.reshape(self.Ns, self.Ns)
        drho_dt = -1j * (self.ham_eff.T @ rho - rho @ self.ham_eff.conj()) 
        if self.lindblad_ops is None:
            return drho_dt.flatten()
        for lo in self.lindblad_ops:
            drho_dt += lo.T @ rho @ lo.conj()
        return drho_dt.flatten()
 
    def to_matrix(self):
        r"""Convert the Liouvillian to a matrix form.

        Returns
        -------
        sps.csr_array
            The matrix form of the Liouvillian.
        
        Notes
        -----
        can be used to get the matrix form of the Liouvillian for time evolution.
        
        Example
        -------
        >>> mat = lvn.to_matrix()
        >>> res = qt.linalg.time_measurements(
        ...     mat, rhoinit.flatten(), [10, 20, 30, 40, 50], 
        ...     measure=lambda rho: np.real_if_close([np.trace(rho.reshape(L,L) @ n) for n in particle_number]), 
        ...     type='imag-time')
        >>> plt.plot(res.T, 'o-')
        """
        eye = sps.eye(self.Ns)
        nonherm = -1j * (sps.kron(self.ham_eff, eye) - sps.kron(eye, self.ham_eff.conj()))
        if self.lindblad_ops is None:
            return nonherm
        return nonherm + self.sum_jump

    def evolve_and_measure(
            self,
            inistate:_np.ndarray,
            tlist:list|_np.ndarray|None,
            measure:Callable|_np.ndarray,
            method:Literal['eig', 'cpu_mul', 'gpu_mul-cuda:0', 
                        'linear_operator','RK45', 'RK23', 'DOP853', 'Radau', 
                        'BDF', 'LSODA']='cpu_mul',
            **ivp_kwargs
        ):
        r"""evolve the state and measure the observables

        .. math::
            \frac{d \rho}{dt} = \mathcal{L}(\rho)

        Parameters
        ----------
        inistate : _np.ndarray
            initial state
        tlist : list | _np.ndarray
            time list
        measure : Callable | _np.ndarray | None
            - `list of sparse matrices`: calculate the measurement values of each
            measurement operator at different time points
            - `function`: calculate the measurement values of the function at
                different time points, reflected in the second and subsequent indices of the
                return value
            - `None`: return the time evolution state at different time points
        method : str, optional 
            the method to use for time evolution, by default 'cpu_mul'
            - `eig` : use the exact diagonalization method
            - `cpu_mul`, `gpu_mul-cuda:0`: use the matrix multiplication method
            - `linear_operator`: use the linear operator method, slow but memory efficient
            - `RK45`, `RK23`, `DOP853`, `Radau`, `BDF`, `LSODA`: use the scipy ODE solver
            for more information, please refer to the `scipy.integrate.solve_ivp` documentation.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        **kwargs : dict
            additional arguments for the scipy ODE solver

        Returns
        -------
        _np.ndarray
            the measurement results at different time points
        """
        d = self.Ns
        tlist = _np.asarray(tlist)

        if measure is None:
            measure = lambda t, rho: rho.reshape(d,d)
        elif isinstance(measure, list):
            if method[:7] == 'gpu_mul':
                # convert measure to function
                from ..torch_utils.utils import totc
                import torch as tc
                measure = totc(measure, device=method[8:])
                obs = lambda t, rho: _np.real_if_close([tc.trace(rho.reshape(d,d) @ n).item() for n in measure])
            else:
                obs = lambda t, state: [_np.trace(obsmat @ state.reshape(d,d)) for obsmat in measure]
        else:
            obs = measure
        
        matmul = self.to_matrix() if method[:7] in ['eig', 'cpu_mul', 'gpu_mul'] else self

        if method == 'linear_operator':
            traceA = self.trace
            method = 'cpu_mul'
        else:
            traceA = None

        return evolve_and_measure(
            matmul, inistate.flatten(), tlist, measure=obs, 
            normalize=False, ttype='imag-time', method=method, 
            traceA=traceA, herm=False, ivp_kwargs=ivp_kwargs
        )
        

    def steady_state(self, method:Literal['direct', 'eig', 'svd'] = 'direct'):
        if method == 'direct':
            # Find the weight, to stable the iteration
            L_mat = self.to_matrix()
            weight = _np.mean(abs(L_mat.data))

            # add normalization constraint by adding a row of vec(weight*I)
            n = self.Ns
            N = n * n
            # Create an n x n sparse matrix with the first row as (weight * I).reshape(1, -1), others are zeros
            eye_row = sps.lil_array((N, N))
            eye_row[0, :] = (sps.eye(n, format='lil') * weight).reshape(1, -1)
            L_mat_aug = L_mat + eye_row.tocsr()

            # initial guess
            x0 = _np.zeros((N, 1), dtype=_np.complex128)
            x0[0, 0] = weight

            out = spsolve(L_mat_aug, x0)
            return out.reshape(n, n)
        elif method == 'eig':
            L = self.to_matrix()
            n = self.Ns
            N = n * n
            # from .usenumba.operations_numba import dot_parallel
            def LdagL_matvec(x):
                # return dot_parallel(L.conj().T, dot_parallel(L, x))
                return L.conj().T @ (L @ x)
            linop = LinearOperator((N, N), matvec=LdagL_matvec, dtype=_np.complex128)
            val, vec = eigsh(linop, k=1, which='SM')
            rho = vec.reshape(self.Ns, self.Ns)
            return rho / _np.trace(rho)
        elif method == 'svd':
            n = self.Ns
            N = n * n
            L_mat = self.to_matrix()
            u, s, v = svds(L_mat, k=1, which='SM')
            rho = v.reshape(n, n)
            return rho / rho.trace()
        else:
            raise ValueError("method should be 'direct' or 'eig' or 'svd'")

    def trajectory_measure(
        self, 
        inistate:_np.ndarray, 
        tlist:list|_np.ndarray, 
        measure:Callable|_np.ndarray,
        method:Literal['cpu_mul', 'gpu_mul', 'linear_operator',
                    'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']='cpu_mul',
        **kwargs
    ):
        for t in tqdm(tlist, ascii=True):

            pass
        # it should look like,
        
        # def integrate(self, t, copy=False):
        #     t_old, y_old = self._integrator.get_state(copy=False)
        #     norm_old = self._prob_func(y_old)
        #     while t_old < t:
        #         t_step, state = self._integrator.mcstep(t, copy=False)
        #         norm = self._prob_func(state)
        #         if norm <= self.target_norm:
        #             t_col, state = self._find_collapse_time(norm_old, norm,
        #                                                     t_old, t_step)
        #             self._do_collapse(t_col, state)
        #             t_old, y_old = self._integrator.get_state(copy=False)
        #             norm_old = 1.
        #         else:
        #             t_old, y_old = t_step, state
        #             norm_old = norm

        #     return t_old, _data.mul(y_old, 1 / self._norm_func(y_old))

        # def run(self, tlist):
        #     for t in tlist[1:]:
        #         yield self.integrate(t, False)
    
        # reference https://qutip.org/docs/4.7/guide/dynamics/dynamics-monte.html