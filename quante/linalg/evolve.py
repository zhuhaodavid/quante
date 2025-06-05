# # -*- coding: utf-8 -*-
# # @Author: hzhu
# # @Date:   2023-10-22 17:13:49
# # @Last Modified by:   hzhu
# # @Last Modified time: 2025-05-23 11:04:45

from scipy import sparse as sps
from scipy.special import jv
from scipy.sparse.linalg import LinearOperator, expm_multiply
from scipy.integrate import solve_ivp

import numpy as _np
import warnings as _warnings
from typing import Callable, Union, Literal, TYPE_CHECKING
from functools import lru_cache
from typing import Literal
from tqdm import tqdm

from .operations import expect
from .super_operator import Liouvillian

if TYPE_CHECKING:  # 类型检查时，导入 torch
    import torch as _tc

__all__ = [
    "evolve_and_measure",
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


# ====================
#   ED Time Evolution 
# ======================
def Uinvpsi(pkg, eigenstates, initial_state, herm):
    # U† |psi>
    if herm:
        if eigenstates.dtype == pkg.float64 and initial_state.dtype == pkg.complex128:
            # 直接分别计算实部和虚部，避免构造复数再分解
            udagger_psi = eigenstates.T @ initial_state.real + 1j * (eigenstates.T @ initial_state.imag)
        else:
            udagger_psi = eigenstates.T.conj() @ initial_state
    else:
        udagger_psi = pkg.linalg.solve(eigenstates, initial_state)
    return udagger_psi.reshape(1,-1)

def Uexp(pkg, eigenvalues, eigenstates, times, udagger_psi, scale, shift=False):
    # Ensure correct dtype for broadcasting and computation
    if hasattr(pkg, "ndarray"):  # numpy
        times = pkg.asarray(times)
    else:  # torch
        times = pkg.asarray(times)
        times = times if times.device == eigenvalues.device else times.to(eigenvalues.device)

    # Broadcasting for time evolution
    times_E = times.reshape(-1, 1) * eigenvalues.reshape(1, -1)
    if eigenstates.dtype == udagger_psi.dtype == pkg.float64 and scale == -1j:
        # Real-time evolution: exp(-i E t)
        # exp(-i E t) = cos(E t) - i sin(E t)
        real_part = pkg.cos(times_E) * udagger_psi
        imag_part = pkg.sin(times_E) * udagger_psi
        res = eigenstates @ real_part.T - 1j * (eigenstates @ imag_part.T)
    else:
        if shift:
            scaled_times_E = scale * times_E
            shift_times_E = pkg.max(pkg.real(scaled_times_E), axis=1).reshape(-1, 1)
            exp_timeE_psi = pkg.exp(scaled_times_E - shift_times_E) * udagger_psi
        else:
            exp_timeE_psi = pkg.exp(scale * times_E) * udagger_psi
        if eigenstates.dtype == pkg.float64 and exp_timeE_psi.dtype == pkg.complex128:
            # 直接分别计算实部和虚部，避免构造复数再分解
            res = eigenstates @ exp_timeE_psi.real.T + 1j * (eigenstates @ exp_timeE_psi.imag.T)
        else:
            # Imaginary-time evolution: exp(E t)
            res = eigenstates @ exp_timeE_psi.T
    return res

# -> CPU
def _in_CPU(
    initial_state: _np.ndarray,
    eigenvalues: _np.ndarray,
    eigenstates: _np.ndarray,
    times:_np.ndarray,
    herm:bool=True,
    scale=-1j, 
    shift=False,
) -> _np.ndarray:
    # 确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    if _np.iscomplexobj(eigenstates) or _np.iscomplexobj(initial_state):
        eigenstates = eigenstates.astype(_np.complex128)
        initial_state = initial_state.astype(_np.complex128)
    udagger_psi = Uinvpsi(_np, eigenstates, initial_state, herm)  # U† |psi>
    # U exp(-iEt) U† |psi>
    return Uexp(_np, eigenvalues, eigenstates, times, udagger_psi, scale, shift=shift)

# -> GPU
def _in_GPU(
    initial_state: '_tc.Tensor',
    eigenvalues: '_tc.Tensor',
    eigenstates: '_tc.Tensor',
    times: '_tc.Tensor',
    device,
    herm,
    scale=-1j,
    shift=False,
) -> _np.ndarray:
    """
    在 GPU 上计算初始态在不同时刻的时间演化态。
    """
    import torch as _tc
    from ..torch_utils import totc

    # 将数据从 numpy 数组转换为 GPU 上的 torch.Tensor。
    initial_state = totc(initial_state, device=device)
    eigenvalues = totc(eigenvalues, device=device)
    eigenstates = totc(eigenstates, device=device)
    times = totc(times, device=device)

    # 确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    if eigenstates.dtype == _tc.complex128 or initial_state.dtype == _tc.complex128:
        eigenstates = eigenstates.to(_tc.complex128)
        initial_state = initial_state.to(_tc.complex128)
 
    udagger_psi = Uinvpsi(_tc, eigenstates, initial_state, herm)  # U† |psi>
    return Uexp(_tc, eigenvalues, eigenstates, times, udagger_psi, scale, shift=shift) 
    # 将结果从 GPU 转回 CPU，并转换为 numpy 数组

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
        time_states = _in_GPU(initial_state, eigenvalues, eigenstates, times, device, herm, scale).cpu().numpy()
    except Exception as e:
        if not failback_to_CPU:
            raise e
        time_states = _in_CPU(initial_state, eigenvalues, eigenstates, times, herm, scale)
    return time_states


# =============================================
# evolve and measure
# ==============================================


class EvolveEngine:
    """            
    This class make use of the `expm_multiply` function to evolve the state vector.
    The precalculation will be done in the constructor here, which will speed up the
    calculation of the time evolution.
    """
    def __init__(
        self,
        ham: _np.ndarray | sps.csr_array | LinearOperator,
        init_state: _np.ndarray,
        ts: _np.ndarray,
        *,
        normalize: bool = False,
        ttype: Literal['real-time', 'imag-time'] = 'real-time',
        traceA: float | None = None,
        dtype: _np.dtype | None = None,
        herm: bool | Callable = None,
        method: Literal[
            'eig-cpu', 'eig-cuda:0', 'mul-cpu', 'mul-cuda:0',
            'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        ] = 'mul-cpu',
        isdm: bool = False,
        ivp_kwargs: dict = {}
    ):
        """calculate the time evolution of the state vector

        Parameters
        ----------
        ham : ndarray | sps.csr_array | LinearOperator
            the Hamiltonian matrix
        init_state : ndarray
            the initial state vector
        ts : ndarray
            the time list
        normalize : bool, optional
            if True, normalize the state after each evolution, by default False
        ttype : str, optional
            - `ttype='real-time'`: real-time evolution using `exp(-1j * H * t)`
            - `ttype='imag-time'`: imaginary-time evolution using `exp(H * t)`
        traceA : float | None, optional
            the trace of the matrix, by default None
            if matrix is a LinearOperator and the method is `mul-cpu`, this parameter is required
        dtype : dtype | None, optional
            the data type of the matrix, by default None
            only need to be set when only float is involved
        herm : bool | Callable, optional
            whether the matrix is hermitian, by default None
            - if matrix is a LinearOperator, this parameter is required
            - if matrix is a matrix, this parameter is optional. `herm=True` will accelerate a little bit
            - if None is passed, the hermitian property will be checked
        method : str, optional, by default `mul-cpu`
            - `method='eig-cpu'`: use the exact diagonalization method to calculate the time evolution (will
            convert the sparse matrix to dense matrix, **not support LinearOperator**)
            - `method='eig-cuda:0'`: use the GPU method to calculate the time evolution. (will convert the
            sparse matrix to dense matrix, **not support LinearOperator**)
            - `method='mul-cpu'`: use the CPU method to calculate the time evolution. (when matrix is
            a LinearOperator, the traceA should be passed in)
            - `method='mul-cuda:0'`: use the GPU method to calculate the time evolution. (LinearOperator
            is not supported in this case, **not support LinearOperator**)
            - `method='RK45'` ...: use the RK45 ... method to calculate the time evolution, for more
            information, please refer to the `scipy.integrate.solve_ivp` documentation. See:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
            notes: # todo support torch ode by `torchdiffeq`
        ivp_kwargs : dict, optional
            the parameters for the `scipy.integrate.solve_ivp` function, by default {}
        """
        device = method[4:] if method[:3] in ['eig', 'mul'] else 'cpu'
        
        if isinstance(ham, Liouvillian):
            isdm = True
            # convert the Liouvillian to a csr sparse matrix if needed
            # else it will be a LinearOperator, along with the traceA
            ham, traceA = ham._tolo()
        if device == 'cpu':
            # for cpu, the hamiltonian can be a LinearOperator or a csr sparse matrix
            if isinstance(ham, LinearOperator):
                self.csr_mt = ham
                assert method[:3] != 'eig', "eig method is not supported when using LinearOperator"
                if device != 'cpu':
                    assert method != 'mul', "mul-cuda method is not supported when using LinearOperator"
            else:
                try:
                    self.csr_mt = ham.tocsr()
                except:
                    self.csr_mt = sps.csr_array(ham)
        
            # the initial state should be a column vector
            dtype = _np.complex128 if dtype is None else dtype
            self.psi = init_state.reshape(-1, 1).astype(dtype)
            self.pkg = _np

        else:
            # first move the data to the device
            # assert sps.issparse(ham), "ham should be sparse array"
            from ..torch_utils.utils import totc
            import torch as tc
            dtype = tc.complex128 if dtype is None else dtype
            self.csr_mt = totc(ham, device=device)
            self.psi = totc(init_state, device=device, dtype=dtype).reshape(-1, 1)
            self.pkg = tc

        self.cur_state = self.psi
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
        self.ivp_kwargs = dict(rtol=1e-9, atol=1e-12)
        self.ivp_kwargs.update(**ivp_kwargs)
        self.isdm = isdm
        if isdm:
            d = int(self.csr_mt.shape[0]**0.5)
            self.state_shape = (d, d)
        else:
            self.state_shape = (self.csr_mt.shape[0], )
        self._eigen = self._UinvPsi = self._all_states = None
    
    @property
    def eigen(self):
        if self._eigen is None:
            mat = self.csr_mt.toarray() if self.device == 'cpu' else self.csr_mt.to_dense()
            herm = self.pkg.allclose(mat, mat.conj().T)
            if self.herm is None:
                self.herm = herm
            else:
                assert self.herm == herm, "the hermitian property is not consistent with the matrix"
            eigf = self.pkg.linalg.eigh if self.herm else self.pkg.linalg.eig
            self._eigen = eigf(mat)
        return self._eigen
    
    @property
    def UinvPsi(self):
        if self._UinvPsi is None:
            self._UinvPsi = Uinvpsi(self.pkg, self.eigen[1], self.psi, self.herm)
        return self._UinvPsi
    
    # confine the maxsize of the cache to 2 to avoid memory overflow
    # since the matrix is large, the cache will be large
    @lru_cache(maxsize=2)
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
            self.evolved_time += dt
            # ============= main =============
            # we can choose to use 'eig', 'mul' or 'ivp' method
            # it depends on the string in method
            if self.method[:3] == 'mul':
                ee = self.get_evolve_engine(round(dt,14))
                self.cur_state = ee(self.cur_state)
            elif self.method[:3] == 'eig':
                uinpsi = Uinvpsi(self.pkg, self.eigen[1], self.cur_state, self.herm)
                self.cur_state = Uexp(
                    self.pkg, *self.eigen, self.evolved_time,
                    uinpsi, self.scale, norm=self.normalize
                )
            else:
                if self.scale == 1.:
                    matmul = lambda t, state: self.csr_mt @ state
                else:
                    matmul = lambda t, state: -1j * (self.csr_mt @ state)
                sol = solve_ivp(
                    matmul, (0, dt), self.cur_state.flatten(), t_eval=[dt],
                    method=self.method, **self.ivp_kwargs
                )
                if not sol.success:
                    raise RuntimeError(
                        f"ODE solver failed with message: {sol.message}"
                    )
                self.cur_state = sol.y
            # ============= end =============
            if self.normalize:
                self.cur_state /= self.pkg.linalg.norm(self.cur_state)
        return self.cur_state
    
    def pre_obs(self, obs):
        if obs is None:
            return lambda t, state: state.reshape(self.state_shape)
        elif isinstance(obs, (sps.sparray, sps.spmatrix, list, _np.ndarray)):
            if self.device != 'cpu':
                from ..torch_utils import totc
                obs = totc(obs, device=self.device)
            return lambda t, state: expect(obs, state.reshape(self.state_shape), isdm=self.isdm)
        elif callable(obs):
            return obs
        else:
            raise ValueError("obs should be a list of sparse matrices or a function")
        
    def measure(
        self, 
        obs: (None 
              | _np.ndarray
              | sps.csr_array 
              | list[sps.csr_array] 
              | list[_np.ndarray] 
              | Callable[[float, _np.ndarray], _np.ndarray] 
        ) = None,
        *,
        progressbar: bool = True
    ):
        """calculate the expectation value of the observable
        
        Parameters
        ----------
        obs : ndarray | csr_array | list[ndarray] | list[csr_array] | Callable[[float, ndarray], ndarray] | None, optional
            which observable to measure, by default None
            - `sparse/dense matrix`: calculate the measurement values of the observable
            - `list of sparse/dense matrices`: calculate the measurement values of each
            measurement operator at different time points
            - `function`: calculate the measurement values of the function at
            different time points, reflected in the second and subsequent indices of the
            return value
            - `None`: return the time evolution state at different time points
        progressbar : bool, optional
            whether to show the progress bar, by default True

        Returns
        -------
        ndarray
            the expectation value of the observable
        """
        if self.method[:3] == 'eig':
            # it would be faster if all states are calculated at once
            # we should move the cur_step and psi so that it is 
            # consistent with the run() method used in other methods
            self.cur_step = len(self.tlist)
            self.cur_state = self.all_states[:,-1]
            return self._eigen_measure(obs)
        else:
            obs = self.pre_obs(obs)
            res = []
            t_iter = tqdm(self.tlist, ascii=True) if progressbar else self.tlist
            for t in t_iter:
                state = self.run()
                try:
                    res.append(obs(t, state))
                except Exception as e:
                    raise MeasureError(f"Error in measure: {e}. \n"
                            "Please check the measure function so that it can deal with the"
                            f"states with \ntype:{type(state)}, shape:{state.shape}, "
                            f"dtype:{state.dtype}") from e
            try:
                return _np.real_if_close(res)    
            except:
                return res

    def plot_measure(
        self,
        obs:list[sps.csr_array]|Callable[[float, _np.ndarray], _np.ndarray], 
        *args, 
        ax=None, 
        **kwargs
    ):
        """dynamic plot the expectation value of the observable

        Parameters
        ----------
        obs : list[sps.csr_array] | Callable[[float, _np.ndarray], _np.ndarray]
            the observable matrix or a function that takes the time and state as input
        ax : matplotlib.axes.Axes, optional
            the axes to plot on, by default None
        *args : tuple
            additional arguments to pass to the plot function
        **kwargs : dict
            additional keyword arguments to pass to the plot

        Returns
        -------
        _np.ndarray
            the expectation value of the observable
        
        Example
        -------
        >>> import numpy as np
        >>> import quante as qt
        >>> op = qt.generate.operas
        >>> tlist = np.linspace(0, 10, 100)
        >>> # Model
        >>> L=10
        >>> J, γ = 1., 0.1
        >>> builder = op.SpinBuilder()
        >>> for l in range(L-1):
        ...     builder += "+-", [l+1, l], (J+γ)/2
        ...     builder += "+-", [l, l+1], (J-γ)/2
        >>> ham = builder.build()
        >>> basis = qt.generate.basis.spin_basis(L=L)
        >>> hammat = ham.to_matrix(basis=basis, sparse=True)
        >>> # Observation
        >>> obsmatlist = [op.z(L//2).to_matrix(basis, sparse=True)]
        >>> init_state = qt.generate.state.neel(L=L, down_first=True)
        >>> # Plot
        >>> evolve_engine = qt.linalg.EvolveEngine(
        ...     hammat, init_state, ts=tlist, normalize=True
        ... )
        >>> res = evolve_engine.plot_measure(obsmatlist, 'o-') 
        """
        obs = self.pre_obs(obs)
        from ..basicfun import DynamicPlot
        try:
            dp = DynamicPlot(self.tlist, ax, *args, **kwargs)
            for t in self.tlist:
                state = self.run()
                res_t = _np.real_if_close(obs(t, state))
                dp.append(res_t)
            res = dp.data
        except Exception as e:
            # raise MeasureError(f"Error in measure: {e}")
            _warnings.warn(f"DynamicPlot error: {e}, with result res: {res}")
        return res
    
    @property
    def all_states(self):
        if self._all_states is None:
            if self.device == 'cpu':
                self._all_states = _in_CPU(
                    self.psi, *self.eigen, self.tlist, 
                    herm=self.herm, scale=self.scale, shift=self.normalize
                )
            else:
                self._all_states = _in_GPU(
                    self.psi, *self.eigen, self.tlist, self.device, 
                    herm=self.herm, scale=self.scale, shift=self.normalize
                )
            if self.normalize:
                self._all_states /= self.pkg.linalg.norm(self._all_states, axis=0)
        return self._all_states
    
    def _eigen_measure(self, measure):
        states = self.all_states
        if measure is None:
            return states.T
        try:
            if isinstance(measure, (sps.sparray, sps.spmatrix, list, _np.ndarray)):
                if self.device != 'cpu':
                    from ..torch_utils import totc
                    measure = totc(measure, device=self.device)
                return expect(measure, states.reshape(*self.state_shape,-1), isdm=self.isdm).T
            else:
                return _np.array([
                    measure(t, states[:, i].reshape(self.state_shape)) 
                    for i, t in enumerate(self.tlist)
                ])
        except Exception as e:
            raise MeasureError(f"Error in measure: {e}. \n"
                            "Please check the measure function so that it can deal with the"
                            f"states with \ntype:{type(states)}, shape:{states.shape}, "
                            f"dtype:{states.dtype}") from e
    

class MeasureError(Exception):
    """Custom exception for measurement errors."""
    pass

def evolve_and_measure(
    matrix: _np.ndarray | sps.csr_array | Liouvillian,
    inistate: _np.ndarray,
    tlist: _np.ndarray,
    *,
    measure: (None 
              | _np.ndarray
              | sps.csr_array 
              | list[sps.csr_array] 
              | list[_np.ndarray] 
              | Callable[[float, _np.ndarray], _np.ndarray] 
    ) = None,
    normalize: bool = False,
    method: Literal[
        'eig-cpu', 'eig-cuda:0', 'mul-cpu', 'mul-cuda:0',
        'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
    ] = 'mul-cpu',
    herm = None,
    progressbar: bool = True,
    ivp_kwargs = {},
):
    """
    A wrapper for the `EvolveEngine().measure()`

    Calculate the time evolution of the state vector and measure the observable.

    For LinearOperator, _np.ndarray and sps.csr_array, the time evolution is calculated as
    .. math::
        \\psi(t) = \\exp(-i H t) \\psi(0)
    where :math:`H` is the Hamiltonian matrix.

    For Liouvillian, the time evolution is calculated as
    .. math::
        \\rho(t) = \\exp(L t) \\rho(0)
    where :math:`L` is the Liouvillian operator.

    Parameters
    ----------
    matrix : ndarray | csr_array | Liouvillian
        the Hamiltonian or Liouvillian matrix/operator
    inistate : ndarray
        the initial state vector
    tlist : ndarray
        the time list
    measure : ndarray | csr_array | list[ndarray] | list[csr_array] | Callable[[float, ndarray], ndarray] | None, optional
        which observable to measure, by default None
        - `sparse/dense matrix`: calculate the measurement values of the observable
        - `list of sparse/dense matrices`: calculate the measurement values of each
          measurement operator at different time points
        - `function`: calculate the measurement values of the function at
          different time points, reflected in the second and subsequent indices of the
          return value
        - `None`: return the time evolution state at different time points
    normalize : bool, optional
        if True, normalize the state after each evolution, by default False
    method : str, optional, by default `mul-cpu`
        - `method='eig-cpu'`: use the exact diagonalization method to calculate the time evolution (will
          convert the sparse matrix to dense matrix, **not support LinearOperator**)
        - `method='eig-cuda:0'`: use the GPU method to calculate the time evolution. (will convert the
          sparse matrix to dense matrix, **not support LinearOperator**)
        - `method='mul-cpu'`: use the CPU method to calculate the time evolution. (when matrix is
          a LinearOperator, the traceA should be passed in)
        - `method='mul-cuda:0'`: use the GPU method to calculate the time evolution. (LinearOperator
          is not supported in this case, **not support LinearOperator**)
        - `method='RK45'` ...: use the RK45 ... method to calculate the time evolution, for more
          information, please refer to the `scipy.integrate.solve_ivp` documentation. See:
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        notes: # todo support torch ode by `torchdiffeq`
    herm : bool, optional
        whether the matrix is hermitian, by default None
        - if matrix is a LinearOperator, this parameter is required
        - if matrix is a matrix, this parameter is optional. `herm=True` will accelerate a little bit
        - if None is given, the function will check the hermitian property of the matrix
    progressbar : bool, optional
        whether to show the progress bar, by default True
        Note that `eig` method will not show the progress bar
    ivp_kwargs : dict, optional
        the parameters for the `scipy.integrate.solve_ivp` function, by default {}

    Returns
    -------
    ndarray
        return a multi-dimensional array, the first dimension is the time point, and the subsequent dimensions are determined by `measure`
    """
    tlist = _np.asarray(tlist)
    ttype = 'imag-time' if isinstance(matrix, Liouvillian) else 'real-time'

    return EvolveEngine(
        matrix, inistate, tlist, ttype=ttype,
        normalize=normalize, method=method,
        herm=herm, ivp_kwargs=ivp_kwargs
    ).measure(measure, progressbar=progressbar)

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
