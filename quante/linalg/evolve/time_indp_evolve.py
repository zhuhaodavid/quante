# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 18:32:54
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-05 21:30:59


from scipy import sparse as sps
from scipy.sparse.linalg import LinearOperator
from scipy.integrate import solve_ivp

import numpy as _np
import warnings as _warnings
from typing import Callable, Literal
from functools import lru_cache
from typing import Literal
from tqdm import tqdm

from ...measure.expect import expect
from .EDevolve import _in_CPU, _in_GPU, Uinvpsi, Uexp

__all__ = [
    'EvolveEngine', 'evolve_and_measure'
]

class EvolveEngine:
    """            
    This class make use of the `expm_multiply` function to evolve the state vector.
    The precalculation will be done in the constructor here, which may speed up the
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
        
        if isinstance(ham, LinearOperator) and traceA is None:
            # convert the Lindbladian to a csr sparse matrix if needed
            # else it will be a LinearOperator, along with the traceA
            if traceA is None:
                try:
                    traceA = ham.trace
                except AttributeError:
                    raise ValueError("ham should have the trace attribute, please pass in the traceA")
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
            # from ...bridge.torch_utils.linalg.sparse import to_csr
            from ...bridge.torch_utils import totc
            import torch as tc
            dtype = tc.complex128 if dtype is None else dtype
            self.csr_mt = totc(ham, device=device)
            # self.psi = to_csr(init_state, device=device, dtype=dtype).reshape(-1, 1)
            self.psi = totc(init_state.reshape(-1, 1), device=device, dtype=dtype)
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
        self._eigen = self._UinvPsi = self._all_states = None

    ####################
    # main evolve
    ####################

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
            from .nbfuc.expm_mul_core import _evolve_engine
            return _evolve_engine(self.csr_mt, scale=self.scale, t=dt, traceA=self.traceA, herm=self.herm)
        else:
            from ...bridge.torch_utils.linalg.expm_multiply import evolve_engine
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
    
    ####################
    # main measure
    ####################

   
    def pre_obs(self, obs):
        if obs is None:
            return lambda t, state: state.reshape(-1)
        elif isinstance(obs, (sps.sparray, sps.spmatrix, list, _np.ndarray)):
            if self.device != 'cpu':
                from ...bridge.torch_utils import totc
                obs = totc(obs, device=self.device)
            return lambda t, state: expect(obs, state.reshape(-1), isdm=False)
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
                    raise MeasureError(
                        "An error occurred while processing the measure function. \n"
                        "Please ensure the measure function can handle the following:\n"
                        f"- State type: {type(state)}\n"
                        f"- State shape: {state.shape}\n"
                        f"- State dtype: {state.dtype}\n"
                        f"Error details:\n{e}"
                    ) from e
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
        from ...basicfun import DynamicPlot
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
                    from ...bridge.torch_utils import totc
                    measure = totc(measure, device=self.device)
                return expect(measure, states, isdm=False).T
            else:
                return _np.array([
                    measure(t, states[:, i]) 
                    for i, t in enumerate(self.tlist)
                ])
        except Exception as e:
            raise MeasureError(
                "An error occurred while processing the measure function. "
                "Please ensure the measure function can handle the following:\n"
                f"- State type: {type(states)}\n"
                f"- State shape: {states.shape}\n"
                f"- State dtype: {states.dtype}\n"
                f"Error details:\n{e}"
            ) from e
    

class MeasureError(Exception):
    """Custom exception for measurement errors."""
    pass

def evolve_and_measure(
    matrix: _np.ndarray | sps.csr_array | LinearOperator,
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
    ttype: Literal['real-time', 'imag-time'] = 'real-time',
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

    Parameters
    ----------
    matrix : ndarray | csr_array | LinearOperator
        the Hamiltonian or Lindbladian matrix/operator
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
    ttype : str, optional
        - `ttype='real-time'`: real-time evolution using `exp(-1j * H * t)`
        - `ttype='imag-time'`: imaginary-time evolution using `exp(H * t)`
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
    return EvolveEngine(
        matrix, inistate, tlist, ttype=ttype,
        normalize=normalize, method=method,
        herm=herm, ivp_kwargs=ivp_kwargs
    ).measure(measure, progressbar=progressbar)
