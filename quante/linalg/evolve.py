# # -*- coding: utf-8 -*-
# # @Author: hzhu
# # @Date:   2023-10-22 17:13:49
# # @Last Modified by:   hzhu
# # @Last Modified time: 2025-05-16 12:55:52

from scipy import sparse as sps
from scipy.special import jv
from scipy.sparse.linalg import LinearOperator, expm_multiply, spsolve, eigsh, svds

import numpy as _np
import warnings as _warnings
from typing import Callable, Union, Literal
from functools import lru_cache
from typing import Literal
from tqdm import tqdm

__all__ = [
    "expm_multiply",
    "EvolveEngine",
    "get_time_evolution_states_ED",
    "chebyshev_evolve",
    "evolve_and_measure",
    "Liouvillian"
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
    def __init__(self, ham, init_state, ts, normalize=False, ttype='real-time', traceA=None):
        if init_state.ndim == 1:
            self.psi = init_state.reshape(-1, 1).astype(_np.complex128)
        else:
            self.psi = init_state.astype(_np.complex128)
        if isinstance(ham, LinearOperator):
            self.csr_mt = ham
        else:
            try:
                self.csr_mt = ham.tocsr()
            except:
                self.csr_mt = sps.csr_array(ham)
        self.dts = _np.diff(ts)
        self.dts = _np.insert(self.dts, 0, ts[0])
        self.evolved_time = 0
        self.cur_step = 0
        self.normalize = normalize
        self.scale = -1j if ttype == 'real-time' else 1.0
        self.traceA = traceA
    
    @lru_cache(maxsize=None)
    def get_evolve_engine(self, dt):
        from .usenumba.expm_multiply_numba import _evolve_engine
        return _evolve_engine(self.csr_mt, scale=self.scale, t=dt, traceA=self.traceA)

    def run(self):
        try:
            dt = self.dts[self.cur_step]
        except:
            _warnings.warn(
                f"t {self.evolved_time} has been reached, dt = {self.dts[-1]} will be used"
            )
            dt = self.dts[-1]
        self.cur_step += 1
        if dt != 0:
            ee = self.get_evolve_engine(round(dt,14))
            self.psi = ee(self.psi)
            if self.normalize:
                self.psi /= _np.linalg.norm(self.psi, ord=2)
            self.evolved_time += dt
        return self.psi

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
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # 类型检查时，导入 torch
    import torch as _tc

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
    
    return res.cpu().numpy()

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
    measure:list[sps.csr_array] | Callable[[float, _np.ndarray], _np.ndarray],
    normalize:bool = False,
    ttype:Literal['real-time', 'imag-time'] = 'real-time',
    method:Literal['auto', 'eig', 'gpu_mul', 'cpu_mul'] = 'auto',
):
    """calculate the measurement values at different time points

    Parameters
    ----------
    matrix : sps.csr_array
        the Hamiltonian matrix
    inistate : _np.ndarray
        the initial state vector
    tlist : _np.ndarray
        the time list
    measure : list[sps.csr_array] | Callable[[_np.ndarray], _np.ndarray]
        - `measure` is a list of sparse matrices: calculate the measurement values of each
            measurement operator at different time points
        - `measure` is a function: calculate the measurement values of the function at
            different time points, reflected in the second and subsequent indices of the
            return value
    normalize : bool, optional
        if True, normalize the state after each evolution, by default False
    type : str, optional, by default 'real-time'
        - `type='real-time'`: real-time evolution using `exp(-1j * H * t)`
        - `type='imag-time'`: imaginary-time evolution using `exp(H * t)`
    method : str, optional, by default 'auto'
        - `method='eig'`: use the exact diagonalization method to calculate the time evolution
        - `method='gpu_mul'`: use the GPU method to calculate the time evolution
        - `method='cpu_mul'`: use the CPU method to calculate the time evolution
        - `method='auto'`: automatically select the calculation method, first try to call the GPU method, if it fails, use the CPU method

    Returns
    -------
    _np.ndarray
        return a multi-dimensional array, the first dimension is the time point, and the subsequent dimensions are determined by `measure`
    """
    assert sps.issparse(matrix), f"matrix should be sparse array not {type(matrix)}"
    assert method in ['auto', 'eig', 'gpu_mul', 'cpu_mul'], f"method should be one of ['auto', 'eig', 'gpu_mul', 'cpu_mul'] not {method}"
    if method == 'eig':
        ###################################################################################
        # Diagonalize
        ###################################################################################
        mat = matrix.toarray()
        # ----------- main ------------
        if _np.allclose(mat, mat.T.conj()):
            engres = _np.linalg.eigh(mat)
            evalstate = get_time_evolution_states_ED(
                inistate, *engres, tlist, failback_to_CPU=True, herm=True, ttype=ttype
            )
        else:
            engres = _np.linalg.eig(mat)
            evalstate = get_time_evolution_states_ED(
                inistate, *engres, tlist, failback_to_CPU=True, herm=False, ttype=ttype
            )
        if normalize:
            evalstate /= _np.linalg.norm(evalstate, ord=2, axis=0)
        # ----------- end main ------------
        
        from .operations import observe_states
        if isinstance(measure, list):
            return _np.real_if_close([observe_states(evalstate, obs.toarray()) for obs in measure]).T
        else:
            return _np.real_if_close([measure(evalstate[:, i]) for i in range(len(tlist))])

    try:
        if method in ['gpu_mul', 'auto']:
            ###################################################################################
            # GPU 
            ###################################################################################
            from ..torch_utils.linalg.expm_multiply import EvolveEngine as tcEvolveEngine
            from ..torch_utils.linalg.sparse import to_csr
            from ..torch_utils.utils import totc
            from tqdm import tqdm
            import torch as tc # type: ignore
            assert tc.cuda.is_available(), "CUDA is not available"
            device = tc.device('cuda:0')
            # convert measure to function
            if isinstance(measure, list):
                obsmatlist = [to_csr(o, device=device, dtype=tc.complex128) for o in measure]
                obs = lambda t, state: [(state.conj().reshape(1,-1) @ (obsmat @ state).reshape(-1,1)).item() for obsmat in obsmatlist]
            else:
                obs = measure

            # ----------- main ------------
            hammat0 = to_csr(matrix, device=device)
            inistate = totc(inistate, device=device)
            evolve_engine = tcEvolveEngine(hammat0, inistate, ts=tlist, normalize=normalize, ttype=ttype)
            res = []
            for t in tqdm(tlist, ascii=True):
                state = evolve_engine.run()
                res.append(obs(t, state))
            # ----------- end main ------------

            return _np.real_if_close(res)
        else:
            raise RuntimeError(f"use cpu")
    except RuntimeError as e:
        if method == 'gpu_mul':
            raise RuntimeError(f"GPU is not available due to {e}, please use CPU or set method='cpu_mul'")

        ###################################################################################
        # CPU
        ###################################################################################
        from tqdm import tqdm
        # convert measure to function
        if isinstance(measure, list):
            obs = lambda t, state: [state.conj().reshape(-1) @ (obsmat @ state).reshape(-1) for obsmat in measure]
        else:
            obs = measure

        # ----------- main ------------
        evolve_engine = EvolveEngine(matrix, inistate, ts=tlist, normalize=normalize, ttype=ttype)
        res = []
        for t in tqdm(tlist, ascii=True):
            state = evolve_engine.run()
            res.append(obs(t, state))
        # ----------- end main ------------

        return _np.real_if_close(res)    


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
        
        Parameters
        ----------
        ham : sps.csr_array | None
            The Hamiltonian of the system.
        lindblad_ops : list[sps.csr_array] | None
            The Lindblad operators of the system.
        """
        if ham is not None:
            assert sps.issparse(ham), "ham must be sparse matrix"
        elif lindblad_ops is not None:
            for lo in lindblad_ops:
                assert sps.issparse(lo), "lindblad_ops must be sparse matrix"
        else:
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
        return a + b

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
            tlist:list|_np.ndarray,
            measure:Callable|_np.ndarray,
            method:Literal['eig', 'cpu_mul', 'gpu_mul', 'linear_operator',
                        'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']='cpu_mul',
            **kwargs
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
        measure : Callable | _np.ndarray
            - `measure` is a list of sparse matrices: measure the expectation value of each operator at different time points
            - `measure` is a function: measure the expectation value of the function at different time points, reflected in the second and subsequent indices of the return value
        method : ['cpu_mul', 'gpu_mul', 'linear_operator', 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'], optional
            the method to use for time evolution
            - `cpu_mul`, `gpu_mul`: use the matrix multiplication method, fast but memory consuming
            - `linear_operator`: use the linear operator method, slow but memory efficient
            - `RK45`, `RK23`, `DOP853`, `Radau`, `BDF`, `LSODA`: use the scipy ODE solver
            by default 'cpu_mul'
        **kwargs : dict
            additional arguments for the scipy ODE solver

        Returns
        -------
        _np.ndarray
            the measurement results at different time points
        """
        d = self.Ns
        tlist = _np.asarray(tlist)
        if method in ['eig', 'cpu_mul', 'gpu_mul']:
            if isinstance(measure, list):
                if method == 'gpu_mul':
                    # convert measure to function
                    obs = lambda t, state: [_np.trace(obsmat @ state.reshape(d,d)) for obsmat in measure]
                else:
                    from ..torch_utils.utils import totc
                    import torch as tc
                    measure = totc(measure, device='cuda')
                    obs = lambda t, rho: _np.real_if_close([tc.trace(rho.reshape(d,d) @ n).item() for n in measure])
            else:
                obs = measure
            return evolve_and_measure(
                self.to_matrix(), inistate.flatten(), tlist, measure=obs, 
                method=method, ttype='imag-time', normalize=False
            )
        elif method == 'linear_operator':  # linear operator support only cpu
            # convert measure to function
            if isinstance(measure, list):
                obs = lambda t, state: [_np.trace(obsmat @ state) for obsmat in measure]
            else:
                obs = measure
            # ----------- main ------------
            evolve_engine = EvolveEngine(
                self, inistate.flatten(), ts=tlist, normalize=False, ttype='imag-time', traceA=self.trace 
            )
            res = []
            for t in tqdm(tlist, ascii=True):
                state = evolve_engine.run()
                res.append(obs(t, state.reshape(d,d)))
            # ----------- end main ------------
            return _np.real_if_close(res)    
        else:
            if isinstance(measure, list):
                obs = lambda t, state: [_np.trace(obsmat @ state) for obsmat in measure]
            else:
                obs = measure
            from scipy.integrate import solve_ivp
            t_cur = 0
            state_cur = inistate.flatten().astype(_np.complex128)
            res = []
            for t in tqdm(tlist, ascii=True):
                sol = solve_ivp(
                    lambda t, rho: self @ rho, 
                    (0, t - t_cur), state_cur, t_eval=[t - t_cur],
                    method=method, rtol=1e-9, atol=1e-12, **kwargs
                )
                t_cur = t
                state_cur = sol.y.flatten()
                res.append(obs(t, state_cur.reshape(d,d)))
            return _np.real_if_close(res)

    def steady_state(self, method:Literal['direct'] = 'direct'):
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

    def trajectory_measure(
        self, 
        inistate:_np.ndarray, 
        tlist:list|_np.ndarray, 
        measure:Callable|_np.ndarray,
        method:Literal['cpu_mul', 'gpu_mul', 'linear_operator',
                    'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']='cpu_mul',
        **kwargs
    ):
        pass