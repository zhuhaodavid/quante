# # -*- coding: utf-8 -*-
# # @Author: hzhu
# # @Date:   2023-10-22 17:13:49
# # @Last Modified by:   hzhu
# # @Last Modified time: 2025-05-14 22:54:08

from scipy import sparse as sps
from scipy.special import jv
from scipy.sparse.linalg import LinearOperator, expm_multiply

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

def expm_multiply(mat:Union[_np.ndarray, Callable[[_np.ndarray], _np.ndarray]], psi0:_np.ndarray, scale=1.0, *, start=None, stop=None, num=None, endpoint=None, traceA=None, herm=False, device=None) -> _np.ndarray:
    """
    计算 `exp(matvec).dot(psi0)` 或 `exp(- 1j * matvec).dot(psi0)`
    
    (分别通过 `scale=1.` 和 `scale=-1.j` 来实现。)
    
    matvec 可以是：
    
    - 函数：
        此时需要给定 `herm` 参数，`herm=True`，表示 `matvec` 是厄密算符，那么 `rmatvec = matvec`
        如果不是厄密算符，那么就需要用 `herm` 传入伴随算符，`rmatvec = herm`
        同时需要传入 `traceA`（`matvec` 的迹），否则内部会估计，影响结果精确度。
        
    - 矩阵（ `numpy` 或 `scipy.sparse` 格式，不能是 `torch` 格式）：
        支持稀疏矩阵和密集矩阵，此时通过 `scale=-1.j` 可以在 `matvec` 是实矩阵的时候节约内存，同时加快计算。
        此时如果指定 `herm=True`，可以稍微加快计算。
    
    `usecuda=True` 表示使用 torch GPU 加速计算，否则使用 CPU numpy 并行计算。或者使用 `torch_utils.linalg.expm_multiply` 函数。
    
    `start` `stop` `endpoint` 与 numpy.linspace 的参数兼容
    
    Notes
    ------
    `num` 在比较小的时候，与逐步迭代的效果相同
    `num` 在比较大的时候，会使用不同的算符，当 `num` 取某些特定值 `n*s+1`（`n`是整数，`s` 是取决于矩阵模的整数的时候），效率会有一些降低，此时会给出警告，不影响结果。可以通过稍微改变 `num` 提高效率
    
    输出 ndarray `expm_A_B` 的形状可以是 1、2 或 3。

    - 如果在单个时间点上计算 expm 对单个向量的作用，`ndim` 将是 `1`。
    - 如果在多个时间点上计算 expm 对向量的作用，或者在单个时间点上计算 expm 对矩阵的作用，`ndim` 将是 `2`。
    - 如果在多个时间点上对具有多列的矩阵进行作用，`ndim` 将是 `3`。
    
    如果计算多个时间点，expm_A_B[0] 将始终是 expm 在第一个时间点上的作用，无论作用是对向量还是矩阵。
        
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
    >>> res = qt.linalg.expm_multiply(mat, state, -1.j, start=0, stop=10, num=100)
    >>> res.shape
    (100, 1024, 1)
    
    更详细的Example，可以参考 example 文件夹下的 `evolve.ipynb` 文件。
    """
    assert scale == 1.0 or scale == - 1j, "only scale=1.0 or scale=-1j is supported for now"
    
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
    >>> hammat = self.to_matrix(basis, pauli=pauli, sparse=True)
    >>> evolve_engine = EvolveEngine(hammat, inistate, ts=tlist)
    >>> obsmatlist = [obs.to_matrix(basis, pauli=pauli, sparse=True) for obs in obslist]
    >>> res = [[]*len(obslist)]
    >>> for _ in tlist:
    >>>     evolve_engine.run()
    >>>     state = evolve_engine.psi
    >>>     for i in range(len(obslist)):
    >>>         value = state.conj().reshape(1,-1) @ (obsmatlist[i] @ state)
    >>>         res[i].append(value[0,0])
    >>> return [np.real_if_close(r) for r in res]
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
import numpy as _np

def _change_dtype_CPU(eigenstates: _np.ndarray, initial_state: _np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
    """
    确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    """
    if _np.issubdtype(eigenstates.dtype, _np.complexfloating) or _np.issubdtype(initial_state.dtype, _np.complexfloating):
        eigenstates = eigenstates.astype(_np.complex128)
        initial_state = initial_state.astype(_np.complex128)
    return eigenstates, initial_state

def _cpu_complex_exp_method(times: _np.ndarray, udagger_psi: _np.ndarray, eigenstates: _np.ndarray, eigenvalues:_np.ndarray) -> _np.ndarray:
    """
    使用复指数计算时间演化态。
    """
    times_E = _np.broadcast_to(-1j*times, (len(eigenvalues), len(times))).T * eigenvalues
    exptimes_E = _np.exp(times_E)
    exptimeEpsi = exptimes_E * udagger_psi
    res = eigenstates @ exptimeEpsi.T
    return res

def _cpu_real_dtype_method(times: _np.ndarray, udagger_psi: _np.ndarray, eigenstates: _np.ndarray, eigenvalues:_np.ndarray) -> _np.ndarray:
    """
    使用实部和虚部计算时间演化态。
    """
    times_E = _np.broadcast_to(times, (len(eigenvalues), len(times))).T * eigenvalues
    real_part = _np.cos(times_E) * udagger_psi
    imag_part = _np.sin(times_E) * udagger_psi
    return eigenstates @ real_part.T - 1j * (eigenstates @ imag_part.T)

def _in_CPU(initial_state: _np.ndarray, eigenvalues: _np.ndarray, eigenstates: _np.ndarray, times: _np.ndarray) -> _np.ndarray:
    eigenstates, initial_state = _change_dtype_CPU(eigenstates, initial_state)
    udagger_psi = (eigenstates.T.conj() @ initial_state).reshape(1, -1)
    _method = _cpu_complex_exp_method if _np.issubdtype(eigenstates.dtype, _np.complexfloating) else _cpu_real_dtype_method
    time_evolution_states = _method(times, udagger_psi, eigenstates, eigenvalues)
    return time_evolution_states


# -> GPU
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # 类型检查时，导入 torch
    import torch as _tc

def _data_to_GPU(initial_state, eigenvalues, eigenstates, times, device) -> tuple['_tc.Tensor', '_tc.Tensor', '_tc.Tensor', '_tc.Tensor']:
    """
    将数据从 numpy 数组转换为 GPU 上的 torch.Tensor。
    """
    import torch as _tc
    initial_state = _tc.from_numpy(initial_state).to(device)
    eigenvalues = _tc.from_numpy(eigenvalues).to(device)
    eigenstates = _tc.from_numpy(eigenstates).to(device)
    times = _tc.from_numpy(times).to(device)
    return initial_state, eigenvalues, eigenstates, times

def _change_dtype_GPU(eigenstates: '_tc.Tensor', initial_state: '_tc.Tensor') -> tuple['_tc.Tensor', '_tc.Tensor']:
    """
    确保 eigenstates 和 initial_state 的数据类型为 complex，如果其中之一是 complex128。
    """
    import torch as _tc
    if eigenstates.dtype == _tc.complex128 or initial_state.dtype == _tc.complex128:
        eigenstates = eigenstates.to(_tc.complex128)
        initial_state = initial_state.to(_tc.complex128)
    return eigenstates, initial_state

def _gpu_complex_exp_method(times: '_tc.Tensor', udagger_psi: '_tc.Tensor', eigenstates: '_tc.Tensor', eigenvalues: '_tc.Tensor') -> '_tc.Tensor':
    """
    使用复指数计算时间演化态
    """
    import torch as _tc
    times_E = (-1j*times).unsqueeze(1) * eigenvalues
    exp_timeE = _tc.exp(times_E)
    exp_timeE_psi = exp_timeE * udagger_psi
    time_evolution_states = eigenstates @ exp_timeE_psi.T
    return time_evolution_states

def _gpu_real_dtype_method(times: '_tc.Tensor', udagger_psi: '_tc.Tensor', eigenstates: '_tc.Tensor', eigenvalues: '_tc.Tensor') -> '_tc.Tensor':
    """
    使用实部和虚部计算时间演化态（方法2）。
    """
    import torch as _tc
    times_E = times.unsqueeze(1) * eigenvalues
    real_part = _tc.cos(times_E) * udagger_psi
    imag_part = _tc.sin(times_E) * udagger_psi
    return eigenstates @ real_part.T - 1j * (eigenstates @ imag_part.T)

def _in_GPU(initial_state: '_tc.Tensor', eigenvalues: '_tc.Tensor', eigenstates: '_tc.Tensor', times: '_tc.Tensor', device) -> _np.ndarray:
    """
    在 GPU 上计算初始态在不同时刻的时间演化态。
    """
    import torch as _tc
    initial_state, eigenvalues, eigenstates, times = _data_to_GPU(initial_state, eigenvalues, eigenstates, times, device)
    eigenstates, initial_state = _change_dtype_GPU(eigenstates, initial_state)
    udagger_psi = eigenstates.T.conj() @ initial_state
    _method = _gpu_complex_exp_method if udagger_psi.dtype == _tc.complex128 else _gpu_real_dtype_method
    time_states = _method(times, udagger_psi, eigenstates, eigenvalues)
    return time_states.cpu().numpy()

def get_time_evolution_states_ED(initial_state: _np.ndarray, eigenvalues: _np.ndarray, eigenstates: _np.ndarray, times: _np.ndarray, *, failback_to_CPU: bool = False, device_name='cpu') -> _np.ndarray:
    """
    基于严格对角化的时间演化
    
    Args:
        initial_state (_np.ndarray): 初始量子态
        eigenvalues (_np.ndarray): 哈密顿量本征值
        eigenstates (_np.ndarray): 哈密顿量本征态
        times (_np.ndarray): 时间列表
    
    Returns:
        _np.ndarray: 时间演化量子态矩阵
        
    Examples
    --------
    >>> import quante as qt
    >>> L = 10
    >>> mat = qt.generate.matrix.heisenberg_matrix(L)
    >>> state = qt.generate.state.random(mat.shape[0], seed=42)
    >>> eigresult = qt.linalg.eigh(mat)
    >>> times = np.linspace(0,10,100)
    >>> res = qt.linalg.get_time_evolution_states_ED(state,*eigresult,times)
    >>> res.shape
    (1024, 100)
    """
    initial_state = _np.squeeze(initial_state)
    try:
        import torch as tc
        device = tc.device(device_name)
        time_states = _in_GPU(initial_state, eigenvalues, eigenstates, times, device)
    except Exception as e:
        if not failback_to_CPU:
            raise e
        time_states = _in_CPU(initial_state, eigenvalues, eigenstates, times)
    return time_states

# =============================================
# measure
# ==============================================
def evolve_and_measure(
    matrix:sps.csr_array,
    inistate:_np.ndarray,
    tlist:_np.ndarray,
    measure:list[sps.csr_array] | Callable[[_np.ndarray], _np.ndarray],
    normalize:bool = False,
    ttype:Literal['real-time', 'imag-time'] = 'real-time',
    method:Literal['auto', 'eig', 'gpu_mul', 'cpu_mul'] = 'auto',
):
    """计算在不同时间点的测量值

    Parameters
    ----------
    matrix : sps.csr_array
        要求解的哈密顿量矩阵
    inistate : _np.ndarray
        初始量子态（向量）
    tlist : _np.ndarray
        时间列表
    measure : list[sps.csr_array] | Callable[[_np.ndarray], _np.ndarray]
        - `measure` 是稀疏矩阵列表：计算每个测量算符在不同时间点的测量值
        - `measure` 是函数：计算函数在不同时间点的测量值，体现在返回值的第二及之后的指标上
    normalize : bool, optional
        是否在每次演化之后进行归一化, by default False
    type : str, optional
        根据 `type` 选择实时间演化和虚时间演化：
        - `type='real-time'`: 实时间演化使用 `exp(-1j * H * t)`
        - `type='imag-time'`: 虚时间演化使用 `exp(H * t)`
        by default 'real-time'
    method : str, optional
        - `method='auto'`: 自动选择计算方法，首先尝试调用 GPU 方法，如果失败则使用 CPU 方法
        - `method='eig'`: 使用严格对角化方法计算时间演化，只能处理厄密矩阵、时间演化
        - `method='gpu_mul'`: 使用 GPU 方法计算时间演化
        - `method='cpu_mul'`: 使用 CPU 方法计算时间演化
        by default 'auto'

    Returns
    -------
    _np.ndarray
        返回一个多维数组，第一维是时间点，之后的维数含义由 `measure` 决定
    """
    assert sps.issparse(matrix), f"matrix should be sparse array not {type(matrix)}"
    assert method in ['auto', 'eig', 'gpu_mul', 'cpu_mul'], f"method should be one of ['auto', 'eig', 'gpu_mul', 'cpu_mul'] not {method}"
    if method == 'eig':
        ###################################################################################
        # Diagonalize
        ###################################################################################
        assert ttype == 'real-time', "only real-time is supported for eig method"
        mat = matrix.toarray()
        if _np.allclose(mat, mat.T.conj()):
            # ----------- main ------------
            engres = _np.linalg.eigh(mat)
            evalstate = get_time_evolution_states_ED(inistate, *engres, tlist, failback_to_CPU=True)
            if normalize:
                evalstate /= _np.linalg.norm(evalstate, ord=2, axis=0)
            # ----------- end main ------------

            from .operations import observe_states
            if isinstance(measure, list):
                return _np.real_if_close([observe_states(evalstate, obs.toarray()) for obs in measure]).T
            else:
                return _np.real_if_close([measure(evalstate[:, i]) for i in range(len(tlist))])
        else:
            raise ValueError("matrix should be hermitian, try method = 'gpu_mul' or 'cpu_mul'")
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
                obs = lambda state: [(state.conj().reshape(1,-1) @ (obsmat @ state).reshape(-1,1)).item() for obsmat in obsmatlist]
            else:
                obs = measure

            # ----------- main ------------
            hammat0 = to_csr(matrix, device=device)
            inistate = totc(inistate, device=device)
            evolve_engine = tcEvolveEngine(hammat0, inistate, ts=tlist, normalize=normalize, ttype=ttype)
            res = []
            for _ in tqdm(tlist, ascii=True):
                state = evolve_engine.run()
                res.append(obs(state))
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
            obs = lambda state: [state.conj().reshape(-1) @ (obsmat @ state).reshape(-1) for obsmat in measure]
        else:
            obs = measure

        # ----------- main ------------
        evolve_engine = EvolveEngine(matrix, inistate, ts=tlist, normalize=normalize, ttype=ttype)
        res = []
        for _ in tqdm(tlist, ascii=True):
            state = evolve_engine.run()
            res.append(obs(state))
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
    这是一个 Chebyshev 的原理验证函数。
    如果需要加速，可以考虑将 mat @ xxx 改为使用 gpu torch 来加速。
    对于更大规模的计算，需要考虑使用 petsc，相关的 c++ 程序见 https://github.com/Phyzch/Chebyshev_method
    参数选择时，需要让 (max_eng - min_eng) * t ~ O(1)
    
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
    def __init__(self, ham:sps.csr_array, lindblad_ops:list[sps.csr_array]):
        r"""
        The Liouvillian is given by the following equation:
        
        .. math::
            \mathcal{L}(\rho) = -i [H, \rho] + \sum_{l} L_l \rho L_l^{\dagger} - \frac{1}{2} \sum_{l} (L_l^{\dagger} L_l \rho + \rho L_l^{\dagger} L_l)
        
        where :math:`H` is the Hamiltonian, :math:`L_l` are the Lindblad operators, and :math:`\rho` is the density matrix.
        
        Parameters
        ----------
        ham : sps.csr_array
            The Hamiltonian of the system.
        lindblad_ops : list[sps.csr_array]
            The Lindblad operators of the system.
        """
        assert sps.issparse(ham), "ham must be sparse matrix"
        for lo in lindblad_ops:
            assert sps.issparse(lo), "lindblad_ops must be sparse matrix"
        self.lindblad_ops = lindblad_ops
        self.ham_eff = ham - 1j * sum(lo.conj().T @ lo for lo in lindblad_ops)/2
        self.Ns = ham.shape[0]
        self.dtype = _np.dtype(_np.complex128)
        self.shape = (self.Ns**2, self.Ns**2)
    
    @property
    def trace(self):
        a = 2 * self.Ns * self.ham_eff.trace().imag
        b = sum(abs(lo.trace())**2 for lo in self.lindblad_ops)
        return a + b

    def __call__(self, rho):
        drho_dt = -1j * (self.ham_eff @ rho - rho @ self.ham_eff.conj().T) 
        for lo in self.lindblad_ops:
            drho_dt += lo @ rho @ lo.conj().T 
        return drho_dt
    
    def _matvec(self, rho):
        return self(rho.reshape(self.Ns, self.Ns)).flatten()

    def _rmatvec(self, rho):
        rho = rho.reshape(self.Ns, self.Ns)
        drho_dt = -1j * (self.ham_eff.T @ rho - rho @ self.ham_eff.conj()) 
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
        # non-hermitian part
        nonherm = -1j * (sps.kron(self.ham_eff, eye) - sps.kron(eye, self.ham_eff.conj()))
        # stochastic part
        stochastic = sum(sps.kron(lo, lo.conj()) for lo in self.lindblad_ops)
        return nonherm + stochastic

    def evolve_and_measure(
            self,
            inistate:_np.ndarray,
            tlist:list|_np.ndarray,
            measure:Callable|_np.ndarray,
            method:Literal['cpu_mul', 'gpu_mul', 'linear_operator',
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
        if method in ['cpu_mul', 'gpu_mul']:
            if isinstance(measure, list):
                if method == 'cpu_mul':
                    # convert measure to function
                    obs = lambda state: [_np.trace(obsmat @ state.reshape(d,d)) for obsmat in measure]
                else:
                    from ..torch_utils.utils import totc
                    import torch as tc
                    measure = totc(measure, device='cuda')
                    obs = lambda rho: _np.real_if_close([tc.trace(rho.reshape(d,d) @ n).item() for n in measure])
            else:
                obs = measure
            return evolve_and_measure(
                self.to_matrix(), inistate.flatten(), tlist, measure=obs, 
                method=method, ttype='imag-time', normalize=False
            )
        elif method == 'linear_operator':  # linear operator support only cpu
            # convert measure to function
            if isinstance(measure, list):
                obs = lambda state: [_np.trace(obsmat @ state) for obsmat in measure]
            else:
                obs = measure
            # ----------- main ------------
            evolve_engine = EvolveEngine(
                self, inistate.flatten(), ts=tlist, normalize=False, ttype='imag-time', traceA=self.trace 
            )
            res = []
            for _ in tqdm(tlist, ascii=True):
                state = evolve_engine.run()
                res.append(obs(state.reshape(d,d)))
            # ----------- end main ------------
            return _np.real_if_close(res)    
        else:
            if isinstance(measure, list):
                obs = lambda state: [_np.trace(obsmat @ state) for obsmat in measure]
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
                res.append(obs(state_cur.reshape(d,d)))
            return _np.real_if_close(res)




