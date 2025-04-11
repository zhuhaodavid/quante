# # -*- coding: utf-8 -*-
# # @Author: hzhu
# # @Date:   2023-10-22 17:13:49
# # @Last Modified by:   dzwang
# # @Last Modified time: 2025-03-19 15:36:38

import scipy.sparse.linalg as _spalg
import scipy.sparse as _sparse
import numpy as _np
import warnings as _warnings
from typing import Callable, Union
from functools import lru_cache

__all__ = [
    "EvolveEngine",
    "get_time_evolution_states_ED",
    "expm_multiply"
]

def expm_multiply(mat:Union[_np.ndarray, Callable[[_np.ndarray], _np.ndarray]], psi0:_np.ndarray, scale=1.0, *, start=None, stop=None, num=None, endpoint=None, traceA=None, herm=False, cudadevice=False) -> _np.ndarray:
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
    
    if cudadevice:
        assert isinstance(mat, (_np.ndarray, _sparse.spmatrix, _sparse.sparray)), "cuda only support numpy.ndarray or scipy.sparse matrix"
        
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
                mat = mat - (mat.trace() / mat.shape[0]) * _sparse.eye(mat.shape[0], format='csr')
                norm1A = max(abs(mat).sum(axis=0).flat)
            hasshifted = True
            
        
        from ..torch_utils.linalg.sparse import to_csr
        from ..torch_utils.linalg.expm_multiply import expm_multiply
        import torch as tc
        
        dtype = tc.complex128 if scale == -1j or _np.iscomplexobj(mat) or _np.iscomplexobj(psi0) else tc.float64
        
        res = expm_multiply(tc.tensor(mat, device=cudadevice) if isinstance(mat, _np.ndarray) else to_csr(mat, device=cudadevice), 
                            tc.tensor(psi0, device=cudadevice, dtype=dtype),
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
                lo = _spalg.LinearOperator((dim,dim), matvec=mat, rmatvec=mat, dtype=dtype) # type: ignore
            elif scale == -1j:
                lo = _spalg.LinearOperator((dim,dim), matvec=lambda v: (-1j) * mat(v), rmatvec=lambda v: (1j) * mat(v), dtype=dtype) # type: ignore
        elif callable(herm):
            assert scale == 1.0
            lo = _spalg.LinearOperator((dim,dim), matvec=mat, rmatvec=herm, dtype=dtype) # type: ignore
        else:
            raise ValueError("herm should be 1 for hermitian or -1 for antihermitian or callable")
    else:
        assert isinstance(mat, (_np.ndarray, _sparse.spmatrix, _sparse.sparray)), "cuda only support numpy.ndarray or scipy.sparse matrix"
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
    def __init__(self, ham, init_state, ts):
        if init_state.ndim == 1:
            self.psi = init_state.reshape(-1, 1).astype(_np.complex128)
        else:
            self.psi = init_state.astype(_np.complex128)
        try:
            self.csr_mt = ham.tocsr()
        except:
            self.csr_mt = _sparse.csr_array(ham)
        self.dts = _np.diff(ts)
        self.dts = _np.insert(self.dts, 0, ts[0])
        self.evolved_time = 0
        self.cur_step = 0
    
    @lru_cache(maxsize=None)
    def get_evolve_engine(self, dt):
        from .usenumba.expm_multiply_numba import _evolve_engine
        return _evolve_engine(self.csr_mt, scale=-1j, t=dt)

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
