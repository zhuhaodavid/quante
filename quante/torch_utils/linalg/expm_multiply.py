# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-05 10:43:57
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-15 00:52:31

# 下面的代码来自 scipy.sparse.linalg._expm_multiple
# 有一些改动

"""Compute the action of the matrix exponential."""
from warnings import warn

import numpy as np

import scipy.linalg  # type: ignore
import scipy.sparse.linalg  # type: ignore
import torch as tc
from ..sparse import trace, norm, eye
from typing import Union
from functools import lru_cache

__all__ = ['expm_multiply', 'evolve_engine', 'expm']

def expm(A:tc.Tensor, c: Union[float, complex] = 1.0) -> tc.Tensor:
    """Exponential Matrix, Hermitian matrix can be accelerated
    """
    is_herm = tc.allclose(A, A.conj().T)
    if is_herm:
        eigenvalues, eigenstates = tc.linalg.eigh(A)
        if tc.isreal(A).all() and tc.isreal(c):
            new_eigenvalues = tc.exp(eigenvalues * c)
        else:
            new_eigenvalues = tc.exp(eigenvalues * c).astype(complex)
        return (eigenstates * new_eigenvalues) @ eigenstates.conj().transpose()
    else:
        return tc.matrix_exp(c*A)


def expm_multiply(A:tc.Tensor, B:tc.Tensor, scale=1.0, start=None, stop=None, num=None,
                  endpoint=None, traceA=None, norm1A=None, hasshifted=False, herm=False):
    """
    计算 `exp(matvec).dot(psi0)` 或 `exp(- 1j * matvec).dot(psi0)`
    
    (分别通过 `scale=1.` 和 `scale=-1.j` 来实现。)
    
    A 可以是密集矩阵，也可以是稀疏矩阵
    `start` `stop` `endpoint` 与 numpy.linspace 的参数兼容
    
    如果不给 `traceA`, `norm1A`, `hasshifted` 将会尝试使用 cupy 计算，如果没有 cupy，将会使用 scipy.sparse.linalg.norm 计算
    为避免 GPU, CPU 之间的数据传输，最好安装 cupy，或者完成 shift 后给入函数（具体做法参考 ..linalg.evolve.py 中的做法
    
    更详细的Example，可以参考 example 文件夹下的 `evolve.ipynb` 文件。
    
    """
    assert isinstance(A, tc.Tensor) and isinstance(B, tc.Tensor)
    assert A.device == B.device, "A 和 B 应当在同一个设备上"
    if all(arg is None for arg in (start, stop, num, endpoint)):
        X, s = _expm_multiply_simple(A, B, scale, traceA=traceA, herm=herm, norm1A=norm1A, hasshifted=hasshifted)
    else:
        X, status, s = _expm_multiply_interval(A, B, scale, start, stop, num,
                                            endpoint, traceA=traceA, herm=herm, norm1A=norm1A, hasshifted=hasshifted)
    return X

def evolve_engine(A:tc.Tensor, scale=1., n0=1, herm=False):
    """
    用 evolve engine 的形式，用法如下：
    
    Examples
    --------
    >>> import quante as qt
    >>> from quante.torch_utils.linalg import evolve_engine, to_csr
    >>> import torch as tc
    >>> L = 10
    >>> ham = qt.generate.operas.heisenberg_operator(L)
    >>> basis = qt.generate.basis.spin_basis(L)
    >>> tcmat = to_csr(ham.to_matrix(basis, sparse=True), device='cuda')
    >>> tcstate = tc.tensor(qt.generate.state.random(basis.Ns, seed=42), device='cuda')
    >>> eg = evolve_engine(tcmat, scale=-1j, herm=True)
    >>> for i in range(10):
    >>>     nextstate = eg(tcstate)
    >>>     assert tc.allclose(nextstate, tc.matrix_exp((-1j)*tcmat.to_dense()) @ tcstate)
    >>>     tcstate = nextstate
    """
    assert scale == 1.0 or scale == -1j, "scale 应当为 1.0 或 -1j"
    
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    n = A.shape[0]
    u_d = 2**-53
    tol = u_d
    traceA = trace(A)
    mu = traceA / n
    if A.is_sparse_csr:
        A = A + (-mu) * eye(n, device=A.device)
    else:
        A = A + (-mu) * tc.eye(n, device=A.device)
    A_1_norm = norm(A, ord=1)
    if A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(A, A_1_norm=A_1_norm, ell=ell, herm=herm)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)

    tc.cuda.empty_cache()
    
    if A.is_complex() and scale == -1j:
        A = (-1j) * A
        scale = 1.0
    
    def _engine(B):
        F = B.clone()
        if n0 == 1:
            assert B.ndim == 1 or B.shape[1] == 1
        else:
            assert B.shape[1] == n0
        
        _expm_multiply_simple_core(A, B, scale, F, 1.0, mu, m_star, s, tol)
        return F
    
    return _engine
    

class EvolveEngine:
    def __init__(self, ham, init_state, ts, device='cuda'):
        if init_state.ndim == 1:
            self.psi = init_state.reshape(-1, 1).to(dtype=tc.complex128)
        else:
            self.psi = init_state.to(dtype=tc.complex128)
        self.csr_mt = ham
        self.dts = np.diff(ts)
        self.dts = np.insert(self.dts, 0, ts[0])
        self.evolved_time = 0
        self.cur_step = 0
    
    @lru_cache(maxsize=None)
    def get_evolve_engine(self, dt):
        return evolve_engine(dt * self.csr_mt, scale=-1j)

    def run(self):
        try:
            dt = self.dts[self.cur_step]
        except:
            warn(
                f"t {self.evolved_time} has been reached, dt = {self.dts[-1]} will be used"
            )
            dt = self.dts[-1]
        self.cur_step += 1
        if dt != 0:
            ee = self.get_evolve_engine(round(dt,14))
            self.psi = ee(self.psi)
            self.evolved_time += dt
            


def _expm_multiply_simple(A, B, scale, traceA=None, norm1A=None, hasshifted=False, herm=False):
    """
    Notes
    -----
    This is algorithm (3.2) in Al-Mohy and Higham (2011).
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    if traceA is None:
        traceA = trace(A)
    mu = traceA / n
    if not hasshifted and abs(mu) > 1e-10:
        if A.is_sparse_csr:
            A = A + (-mu) * eye(n, device=A.device)
        else:
            A = A + (-mu) * tc.eye(n, device=A.device)
    A_1_norm = norm(A, ord=1) if norm1A is None else norm1A
    if A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(A, A_1_norm=A_1_norm, ell=ell, herm=herm)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)

    tc.cuda.empty_cache()
    F = B.clone()
    _expm_multiply_simple_core(A, B, scale, F, 1.0, mu, m_star, s, tol)
    return F, s
    

def _expm_multiply_simple_core(A, B, scale, F, t, mu, m_star, s, tol=None):
    """
    A helper function.
    """
    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    tmp = t*mu*scale / s
    eta = tc.exp(tmp) if isinstance(tmp, tc.Tensor) else np.exp(tmp)
    tmp1 = B.clone()
    tmp2 = B.clone()
    for i in range(s):
        c1 = tc.norm(B, tc.inf)
        for j in range(m_star):
            coeff = t / float(s*(j+1))
            if scale.real == 0 and not A.is_complex():
                # B =  - 1j * (A @ B.real) + A @ B.imag
                # B.mul_(coeff)
                tc.matmul(A, tmp1.imag, out=tmp2.real).mul_(coeff)
                tc.matmul(A, tmp1.real, out=tmp2.imag).mul_(-coeff)
                # tmp2.mul_()
            elif scale.real == 0:
                # B =  (- 1j) * (A @ B)
                # B.mul_(coeff)
                tc.matmul(A, tmp1, out=tmp2).mul_(-1j*coeff)
            else:
                tc.matmul(A, tmp1, out=tmp2)
                tmp2.mul_(coeff)
                # B = A.matmul(B)
                # B.mul_(coeff)
            c2 = tc.norm(tmp2, tc.inf)
            F.add_(tmp2)
            if c1 + c2 <= tol * tc.norm(F, tc.inf):
                break
            c1 = c2
            tmp1, tmp2 = tmp2, tmp1
        F.mul_(eta)
        tmp1.copy_(F)


# This table helps to compute bounds.
# They seem to have been difficult to calculate, involving symbolic
# manipulation of equations, followed by numerical root finding.
_theta = {
        # The first 30 values are from table A.3 of Computing Matrix Functions.
        1: 2.29e-16,
        2: 2.58e-8,
        3: 1.39e-5,
        4: 3.40e-4,
        5: 2.40e-3,
        6: 9.07e-3,
        7: 2.38e-2,
        8: 5.00e-2,
        9: 8.96e-2,
        10: 1.44e-1,
        # 11
        11: 2.14e-1,
        12: 3.00e-1,
        13: 4.00e-1,
        14: 5.14e-1,
        15: 6.41e-1,
        16: 7.81e-1,
        17: 9.31e-1,
        18: 1.09,
        19: 1.26,
        20: 1.44,
        # 21
        21: 1.62,
        22: 1.82,
        23: 2.01,
        24: 2.22,
        25: 2.43,
        26: 2.64,
        27: 2.86,
        28: 3.08,
        29: 3.31,
        30: 3.54,
        # The rest are from table 3.1 of
        # Computing the Action of the Matrix Exponential.
        35: 4.7,
        40: 6.0,
        45: 7.2,
        50: 8.5,
        55: 9.9,
        }



class LazyOperatorNormInfo:
    """
    Information about an operator is lazily computed.

    The information includes the exact 1-norm of the operator,
    in addition to estimates of 1-norms of powers of the operator.
    This uses the notation of Computing the Action (2011).
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """

    def __init__(self, A, A_1_norm=None, ell=2, scale=1, herm=False):
        """
        Provide the operator and some norm-related information.

        Parameters
        ----------
        A : linear operator
            The operator of interest.
        A_1_norm : float, optional
            The exact 1-norm of A.
        ell : int, optional
            A technical parameter controlling norm estimation quality.
        scale : int, optional
            If specified, return the norms of scale*A instead of A.

        """
        self._A = A
        self._A_1_norm = A_1_norm
        self._ell = ell
        self._d = {}
        self._scale = scale
        self.herm = herm

    def set_scale(self,scale):
        """
        Set the scale parameter.
        """
        self._scale = scale

    def onenorm(self):
        """
        Compute the exact 1-norm.
        """
        if self._A_1_norm is None:
            self._A_1_norm = norm(self._A, ord=1)
        return self._scale*self._A_1_norm

    def d(self, p):
        """
        Lazily estimate :math:`d_p(A) ~= || A^p ||^(1/p)` where :math:`||.||` is the 1-norm.
        """
        if p not in self._d:
            est = _onenormest_matrix_power(self._A, p, t=self._ell, herm=self.herm)
            self._d[p] = est ** (1.0 / p)
        return self._scale*self._d[p]

    def alpha(self, p):
        """
        Lazily compute max(d(p), d(p+1)).
        """
        return max(self.d(p), self.d(p+1))

def _compute_cost_div_m(m, p, norm_info):
    """
    A helper function for computing bounds.

    This is equation (3.10).
    It measures cost in terms of the number of required matrix products.

    Parameters
    ----------
    m : int
        A valid key of _theta.
    p : int
        A matrix power.
    norm_info : LazyOperatorNormInfo
        Information about 1-norms of related operators.

    Returns
    -------
    cost_div_m : int
        Required number of matrix products divided by m.

    """
    nm = norm_info.alpha(p)
    return tc.ceil(nm / _theta[m]).to(dtype=tc.int64)


def _compute_p_max(m_max):
    """
    Compute the largest positive integer p such that p*(p-1) <= m_max + 1.

    Do this in a slightly dumb way, but safe and not too slow.

    Parameters
    ----------
    m_max : int
        A count related to bounds.

    """
    sqrt_m_max = np.sqrt(m_max)
    p_low = int(np.floor(sqrt_m_max))
    p_high = int(np.ceil(sqrt_m_max + 1))
    return max(p for p in range(p_low, p_high+1) if p*(p-1) <= m_max + 1)


def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    norm_info : LazyOperatorNormInfo
        Information about norms of certain linear operators of interest.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    tol : float
        Expected to be
        :math:`2^{-24}` for single precision or
        :math:`2^{-53}` for double precision.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.

    Returns
    -------
    best_m : int
        Related to bounds for error control.
    best_s : int
        Amount of scaling.

    Notes
    -----
    This is code fragment (3.1) in Al-Mohy and Higham (2011).
    The discussion of default values for m_max and ell
    is given between the definitions of equation (3.11)
    and the definition of equation (3.12).

    """
    if ell < 1:
        raise ValueError('expected ell to be a positive integer')
    best_m = None
    best_s = None
    if _condition_3_13(norm_info.onenorm(), n0, m_max, ell):
        for m, theta in _theta.items():
            tmp = norm_info.onenorm()
            if isinstance(tmp, tc.Tensor):
                s = int(tc.ceil(tmp / theta).cpu().numpy().item())
            else:
                s = int(np.ceil(tmp / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m = m
                best_s = s
    else:
        # Equation (3.11).
        for p in range(2, _compute_p_max(m_max) + 1):
            for m in range(p*(p-1)-1, m_max+1):
                if m in _theta:
                    s = int(_compute_cost_div_m(m, p, norm_info).cpu().numpy().item())
                    if best_m is None or m * s < best_m * best_s:
                        best_m = m
                        best_s = s
        best_s = max(best_s, 1)
    return best_m, best_s


def _condition_3_13(A_1_norm, n0, m_max, ell):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    A_1_norm : float
        The precomputed 1-norm of A.
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
    m_max : int
        A value related to a bound.
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.

    Returns
    -------
    value : bool
        Indicates whether or not the condition has been met.

    Notes
    -----
    This is condition (3.13) in Al-Mohy and Higham (2011).

    """

    # This is the rhs of equation (3.12).
    p_max = _compute_p_max(m_max)
    a = 2 * ell * p_max * (p_max + 3)

    # Evaluate the condition (3.13).
    b = _theta[m_max] / float(n0 * m_max)
    return A_1_norm <= a * b


def _expm_multiply_interval(A, B, scale, start=None, stop=None, num=None,
                            endpoint=None, traceA=None,
                            status_only=False, herm=False, norm1A=None, hasshifted=False):
    """
    Compute the action of the matrix exponential at multiple time points.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable
    endpoint : bool, optional
        If True, `stop` is the last time point. Otherwise, it is not included.
    status_only : bool
        A flag that is set to True for some debugging and testing operations.

    Returns
    -------
    F : ndarray
        :math:`e^{t_k A} B`
    status : int
        An integer status for testing and debugging.

    Notes
    -----
    This is algorithm (5.2) in Al-Mohy and Higham (2011).

    There seems to be a typo, where line 15 of the algorithm should be
    moved to line 6.5 (between lines 6 and 7).

    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))
    assert not isinstance(A, scipy.sparse.linalg.LinearOperator), '这里不支持 LinearOperator'
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2**-53
    tol = u_d
    if traceA is None:
        traceA = trace(A)
    mu = traceA / float(n)

    # Get the linspace samples, attempting to preserve the linspace defaults.
    linspace_kwargs = {'retstep': True}
    if num is not None:
        linspace_kwargs['num'] = num
    if endpoint is not None:
        linspace_kwargs['endpoint'] = endpoint
    samples, step = np.linspace(start, stop, **linspace_kwargs)

    # Convert the linspace output to the notation used by the publication.
    nsamples = len(samples)
    if nsamples < 2:
        raise ValueError('at least two time points are required')
    q = nsamples - 1
    h = step
    t_0 = samples[0]
    t_q = samples[q]

    # Use an ndim=3 shape, such that the last two indices
    # are the ones that may be involved in level 3 BLAS operations.
    t = t_q - t_0
    
    if not hasshifted and abs(mu) > 1e-10:
        if A.is_sparse_csr:
            A = A + (-mu) * eye(n, device=A.device, dtype=A.dtype)
        else:
            A = A - mu * tc.eye(n, device=A.device, dtype=A.dtype)

    A_1_norm = norm(A, ord=1) if norm1A is None else norm1A
    
    ell = 2
    norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell, herm=herm)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    
    tc.cuda.empty_cache()
    
    X_shape = (nsamples,) + B.shape
    X = tc.empty(X_shape, dtype=B.dtype, device=B.device)
    X[0].copy_(B)

    # Compute the expm action up to the initial time point.
    _expm_multiply_simple_core(A, B, scale, X[0], t_0, mu, m_star, s)
    
    tc.cuda.empty_cache()
    # Compute the expm action at the rest of the time points.
    if q <= s:
        if status_only:
            return 0
        else:
            X, status = _expm_multiply_interval_core_0(A, X,
                    scale, h, mu, q, norm_info, tol, ell,n0)
    elif not (q % s):
        if status_only:
            return 1
        else:
            warn(f"Using the q % s == {q} % {s} == 0 case, for best perfermance, change num slightly.", stacklevel=3)
            X, status = _expm_multiply_interval_core_1(A, X,
                    scale, h, mu, m_star, s, q, tol)
    elif (q % s):
        if status_only:
            return 2
        else:
            X, status = _expm_multiply_interval_core_2(A, X,
                    scale, h, mu, m_star, s, q, tol)
    else:
        raise Exception('internal error')
    
    return X, status, s


def _expm_multiply_interval_core_0(A, X, scale, h, mu, q, norm_info, tol, ell, n0):
    """
    A helper function, for the case q <= s.
    """

    # Compute the new values of m_star and s which should be applied
    # over intervals of size t/q

    if norm_info.onenorm() == 0:
        m_star, s = 0, 1
    else:
        norm_info.set_scale(1./q)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
        norm_info.set_scale(1)

    for k in range(q):
        X[k+1] = X[k]
        _expm_multiply_simple_core(A, X[k], scale, X[k+1], h, mu, m_star, s)
    return X, 0


def _expm_multiply_interval_core_1(A, X, scale, h, mu, m_star, s, q, tol):
    """
    A helper function, for the case q > s and q % s == 0.
    """
    d = q // s
    input_shape = X.shape[1:]
    K_shape = (m_star + 1, ) + input_shape
    K = tc.empty(K_shape, dtype=X.dtype, device=X.device)
        
    for i in range(s):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        for k in range(1, d+1):
            F = K[0].clone()
            c1 = tc.norm(F, tc.inf)
            for p in range(1, m_star+1):
                
                if p > high_p:
                    # K[p] = h * A.dot(K[p-1]) / float(p)   #!! main
                    
                    if scale.real == 0 and not A.is_complex():
                        # K[p] = A.matmul(K[p-1].imag) - 1j * A.matmul(K[p-1].real)
                        tc.matmul(A, K[p-1].imag, out=K[p].real)
                        tc.matmul(A, K[p-1].real, out=K[p].imag).mul_(-1)
                        K[p].mul_(h/float(p))
                        
                    elif scale.real == 0:
                        # K[p] = (-1j) * A.matmul(K[p-1])
                        tc.matmul(A, K[p-1], out=K[p])
                        K[p].mul_(-1j*h/float(p))
                    else:
                        # K[p] = A.matmul(K[p-1])
                        tc.matmul(A, K[p-1], out=K[p])
                        K[p].mul_(h/float(p))
                    
                    # tmpKp = K[p].reshape(-1)
                    # prodscale(tmpKp, h/float(p))
                    
                coeff = float(k**p)
                
                # F += coeff * K[p]  #!! main
                F.add_(K[p], alpha=coeff)
                
                inf_norm_K_p_1 = tc.norm(K[p], tc.inf)
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * tc.norm(F, tc.inf):
                    break
                c1 = c2
            tmp = k*h*mu*scale
            eta = tc.exp(tmp) if isinstance(tmp, tc.Tensor) else np.exp(tmp)
            X[k + i*d] = eta * F
    return X, 1


def _expm_multiply_interval_core_2(A, X, scale, h, mu, m_star, s, q, tol):
    """
    A helper function, for the case q > s and q % s > 0.
    """
    d = q // s
    j = q // d
    r = q - d * j
    input_shape = X.shape[1:]
    K_shape = (m_star + 1, ) + input_shape
    K = tc.empty(K_shape, dtype=X.dtype, device=X.device)
    F = K[0].clone()
    for i in range(j + 1):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        if i < j:
            effective_d = d
        else:
            effective_d = r
        for k in range(1, effective_d+1):
            F.copy_(K[0])
            c1 = tc.norm(F, tc.inf)
            for p in range(1, m_star+1):
                if p == high_p + 1:
                    # K[p] = h * A.dot(K[p-1]) / float(p)   #!! main
                    if scale.real == 0 and not A.is_complex():
                        # K[p] = A.matmul(K[p-1].imag) - 1j * A.matmul(K[p-1].real)
                        tc.matmul(A, K[p-1].imag, out=K[p].real)
                        tc.matmul(A, K[p-1].real, out=K[p].imag).mul_(-1)
                        K[p].mul_(h/float(p))
                        
                    elif scale.real == 0:
                        # K[p] = (-1j) * A.matmul(K[p-1])
                        tc.matmul(A, K[p-1], out=K[p])
                        K[p].mul_(-1j*h/float(p))
                    else:
                        # K[p] = A.matmul(K[p-1])
                        tc.matmul(A, K[p-1], out=K[p])
                        K[p].mul_(h/float(p))
                    
                    # tmpKp = K[p].reshape(-1)
                    # prodscale(tmpKp, h/float(p))
                    
                    high_p = p
                coeff = float(pow(k, p))
                
                # F += coeff * K[p]  #!! main
                F.add_(K[p], alpha=coeff)
                
                inf_norm_K_p_1 = tc.norm(K[p], tc.inf)
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * tc.norm(F, tc.inf):
                    break
                c1 = c2
            tmp = k*h*mu*scale
            eta = tc.exp(tmp) if isinstance(tmp, tc.Tensor) else np.exp(tmp)
            X[k + i*d] = eta * F
    return X, 2


# ========================== 下面的代码来着 scipy.sparse.linalg._onenormest ==========================

def _onenormest_matrix_power(A, p=1, t=2, itmax=5, compute_v=False, compute_w=False, herm=False):
    # Check the input.
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected the operator to act like a square matrix')

    # If the operator size is small compared to t,
    # then it is easier to compute the exact norm.
    # Otherwise estimate the norm.
    n = A.shape[1]
    if t >= n:
        tmp = A.clone()
        for i in range(p-1):
            tmp = tc.matmul(A, tmp)
        est = norm(tmp, ord=1)
    else:
        AH = A if herm else A.H
        est, v, w, nmults, nresamples = _onenormest_core(A, AH, p, t, itmax)
    # Report the norm estimate along with some certificates of the estimate.
    if compute_v or compute_w:
        result = (est,)
        if compute_v:
            result += (v,)
        if compute_w:
            result += (w,)
        return result
    else:
        return est

def elementary_vector(n, i):
    v = np.zeros(n, dtype=float)
    v[i] = 1
    return v

def resample_column(i, X):
    X[:, i] = tc.randint(0, 2, size=(X.shape[0],), device=X.device)*2 - 1

def column_needs_resampling(i, X, Y=None):
    # column i of X needs resampling if either
    # it is parallel to a previous column of X or
    # it is parallel to a column of Y
    n, t = X.shape
    v = X[:, i]
    if any(vectors_are_parallel(v, X[:, j]) for j in range(i)):
        return True
    if Y is not None:
        if any(vectors_are_parallel(v, w) for w in Y.T):
            return True
    return False

def vectors_are_parallel(v, w):
    # Columns are considered parallel when they are equal or negative.
    # Entries are required to be in {-1, 1},
    # which guarantees that the magnitudes of the vectors are identical.
    if v.ndim != 1 or v.shape != w.shape:
        raise ValueError('expected conformant vectors with entries in {-1,1}')
    n = v.shape[0]
    return v.to(dtype=tc.complex128).matmul(w.to(dtype=tc.complex128)) == n


def _sum_abs_axis0(X):
    block_size = 2**20
    r = None
    for j in range(0, X.shape[0], block_size):
        y = tc.sum(tc.abs(X[j:j+block_size]), axis=0)
        if r is None:
            r = y
        else:
            r += y
    return r


# @_blocked_elementwise_tc
def sign_round_up(X):
    """
    This should do the right thing for both real and complex matrices.

    From Higham and Tisseur:
    "Everything in this section remains valid for complex matrices
    provided that sign(A) is redefined as the matrix (aij / |aij|)
    (and sign(0) = 1) transposes are replaced by conjugate transposes."

    """
    Y = X.clone()
    Y[Y == 0] = 1
    Y.div_(tc.abs(Y))
    return Y

def every_col_of_X_is_parallel_to_a_col_of_Y(X, Y):
    for v in X.T:
        if not any(vectors_are_parallel(v, w) for w in Y.T):
            return False
    return True

# @_blocked_elementwise
def _max_abs_axis1(X):
    return tc.max(tc.abs(X), axis=1).values

def elementary_vector(n, i):
    v = tc.zeros(n, dtype=tc.float64, device='cuda')
    v[i] = 1
    return v

def tcmatmul(A:tc.Tensor, S, p):
    res1 = S.clone().to(A.dtype)
    res2 = S.clone().to(A.dtype)
    for i in range(p):
        tc.matmul(A, res1, out=res2)
        res1, res2 = res2, res1
    return res1


def _onenormest_core(A, AT, p, t, itmax):
    # This function is a more or less direct translation
    # of Algorithm 2.4 from the Higham and Tisseur (2000) paper.
    
    if itmax < 2:
        raise ValueError('at least two iterations are required')
    if t < 1:
        raise ValueError('at least one column is required')
    n = A.shape[0]
    if t >= n:
        raise ValueError('t should be smaller than the order of A')
    # Track the number of big*small matrix multiplications
    # and the number of resamplings.
    nmults = 0
    nresamples = 0
    # "We now explain our choice of starting matrix.  We take the first
    # column of X to be the vector of 1s [...] This has the advantage that
    # for a matrix with nonnegative elements the algorithm converges
    # with an exact estimate on the second iteration, and such matrices
    # arise in applications [...]"
    X = tc.ones((n, t), dtype=tc.float64, device=A.device)
    # "The remaining columns are chosen as rand{-1,1},
    # with a check for and correction of parallel columns,
    # exactly as for S in the body of the algorithm."
    if t > 1:
        for i in range(1, t):
            # These are technically initial samples, not resamples,
            # so the resampling count is not incremented.
            resample_column(i, X)
        for i in range(t):
            while column_needs_resampling(i, X):
                resample_column(i, X)
                nresamples += 1
    # "Choose starting matrix X with columns of unit 1-norm."
    X /= float(n)
    # "indices of used unit vectors e_j"
    ind_hist = tc.zeros(0, dtype=tc.int64, device=A.device)
    est_old = tc.tensor(0, dtype=tc.float64, device=A.device)
    S = tc.zeros((n, t), dtype=tc.float64, device=A.device)
    k = 1
    ind = None
    while True:
        Y = tcmatmul(A,X,p)
        nmults += 1
        mags = _sum_abs_axis0(Y)
        est = tc.max(mags)
        best_j = tc.argmax(mags)
        
        if k == 2 or (tc.gt(est, est_old).item()):
            if k >= 2:
                ind_best = ind[best_j]
            w = Y[:, best_j]
        # (1)
        if k >= 2 and est <= est_old:
            est = est_old
            break
        est_old = est
        S_old = S
        if k > itmax:
            break
        S = sign_round_up(Y)
        del Y
        # (2)
        if every_col_of_X_is_parallel_to_a_col_of_Y(S, S_old):
            break
        if t > 1:
            # "Ensure that no column of S is parallel to another column of S
            # or to a column of S_old by replacing columns of S by rand{-1,1}."
            for i in range(t):
                while column_needs_resampling(i, S, S_old):
                    resample_column(i, S)
                    nresamples += 1
        del S_old
        # (3)
        Z = tcmatmul(AT, S, p)
        nmults += 1
        h = _max_abs_axis1(Z)
        del Z
        # tc.cuda.empty_cache()  #!! 如果显存不够可以在这里 释放 Z 的显存
        # (4)
        if k >= 2 and tc.max(h) == h[ind_best]:
            break
        # "Sort h so that h_first >= ... >= h_last
        # and re-order ind correspondingly."
        #
        # Later on, we will need at most t+len(ind_hist) largest
        # entries, so drop the rest
        ind = tc.argsort(h, descending=True)[:t+len(ind_hist)].clone()
        del h
        if t > 1:
            # (5)
            # Break if the most promising t vectors have been visited already.
            if tc.isin(ind[:t], ind_hist).all():
                break
            # Put the most promising unvisited vectors at the front of the list
            # and put the visited vectors at the end of the list.
            # Preserve the order of the indices induced by the ordering of h.
            seen = tc.isin(ind, ind_hist)
            ind = tc.concatenate((ind[~seen], ind[seen]))
        for j in range(t):
            X[:, j] = elementary_vector(n, ind[j])

        new_ind = ind[:t][~tc.isin(ind[:t], ind_hist)]
        ind_hist = tc.concatenate((ind_hist, new_ind))
        k += 1
    v = elementary_vector(n, ind_best)
    return est, v, w, nmults, nresamples
