# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-05 10:43:57
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:46:52

# 下面的代码来自 scipy.sparse.linalg._expm_multiple
# 有一些改动

"""Compute the action of the matrix exponential."""
from warnings import warn

import numpy as np

import scipy.sparse.linalg  # type: ignore
from scipy.linalg._decomp_qr import qr  # type: ignore
from scipy.sparse._sputils import is_pydata_spmatrix  # type: ignore
from scipy.sparse.linalg import aslinearoperator  # type: ignore
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest

from .expm_mul_nb import addself, prodscale, addtwo
from ...matops.sparse_mul import dot_parallel


def _expm_multiply_numba(A, B, scale=1.0, start=None, stop=None, num=None,
                  endpoint=None, traceA=None, herm=False):
    if all(arg is None for arg in (start, stop, num, endpoint)):
        X, s = _expm_multiply_simple(A, B, scale, traceA=traceA, herm=herm)
    else:
        X, status, s = _expm_multiply_interval(A, B, scale, start, stop, num,
                                            endpoint, traceA=traceA, herm=herm)
    return X


def _expm_multiply_simple(A, B, scale, t=1.0, traceA=None, balance=False, herm=False):
    """
    Notes
    -----
    This is algorithm (3.2) in Al-Mohy and Higham (2011).
    """
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))
    ident = _ident_like(A)
    is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)
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
        if is_linear_operator:
            warn("Trace of LinearOperator not available, it will be estimated."
                 " Provide `traceA` to ensure performance.", stacklevel=3)
        # m3=1 is bit arbitrary choice, a more accurate trace (larger m3) might
        # speed up exponential calculation, but trace estimation is more costly
        traceA = traceest(A, m3=1) if is_linear_operator else _trace(A)
    mu = traceA / float(n)
    A = A - mu * ident
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell, herm=herm)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    
    # 尝试使用 GPU 计算
    F = B.copy()
    
    _expm_multiply_simple_core(A, B, scale, F, t, mu, m_star, s, tol, balance)
    
    return F, s

def _evolve_engine(A, scale, t=1.0, n0=1, traceA=None, balance=False, herm=False):
    """
    Notes
    -----
    This is algorithm (3.2) in Al-Mohy and Higham (2011).
    """
    if herm is None:
        herm = False
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    ident = _ident_like(A)
    is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)
    n = A.shape[0]
    u_d = 2**-53
    tol = u_d
    if traceA is None:
        if is_linear_operator:
            warn("Trace of LinearOperator not available, it will be estimated."
                 " Provide `traceA` to ensure performance.", stacklevel=3)
        # m3=1 is bit arbitrary choice, a more accurate trace (larger m3) might
        # speed up exponential calculation, but trace estimation is more costly
        traceA = traceest(A, m3=1) if is_linear_operator else _trace(A)
    mu = traceA / float(n)
    A = A - mu * ident
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell, herm=herm)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
   
    if np.iscomplexobj(A) and scale == -1j:
        A = (-1j) * A
        scale = 1.0 

    def _engine(B):
        F = B.copy()
        if n0 == 1:
            assert B.ndim == 1 or B.shape[1] == 1
        else:
            assert B.shape[1] == n0
        
        _expm_multiply_simple_core(A, B, scale, F, t, mu, m_star, s, tol)
        return F
    
    return _engine  

def _expm_multiply_simple_core(A, B, scale, F, t, mu, m_star, s, tol=None, balance=False):
    """
    A helper function.
    """
    if balance:
        raise NotImplementedError
    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    eta = np.exp(t*mu*scale / float(s))
    for i in range(s):
        c1 = _exact_inf_norm(B)
        for j in range(m_star):
            coeff = t / float(s*(j+1))
            if scale.real == 0 and not np.iscomplexobj(A):
                AB = -1j * dot_parallel(A, B.real) + dot_parallel(A, B.imag)
            elif scale.real == 0:
                AB = -1j * dot_parallel(A, B)
            else:
                AB = dot_parallel(A, B)
            
            tmpAB = AB.reshape(-1)
            prodscale(tmpAB, coeff)
            c2 = _exact_inf_norm(tmpAB)
            tmpF = F.reshape(-1)
            addtwo(tmpF, tmpAB)
            F = tmpF.reshape(F.shape)
            B = tmpAB.reshape(B.shape)
                
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2
        F *= eta
        B = F


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

def _exact_inf_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.issparse(A):
        return max(abs(A).sum(axis=1).flat)
    elif is_pydata_spmatrix(A):
        return max(abs(A).sum(axis=1))
    else:
        return np.linalg.norm(A, np.inf)


def _exact_1_norm(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.issparse(A):
        return max(abs(A).sum(axis=0).flat)
    elif is_pydata_spmatrix(A):
        return max(abs(A).sum(axis=0))
    else:
        return np.linalg.norm(A, 1)


def _trace(A):
    # A compatibility function which should eventually disappear.
    if is_pydata_spmatrix(A):
        return A.to_scipy_sparse().trace()
    else:
        return A.trace()


def traceest(A, m3, seed=None):
    """Estimate `np.trace(A)` using `3*m3` matrix-vector products.

    The result is not deterministic.

    Parameters
    ----------
    A : LinearOperator
        Linear operator whose trace will be estimated. Has to be square.
    m3 : int
        Number of matrix-vector products divided by 3 used to estimate the
        trace.
    seed : optional
        Seed for `numpy.random.default_rng`.
        Can be provided to obtain deterministic results.

    Returns
    -------
    trace : LinearOperator.dtype
        Estimate of the trace

    Notes
    -----
    This is the Hutch++ algorithm given in [1]_.

    References
    ----------
    .. [1] Meyer, Raphael A., Cameron Musco, Christopher Musco, and David P.
       Woodruff. "Hutch++: Optimal Stochastic Trace Estimation." In Symposium
       on Simplicity in Algorithms (SOSA), pp. 142-155. Society for Industrial
       and Applied Mathematics, 2021
       https://doi.org/10.1137/1.9781611976496.16

    """
    rng = np.random.default_rng(seed)
    if len(A.shape) != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("Expected A to be like a square matrix.")
    n = A.shape[-1]
    S = rng.choice([-1.0, +1.0], [n, m3])
    Q, _ = qr(A.matmat(S), overwrite_a=True, mode='economic')
    trQAQ = np.trace(Q.conj().T @ A.matmat(Q))
    G = rng.choice([-1, +1], [n, m3])
    right = G - Q@(Q.conj().T @ G)
    trGAG = np.trace(right.conj().T @ A.matmat(right))
    return trQAQ + trGAG/m3


def _ident_like(A):
    # A compatibility function which should eventually disappear.
    if scipy.sparse.issparse(A):
        # Creates a sparse matrix in dia format
        out = scipy.sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
        if isinstance(A, scipy.sparse.spmatrix):
            return out.asformat(A.format)
        return scipy.sparse.dia_array(out).asformat(A.format)
    # elif is_pydata_spmatrix(A):
    #     import sparse
    #     return sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
    elif isinstance(A, scipy.sparse.linalg.LinearOperator):
        return IdentityOperator(A.shape, dtype=A.dtype)
    else:
        return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)


def _onenormest_matrix_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False, herm=False):
    """
    Efficiently estimate the 1-norm of A^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    #XXX Eventually turn this into an API function in the  _onenormest module,
    #XXX and remove its underscore,
    #XXX but wait until expm_multiply goes into scipy.
    from scipy.sparse.linalg._onenormest import onenormest
    return onenormest(aslinearoperator(A) ** p)

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
            self._A_1_norm = _exact_1_norm(self._A)
        return self._scale*self._A_1_norm

    def d(self, p):
        """
        Lazily estimate :math:`d_p(A) ~= || A^p ||^(1/p)` where :math:`||.||` is the 1-norm.
        """
        if p not in self._d:
            est = _onenormest_matrix_power(self._A, p, self._ell, herm=self.herm)
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
    return int(np.ceil(nm / _theta[m]))


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
            s = int(np.ceil(norm_info.onenorm() / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m = m
                best_s = s
    else:
        # Equation (3.11).
        for p in range(2, _compute_p_max(m_max) + 1):
            for m in range(p*(p-1)-1, m_max+1):
                if m in _theta:
                    s = _compute_cost_div_m(m, p, norm_info)
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
                            endpoint=None, traceA=None, balance=False,
                            status_only=False, herm=False):
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
    balance : bool
        Indicates whether or not to apply balancing.
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
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))
    ident = _ident_like(A)
    is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)
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
        if is_linear_operator:
            warn("Trace of LinearOperator not available, it will be estimated."
                 " Provide `traceA` to ensure performance.", stacklevel=3)
        # m3=5 is bit arbitrary choice, a more accurate trace (larger m3) might
        # speed up exponential calculation, but trace estimation is also costly
        # an educated guess would need to consider the number of time points
        traceA = traceest(A, m3=5) if is_linear_operator else _trace(A)
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
    A = A - mu * ident
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)


    X_shape = (nsamples,) + B.shape
    X = np.empty(X_shape, dtype=np.result_type(A.dtype, B.dtype, float))
    X[0] = B

    ell = 2
    norm_info = LazyOperatorNormInfo(t*A, A_1_norm=t*A_1_norm, ell=ell, herm=herm)
    if t*A_1_norm == 0:
        m_star, s = 0, 1
    else:
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)

    # Compute the expm action up to the initial time point.
    _expm_multiply_simple_core(A, B, scale, X[0], t_0, mu, m_star, s)
    
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
    K = np.empty(K_shape, dtype=X.dtype)
        
    for i in range(s):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        for k in range(1, d+1):
            F = K[0].copy()
            c1 = _exact_inf_norm(F)
            for p in range(1, m_star+1):
                
                if p > high_p:
                    # K[p] = h * A.dot(K[p-1]) / float(p)   #!! main
                    
                    dot_parallel(A,K[p-1],K[p])
                    K[p] *= h/float(p)*scale
                    
                    # tmpKp = K[p].reshape(-1)
                    # prodscale(tmpKp, h/float(p))
                    
                coeff = float(k**p)
                
                # F += coeff * K[p]  #!! main
                tmpF = F.reshape(-1)
                addself(tmpF, K[p].reshape(-1), coeff)
                F = tmpF.reshape(*F.shape)
                
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i*d] = np.exp(k*h*mu*scale) * F
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
    K = np.empty(K_shape, dtype=X.dtype)
        
    for i in range(j + 1):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        if i < j:
            effective_d = d
        else:
            effective_d = r
        for k in range(1, effective_d+1):
            F = K[0].copy()
            c1 = _exact_inf_norm(F)
            for p in range(1, m_star+1):
                if p == high_p + 1:
                    # K[p] = h * A.dot(K[p-1]) / float(p)   #!! main
                    dot_parallel(A,K[p-1],K[p])
                    K[p] *= h/float(p)*scale
                    
                    # tmpKp = K[p].reshape(-1)
                    # prodscale(tmpKp, h/float(p))
                    
                    high_p = p
                coeff = float(pow(k, p))
                
                # F += coeff * K[p]  #!! main
                tmpF = F.reshape(-1)
                addself(tmpF, K[p].reshape(-1), coeff)
                F = tmpF.reshape(*F.shape)
                
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i*d] = np.exp(k*h*mu*scale) * F
    return X, 2

