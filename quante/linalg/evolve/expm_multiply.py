# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 18:31:17
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 18:40:08


from scipy import sparse as sps
from scipy.sparse.linalg import LinearOperator

import numpy as _np
from typing import Callable, Union
from typing import Literal
 
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
            
        
        from ...torch_utils.linalg.sparse import to_csr
        from ...torch_utils.linalg.expm_multiply import expm_multiply
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
    from ..nbfuc.expm_multiply_numba import _expm_multiply_numba
    return _expm_multiply_numba(lo, psi0, scale=scale, start=start, stop=stop, num=num, endpoint=endpoint, traceA=traceA)

