# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 16:50:19
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-21 22:18:45
"""
生成矩阵：(`np.ndarray`)
- `pauli_matrix`
- `heisenberg_matrix`
- `random_matrix`
- 其他的预设矩阵
"""

import functools

import re
import numpy as _np
import scipy.sparse as _sparse

from ..linalg.operations import kron, ikron
from .basis.symmetry.basis_wrapped import _check_spin_number # type: ignore

from typing import Optional, Callable, Union, Iterable
number = Union[int, float, complex]

__all__ = [
    "pauli_matrix",
]

__all__ += [
    "random_matrix",
    "random_orthorgonal_matrix_close_I",
    "random_unitary_matrix_close_I",
    "random_sparse_matrix",
]
    

__all__ += [
    "hadamard_gate",
    "phase_gate",
    "rotation_gate",
    "heisenberg_matrix",
    "random_phase_model",
    "local_hamiltonian_spin_1D",
]



PAULI_MAT = {
    "X": _np.array([[0.,1.],[1.,0.]]),
    "Y": _np.array([[0.,-1.j],[1.j,0.]]),
    "Z": _np.array([[1.,0.],[0.,-1.]]),
    "P": _np.array([[0.,1.],[0.,0.]]),
    "M": _np.array([[0.,0.],[1.,0.]]),
    "I": _np.array([[1.,0.],[0.,1.]]),
    "x": _np.array([[0.,0.5],[0.5,0.]]),
    "y": _np.array([[0.,-0.5j],[0.5j,0.]]),
    "z": _np.array([[0.5,0.],[0.,-0.5]]),
    "p": _np.array([[0.,1.],[0.,0.]]),
    "m": _np.array([[0.,0.],[1.,0.]]),
    "+": _np.array([[0.,1.],[0.,0.]]),
    "-": _np.array([[0.,0.],[1.,0.]]),
    "i": _np.array([[1.,0.],[0.,1.]]),
    "u": _np.array([[1.], [0.]]),
    "d": _np.array([[0.], [1.]]),
}

for k, v in PAULI_MAT.items():
    v.setflags(write=False)

def pauli_matrix(
    stri: str,
    S: Union[str, float, int] = '1/2',
    view: bool = False,
) -> _np.ndarray:
    """
    !! 多个算符的时候 `+`, `-`, `u`, `d` 这四个算符一定要十分小心，**最好不要用这四个算符**
    
    - `xx` -> kron(pauli_x, pauli_x)
    - `x3y3` -> `xxxyyy`
    - `x2y3!5` -> `iixyi`
    - `xy!5` -> `iiixy+iixyi+ixyii+xyiii`
    
    Examples
    --------
    >>> op.pauli_matrix("x")
    >>> op.pauli_matrix("xx+yy")
    >>> op.pauli_matrix("xx+3*yy")

    Generates a Pauli operator matrix based on the input string.

    Parameters
    -----------
        - `stri`: String specifying the Pauli operator configuration.
        - `dtype`: Data type of the resulting matrix (default: np.complex128).
        - `view`: If True, prints the intermediate result (default: False).
    """
    iS: Union[int, float] = _check_spin_number(S)

    # 定义获得单个算符的方法
    if iS == 0.5:
        _single_oper = functools.partial(_pauli_matrix_single)
    else:
        _single_oper = functools.partial(_spin_oper_single, S=iS)

    # 如果长度是 1, 直接返回结果
    if len(stri) == 1:
        res = _single_oper(stri).copy()
    else:
        # 用正则表达式转化为可识别的形式
        evalable_string = _standardlize_string(stri)  
        if view:
            print(evalable_string)
        res = _evaluate_string(evalable_string, _single_oper, kron)  # 运行字符串
        
    return _np.real_if_close(res)
        
def _pauli_matrix_single(stri):
    return PAULI_MAT[stri]

def _spin_oper_single(
    label:str,
    S:Union[float, int]=0.5,
):
    D = 2 * S + 1
    assert D == int(D), f"{S} 不是整数或半整数"
    D = int(D)
    op = _np.zeros((D, D), dtype=complex)
    ms = _np.linspace(S, -S, D)
    label = label.lower()
    if label in {'x', 'y'}:
        for i in range(D - 1):
            c = 0.5 * (S * (S + 1) - (ms[i] * ms[i + 1]))**0.5
            op[i, i + 1] = -1.0j * c if (label == 'y') else c
            op[i + 1, i] = 1.0j * c if (label == 'y') else c
    elif label == 'z':
        for i in range(D):
            op[i, i] = ms[i]
    elif label in {'+', 'p', '-', 'm'}:
        for i in range(D - 1):
            c = (S * (S + 1) - (ms[i] * ms[i + 1]))**0.5
            if label in {'+', 'p'}:
                op[i, i + 1] = c
            else:
                op[i + 1, i] = c
    elif label in {'i', 'I'}:
        _np.fill_diagonal(op, 1.0)
    else:
        raise ValueError(f"Label '{label}'' not understood, should be one of "
                         "``['X', 'Y', 'Z', '+', '-', 'I']``.")

    return op

def _standardlize_string(stri):
    new = re.sub(r"([xyzpmiXYZPMIud]+[0-9]+)+\![0-9]+", _term, stri)  # x1y2!4 -> ixyi
    new = re.sub(r"[xyzpmiXYZPMIud]+\![0-9]+", _shift, new)  # xy!4 -> xy!5 -> iiixy+iixyi+ixyii+xyiii
    new = re.sub(r"[xyzpmiXYZPMIud]+[0-9]+", _duplicate, new)  # x3y3 -> xxxyyy
    new = re.sub("[xyzpmiXYZPMIud]+", _rpmethod, new)
    return new

def _term(match):
    res = match.group()
    res = re.split(r'(\d+)', res)
    n = eval(res[-2])
    out = ['i']*n
    for i in range(0, len(res)-3, 2):
        ind = eval(res[i+1])
        out[ind] = res[i]
    return "".join(out)

def _shift(match):
    res = match.group()
    stri, n = res.split("!")
    n = eval(n)
    res = ""
    for i in range(n-len(stri), -1, -1):
        res += (stri + "i"*i).rjust(n, 'i')
        res += "+"
    return res[:-1]

def _duplicate(match):
    res = match.group()
    res = res[0] * eval(res[1:])
    return res

def _rpmethod(match):
    res = "_kron("
    for xi in match.group():
        res += "_single_oper('" +  xi + "'),"
    return res[:-1] + ")"

def _evaluate_string(evalable_string, _single_oper, _kron):
    assert isinstance(_single_oper, Callable)
    assert isinstance(_kron, Callable)
    return eval(evalable_string)

##################################################
# random matrix
##################################################

def random_matrix(dim, type="simple", seed=None):
    """生成随机矩阵
    
    这个函数本身不支持 njit （因为返回的数据类型不统一）
    如果需要 njit 请使用内部的 _random_simple_matrix 等函数
    
    - type: 
    
        - "simple" -> 每个矩阵元实数高斯分布
        
        - "simple_real" -> 每个矩阵元实数高斯分布
        
        - "GOE"/"symm" -> 实对称矩阵
        
        - "GUE"/"herm" -> Hermite 矩阵
        
        - "CUE"/"unit" -> 幺正矩阵
        
        - "COE"/"orth" -> 正交矩阵(COE)
        
        - "CRE" -> 正交矩阵(CRE)
        
        - "positive" -> 正定矩阵
        
        - "normal" -> 正规矩阵
        
        - "noninv" -> 不可逆矩阵
        
        - "realeig" -> 实本征值的矩阵
        
        - "singular" -> 奇异矩阵
        
        - "rho" -> 密度矩阵
        
    
    矩阵（方阵）的分类：
    -----------------------
    - **简单矩阵** (可以相似对角化)
        
       - **正规矩阵** (M.H @ M = M @ M.H,  可以幺正相似对角化)
       
        （**幺正矩阵**、**厄密矩阵**、**对角矩阵**，但这三者没有包含关系）
        
       - **非正规矩阵** (可以相似对角化，但不可以幺正相似对角化)
        
    - **不简单矩阵** (不可以相似对角化/只能相似到Jordan型上)

    注：
    - 本征值始终是存在的，无论是不是简单矩阵。
    
    - 本征值为实数的矩阵可以是：不简单矩阵、非正规矩阵、厄密矩阵。
    
    - 如果正规矩阵的本征值是实数，则该正规矩阵必定为厄密矩阵。
    
    
    随机矩阵的分类：
    --------------------
    =========== ======================== ======================= ================== ===========
    ensemble    matrix class drawn from  measure                 invariant under    beta
    =========== ======================== ======================= ================== ===========
    GOE         real, symmetric          ``~ exp(-n/4 tr(H^2))`` orthogonal O       1
    ----------- ------------------------ ----------------------- ------------------ -----------
    GUE         hermitian                ``~ exp(-n/2 tr(H^2))`` unitary U          2
    ----------- ------------------------ ----------------------- ------------------ -----------
    CRE         O(n)                     Haar                    orthogonal O       /
    ----------- ------------------------ ----------------------- ------------------ -----------
    COE         U in U(n) with U = U^T   Haar                    orthogonal O       1
    ----------- ------------------------ ----------------------- ------------------ -----------
    CUE         U(n)                     Haar                    unitary U          2
    ----------- ------------------------ ----------------------- ------------------ -----------
    O_close_1   O(n)                     ?                       /                  /
    ----------- ------------------------ ----------------------- ------------------ -----------
    U_close_1   U(n)                     ?                       /                  /
    =========== ======================== ======================= ================== ===========
    
    
    随机矩阵的概率性质：
    ---------------------
    - 简单矩阵（复数或实数）：    
    
        - 以概率 1 可相似对角化;
        
        - 以概率 1 不可酉相似对角化
        
        - 概率 1 可逆
        
    
    - 正规矩阵（厄密矩阵，对称矩阵，幺正矩阵，实正交阵）：
    
        - 必可以相似对角化
        
        - 必可以酉相似对角化
        
        - 概率 1 可逆
        
    
    - 非正规矩阵：
    
        - 必可以相似对角化
        
        - 概率 1 可逆
        
    
    - 不简单矩阵：
    
        - 不可以相似对角化
        
        - 只能相似到Jordan型上
        
    
    - 奇异矩阵：
    
        - 不能被相似对角化（用来对角化的矩阵的逆矩阵矩阵元发散。）
        
        - 不能被酉相似对角化
        
        - 概率 1 可逆
        
        
    - 不可以逆矩阵：
    
        - 必不可逆 或者 逆矩阵发散
        
        - 以概率 1 可相似对角化
        
        - 以概率 1 不可酉相似对角化
        
    
    - 实本征值单阵
    
        - 以概率 1 可相似对角化

        - 以概率 1 不可酉相似对角化

        - 以概率 1 可逆

        
    生成随机数的方法参见：https://numpy.org/doc/stable/reference/random/generator.html#
    """
    type_to_function = {
        "simple": _random_simple_matrix,
        "realsimple": _random_simple_real_matrix,
        "GOE": _random_symmetric_matrix_goe,
        "GUE": _random_hermition_matrix_gue,
        "CUE": _random_unitary_matrix_cue,
        "COE": _randomize_orthogonal_coe,
        "CRE": _random_orthogonal_matrix_cre,
        "singular": _random_singular_matrix,
        "normal": _random_normal_matrix,
        "noninv": _random_noninv_matrix,
        "realeig": _random_real_eigen_matrix,
        "positive": _random_positive_matrix,
        "rho": _random_density_matrix
    }

    type = type.replace("herm", "GUE")
    type = type.replace("unit", "CUE")
    type = type.replace("symm", "GOE")
    type = type.replace("orth", "COE")
    if type in type_to_function:
        return type_to_function[type](dim, seed=seed)
    else:
        raise ValueError(f"Unknown type '{type}'.")


# @njit
def _random_simple_matrix(dim, seed=None):
    if seed is not None:
        _np.random.seed(seed)
    real_part = _np.random.randn(dim, dim)
    imag_part = _np.random.randn(dim, dim)
    return real_part + 1.0j * imag_part

# @njit
def _random_simple_real_matrix(dim, seed):
    if seed is not None:
        _np.random.seed(seed)
    return _np.random.randn(dim, dim)

# @njit
def _random_symmetric_matrix_goe(dim, seed):
    real_matrix = _random_simple_real_matrix(dim, seed)
    return 0.5 * (real_matrix + real_matrix.T)

# @njit
def _random_hermition_matrix_gue(dim, seed):
    complex_matrix = _random_simple_matrix(dim, seed)
    return (complex_matrix + complex_matrix.T.conj()) * 0.5


# @njit
def _random_unitary_matrix_cue(dim, seed):
    # rng = _np.random.default_rng(seed=seed)
    # A = rng.standard_normal((dim, dim)) + 1.j * rng.standard_normal((dim, dim))
    A = _random_simple_matrix(dim, seed)
    Q, R = _np.linalg.qr(A)
    # Q-R is not unique; to make it unique ensure that the diagonal of R is positive
    # Q' = Q*L; R' = L^{-1} *R, where L = diag(phase(diagonal(R)))
    L = _np.array([R[i, i] for i in range(R.shape[0])])
    L[_np.abs(L) < 1.e-15] = 1.
    Q *= L / _np.abs(L)
    return Q

# @njit
def _randomize_orthogonal_coe(dim, seed):
    U = _random_unitary_matrix_cue(dim, seed)
    U_contiguous = _np.ascontiguousarray(U)
    return _np.dot(U_contiguous.T, U_contiguous)

# @njit
def _random_orthogonal_matrix_cre(dim, seed):
    A = _random_simple_real_matrix(dim, seed)
    Q, R = _np.linalg.qr(A)
    # Q-R is not unique; to make it unique ensure that the diagonal of R is positive
    # Q' = Q*L; R' = L^{-1} *R, where L = diag(phase(diagonal(R)))
    L = _np.array([R[i, i] for i in range(R.shape[0])])
    Q *= _np.sign(L)
    return Q

# @njit
def _random_singular_matrix(dim, seed=None):
    if seed is not None:
        _np.random.seed(seed)
    n1 = _np.random.randint(low=0, high=2, size=dim - 1)
    while _np.all(n1 == 0):
        n1 = _np.random.randint(low=0, high=2, size=dim - 1)
    n = _np.random.standard_normal(size=dim)
    for i in range(dim - 1):
        if n1[i] == 1:
            n[i + 1] = n[i]
    a = _np.diag(n) + _np.diag(n1, 1)
    u = _np.random.standard_normal((dim, dim))
    return _np.linalg.inv(u) @ a @ u

# @njit
def _random_normal_matrix(dim, seed=None):
    v = _random_simple_matrix(dim, seed)
    u = _random_unitary_matrix_cue(dim, seed=seed)
    u_contiguous = _np.ascontiguousarray(u)
    v_contiguous = _np.ascontiguousarray(_np.diag(v))
    return u_contiguous @ v_contiguous @ u_contiguous.conj().T

# @njit
def _random_noninv_matrix(dim, seed=None):
    if seed is not None:
        _np.random.seed(seed)
    n = _np.random.randint(1, dim)
    v = _np.random.standard_normal(dim)
    poslis = list(range(n))
    for _ in range(n):
        v[poslis.pop(_np.random.randint(len(poslis)))] = 0
    u = _np.random.standard_normal((dim, dim))
    u_contiguous = _np.ascontiguousarray(u)
    v_contiguous = _np.ascontiguousarray(_np.diag(v))
    return _np.linalg.inv(u_contiguous) @ v_contiguous @ u_contiguous

# @njit
def _random_real_eigen_matrix(dim, seed):
    v = _random_simple_real_matrix(dim, seed)
    u = _random_simple_matrix(dim, seed)
    return (_np.linalg.inv(u) * v) @ u

# @njit
def _random_positive_matrix(dim, seed):
    mat = _random_simple_matrix(dim, seed)
    return mat @ mat.conjugate().transpose()

# @njit
def _random_density_matrix(dim, seed):
    res = _random_positive_matrix(dim=dim, seed=seed)
    trace = _np.sum(_np.diag(res))
    res[:] /= trace
    return res


def random_orthorgonal_matrix_close_I(dim, a=0.01, seed=None):
    r"""返回一个接近单位矩阵的随机正交矩阵。

    参数 a 决定结果与单位矩阵的接近程度；
    
    当 :math:`a \rightarrow 0` 时，:math:`<|O-E|>_a = 0``（其中 `E` 是单位矩阵）。
    """
    A = _random_symmetric_matrix_goe(dim, seed=seed) / (2. * dim)**0.5  # scale such that eigenvalues are in [-1, 1]
    E = _np.eye(dim)
    Q, R = _np.linalg.qr(E + a * A)
    L = _np.diagonal(R)  # make QR decomposition unique & ensure Q is close to one for small `a`
    Q *= _np.sign(L)
    return Q


def random_unitary_matrix_close_I(dim, a=0.01, seed=None):
    r"""返回一个接近单位矩阵的随机正交矩阵。

    接近单位矩阵的正交矩阵（对于小的 `a`）。
    特征值是独立同分布的，形式为 ``exp(1.j*a*x)``，其中 `x` 在 [-1, 1] 区间内均匀分布。
    """
    U = _random_unitary_matrix_cue(dim, seed=seed)
    E = _np.exp(1.j * a * (_np.random.rand(dim) * 2. - 1.))
    return _np.dot(U * E, U.T.conj())


def random_sparse_matrix(dim, density=0.1, seed=None):
    """生成一个稀疏矩阵
    """
    rng = _np.random.default_rng(seed=seed)
    nnz = round(density * dim * dim)
    ijs = rng.choice(range(0, dim**2), size=nnz, replace=False)
    i, j = _np.divmod(ijs, dim)
    data = rng.standard_normal(nnz) + 1.j * rng.standard_normal(nnz)
    return _sparse.coo_matrix((data, (i, j)), shape=(dim, dim)).asformat("csr")


################################
# 并行构建哈密顿量
################################

def parallel_build_matrix(hlocals, positions, coefficients, L, S, pauli=False, sparse=False, parallel=False, nthreads:Optional[int] =None):
    """
    通过直积生成矩阵，效率并不是否高，可以用于验证
    
    Examples
    --------
    >>> from quante.generate.matrix import parallel_build_matrix
    >>> mat = parallel_build_matrix(*ham.split_data(), L, S, pauli=False, sparse=False, parallel=True, nthreads=4)
    """
    
    # 生成操作符字典
    if S == 0.5:
        scale_factor = 2. if pauli else 1.
        OPER = {op: pauli_matrix(op, S=S) * scale_factor if op in 'xyZ' else pauli_matrix(op, S=S) for op in "xyZzpmiI"}
    else:
        OPER = {op: pauli_matrix(op, S=S) for op in "xyzpmiI"}
        OPER["Z"] = pauli_matrix("Z", S=S) * 2
        
    
    ikron_kws = {'sparse': True, 'stype': 'coo', 'coo_build': True}
    dims = (int(2 * S + 1),) * L
    
    def gen_term(args:tuple[str, int, float]):
        opstr, indx, j = args
        tmp = [_sparse.csc_array(OPER[oi]) for oi in opstr]
        return j * ikron(tmp, dims, indx, **ikron_kws) # type: ignore  #todo ikron 需要重写
    
    if parallel:
        from ..linalg.usenumba.numba_settings import get_thread_pool, parallel_reduce
        pool = get_thread_pool(nthreads)
        ham:_sparse.csr_array = parallel_reduce(lambda a,b : a+b, pool.map(gen_term, zip(hlocals, positions, coefficients)))
    else:
        ham:_sparse.csr_array = sum(map(gen_term, zip(hlocals, positions, coefficients))) # type: ignore
    
    if sparse:
        return ham.tocsr()
    
    return ham.toarray()


################################
# 一些常用的其他门与哈密顿量
################################

def heisenberg_matrix(
    L, 
    j: Union[number, tuple[number, number, number]] = 1.0, 
    h: Union[number, tuple[number, number, number]] = 0.0, 
    pauli: bool = False, 
    S: Union[int, float, str] = 1/2, 
    cyclic:bool = False,
    Nup:Optional[int]=None, 
    kblock:Optional[int]=None, 
    pblock:Optional[int]=None, 
    zblock:Optional[int]=None,
    pzblock:Optional[int]=None,
    jmblock:Optional[Union[int, tuple[int, int]]]=None
    ) -> _np.ndarray:
    """
    总是生成矩阵，而不是稀疏矩阵
    
    等价于：
    
    >>> ham = qt.generate.operas.heisenberg_operator(L, j, h, cyclic)
    >>> basis = qt.generate.basis.spin_basis(...)
    >>> mat = ham.to_matrix(basis, pauli, sparse=False)
    
    对于维数较小的矩阵比较高效
    """
    S = _check_spin_number(S)
    from .basis import spin_basis
    basis = spin_basis(L, S=S, Nup=Nup, kblock=kblock, pblock=pblock, zblock=zblock, pzblock=pzblock, jmblock=jmblock)
    try:
        # 尝试使用针对heisenberg链的方法
        try:
            jx, jy, jz = j # type: ignore
        except TypeError:
            jx = jy = jz = j
        assert _np.isclose(jx, jy) and h == 0 and S == 0.5
        if pauli:
            jx = jx * 4
            jz = jz * 4
        return basis._heimat(jx, jz, cyclic) # type: ignore
    except:
        # 如果失败使用一般方法
        from .operas.operas import heisenberg_operator
        ham = heisenberg_operator(L, j, h, cyclic)
        return ham.to_matrix(basis, pauli=pauli, sparse=False)


def hadamard_gate(dtype=complex):
    """The Hadamard gate."""
    return _np.array([[1., 1.],[1., -1.]], dtype=dtype) / 2**0.5

def phase_gate(phi=0.0, dtype=complex):
    """The generalized qubit phase-gate."""
    return _np.array([[1., 0.],[0., _np.exp(1.0j * phi)]], dtype=dtype)

def rotation_gate(phi, xyz='Z', dtype=complex):
    """The single qubit rotation gate."""
    R = _np.cos(phi / 2) * pauli_matrix('I') - 1.0j * _np.sin(phi / 2) * pauli_matrix(xyz)
    return _np.array(R, dtype=dtype)

def random_phase_model(L, q, ε, seed=None):
    """随机相位模型，
    
    但 ε 不太确定，这里用 ε**2 似乎才能与文献对的比较好（q=3 -> ε_c=0.25）
    
    """
    # get W1
    Ui = [_random_unitary_matrix_cue(q, seed=seed) for _ in range(L)]
    W1 = kron(*Ui)
    
    # get W2
    phi = [ε**2*_random_simple_matrix(q, seed) for _ in range(L-1)]
    dim = q**L
    W2 = _np.zeros((dim, dim), dtype=complex)
    
    for i in range(dim):
        res = 0
        for j in range(L-1):
            an = _spin_at_i(state=i, pos=j, L=L)
            anp1 = _spin_at_i(state=i, pos=j+1, L=L)
            res += phi[j][an, anp1]
        W2[i,i] = _np.exp(1j * res)
    return W2 @ W1

def _spin_at_i(state, pos, L):
    idf = 1 << (L - pos - 1)
    return 0 if state & idf == 0 else 1


def local_hamiltonian_spin_1D(model_key:str, pauli:bool=True, **kwargs) -> _np.ndarray:
    """
    Get local Hamiltonian term of 1D spin chain.
    
    Parameters
    ----------
    - XXZ: `J` (xx+yy+ `Δ` zz)+ `h` z
    
    - XX: `J` (xx+yy)+ `h` z
    
    - TFI: `J` zz + `h` x
    
    - TLFI: `J` zz + `g` x + `h` z
    
    Examples
    --------
    >>> qt.generate.matrix.local_hamiltonian_spin_1D("XXZ", J = 0.1, Δ = 1,h = 0.1)
    >>> qt.generate.matrix.local_hamiltonian_spin_1D("XX", J = 0.1, h = 0.1)
    >>> qt.generate.matrix.local_hamiltonian_spin_1D("TFI", J = 0.1, h = 0.1)
    >>> qt.generate.matrix.local_hamiltonian_spin_1D("TLFI", J = 0.1, h = 0.1, g=1)

        
    NOTE
    ----
    The filde terms need divide 2 in the input.

    params would convert inputs (such as x=1., y=2.) into a dictionary (such as {'x': 1., 'y': 2.})
    """
    model_dict = {
        "XX": "{J}*(XX+YY) + {h}*(ZI+IZ)",
        "XXZ": "{J}*(XX+YY+{Δ}*ZZ) + {h}/2*(ZI+IZ)",
        "TFI": "{J}*ZZ + {h}/2*(XI+IX)",
        "tTFI": "{J}*ZZ + {g}/2*(ZI+IZ) + {h}/2*(XI+IX)"
    }
    if model_key in model_dict:
        model_value = model_dict[model_key]
    else:
        raise NotImplementedError(f"'{model_key}' is NOT in model list.")
    if not pauli:
        model_value.replace("X", "x").replace("Y", "y").replace("Z", "z")
    model_value = model_value.format(**kwargs)
    return pauli_matrix(model_value)

