# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-17 22:08:45
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-09 14:16:22

import numpy as np
from scipy.sparse import csr_array
import traceback as tb
from .general import Oper, _single_term, _merge_poscoef
from .fermion import FermionOper
from typing import Literal

class SpinfulFermionOper(Oper):
    
    def __init__(self, data, type = 'sf'):
        super().__init__(data, type)
    
    def _prefix(self) -> str:
        return f"{self.__class__.__name__} (SpinUp | SpinDown) at {hex(id(self))}, \n"
    
    def _seperate_notion(self) -> str:
        return "|" #⇅
    
    def to_quspin(self):
        static = []
        for opnm, (posn, coef) in self.data.items():
            static_bond = []
            for i in range(len(coef)):
                static_bond.append([(coef[i]).item()] + [a.item() for a in posn[i]])
            static.append([opnm, static_bond])
        return static

    def to_matrix(self, basis, sparse=False):
        raise NotImplementedError("SpinfulFermionOper.to_matrix 方法未实现")
        # self._check_length(basis.L)
        # from ...bridge.quspin_utils import hamiltonian
        # return hamiltonian(self, basis, dtype=dtype, sparse=sparse)       
    
    def to_spinless(self, mode:Literal['near', 'extend']='near'):
        r"""将自旋算符转换为无自旋算符
        
        该方法将自旋算符转换为无自旋算符，返回一个新的 `FermionOper` 实例。

        `near` mode:

        .. code-block:: text

            sites:   0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  ...
            up:      0  |  2  |  4  |  6  |  8  | 10  | 12  | 14  ...
            down:    1  |  3  |  5  |  7  |  9  | 11  | 13  | 15  ...
        
        `extend` mode:

        .. code-block:: text
        
            sites:  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  ...
            up:     0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  ...
            down:   8  |  9  | 10  | 11  | 12  | 13  | 14  | 15  ...

        Parameters
        ----------
        mode : str, optional
            转换模式，默认为 'near'。可选值为 'near' 或 'extend'。
        
        Returns
        -------
        FermionOper
            转换后的无自旋算符。
        """
        L = self.L
        data = {}
        for name, (posnlist, coeflist) in self.data.items():
            vbarpos = name.find('|')
            newname = name[:vbarpos] + name[vbarpos+1:]
            newposnlist = posnlist.copy()
            if mode == 'near':
                newposnlist[:, :vbarpos] = 2*posnlist[:, :vbarpos]
                newposnlist[:, vbarpos:] = 2*posnlist[:, vbarpos:] + 1
            else:
                newposnlist[:, vbarpos:] = posnlist[:, vbarpos:] + L
            newcoeflist = coeflist.copy()
            oldpos, oldcoef = data.get(newname, (None, None))
            if oldpos is None and oldcoef is None:
                data[newname] = (newposnlist, newcoeflist)
            else:
                data[newname] = _merge_poscoef([oldpos, newposnlist], [oldcoef, newcoeflist])
        return FermionOper(data)
    
    def builder(self) -> 'SpinfulFermionBuilder':
        r"""返回一个 `SpinfulFermionBuilder` 实例
        
        该方法用于创建一个 `SpinfulFermionBuilder` 实例，方便构建自旋算符。
        
        Returns
        -------
        SpinfulFermionBuilder
            一个新的 `SpinfulFermionBuilder` 实例。
        """
        return SpinfulFermionBuilder()


def p(i:int=0, sigma='up') -> SpinfulFermionOper:
    if sigma in [0, 'u', '+', 'up']:
        return SpinfulFermionOper({'+|': _single_term((i,), 1.)})
    elif sigma in [1, 'd', '-', 'down']:
        return SpinfulFermionOper({'|+': _single_term((i,), 1.)})
    else:
        raise ValueError(f"Invalid sigma {sigma}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")

def m(i:int=0, sigma='up') -> SpinfulFermionOper:
    if sigma in [0, 'u', '+', 'up']:
        return SpinfulFermionOper({'-|': _single_term((i,), 1.)})
    elif sigma in [1, 'd', '-', 'down']:
        return SpinfulFermionOper({'|-': _single_term((i,), 1.)})
    else:
        raise ValueError(f"Invalid sigma {sigma}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")

def n(i:int=0, sigma='up') -> SpinfulFermionOper:
    if sigma in [0, 'u', '+', 'up']:
        return SpinfulFermionOper({'n|': _single_term((i,), 1.)})
    elif sigma in [1, 'd', '-', 'down']:
        return SpinfulFermionOper({'|n': _single_term((i,), 1.)})
    else:
        raise ValueError(f"Invalid sigma {sigma}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")

def pm(i:int=0, sigma_i='up', j:int=0, sigma_j='up') -> SpinfulFermionOper:
    if sigma_i in [0, 'u', '+', 'up']:
        if sigma_j in [0, 'u', '+', 'up']:
            return SpinfulFermionOper({'+-|': _single_term((i, j), 1.)})
        elif sigma_j in [1, 'd', '-', 'down']:
            return SpinfulFermionOper({'+|-': _single_term((i, j), 1.)})
    elif sigma_i in [1, 'd', '-', 'down']:
        if sigma_j in [0, 'u', '+', 'up']:
            return SpinfulFermionOper({'-|+': _single_term((j, i), -1.)})
        elif sigma_j in [1, 'd', '-', 'down']:
            return SpinfulFermionOper({'|+-': _single_term((i, j), 1.)})
    else:
        raise ValueError(f"Invalid sigma {sigma_i, sigma_j}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")

def mp(i:int=0, sigma_i='up', j:int=0, sigma_j='up') -> SpinfulFermionOper:
    if sigma_i in [0, 'u', '+', 'up']:
        if sigma_j in [0, 'u', '+', 'up']:
            return SpinfulFermionOper({'-+|': _single_term((i, j), 1.)})
        elif sigma_j in [1, 'd', '-', 'down']:
            return SpinfulFermionOper({'-|+': _single_term((i, j), 1.)})
    elif sigma_i in [1, 'd', '-', 'down']:
        if sigma_j in [0, 'u', '+', 'up']:
            return SpinfulFermionOper({'+|-': _single_term((j, i), -1.)})
        elif sigma_j in [1, 'd', '-', 'down']:
            return SpinfulFermionOper({'|-+': _single_term((i, j), 1.)})
    else:
        raise ValueError(f"Invalid sigma {sigma_i, sigma_j}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")

def nn(i:int=0, sigma_i='up', j:int=0, sigma_j='up') -> SpinfulFermionOper:
    if sigma_i in [0, 'u', '+', 'up']:
        if sigma_j in [0, 'u', '+', 'up']:
            return SpinfulFermionOper({'nn|': _single_term((i, j), 1.)})
        elif sigma_j in [1, 'd', '-', 'down']:
            return SpinfulFermionOper({'n|n': _single_term((i, j), 1.)})
    elif sigma_i in [1, 'd', '-', 'down']:
        if sigma_j in [0, 'u', '+', 'up']:
            return SpinfulFermionOper({'n|n': _single_term((i, j), 1.)})
        elif sigma_j in [1, 'd', '-', 'down']:
            return SpinfulFermionOper({'|nn': _single_term((i, j), 1.)})
    else:
        raise ValueError(f"Invalid sigma {sigma_i, sigma_j}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")

def sum(oper):
    data = {}
    stype = None
    for opx in oper:
        if isinstance(opx, (int,float,complex)):
            iterterm = (('I|', (np.array([[0]], dtype=int), np.array([opx]))),)
        else:
            assert isinstance(opx, SpinfulFermionOper), "Operands must be instances of SpinfulFermionOper"
            iterterm = opx.data.items()
        for name, (posn, coef) in iterterm:
            posnlist, coeflist = data.get(name, (None,None))
            if posnlist is None and coeflist is None:
                data[name] = ([posn], [coef])
            else:
                posnlist.append(posn)
                coeflist.append(coef)
    # merge terms
    newdata = {}
    for name, (posnlist, coeflist) in data.items():
        newposn, newcoef = _merge_poscoef(posnlist, coeflist)
        if len(newposn) > 0:
            newdata[name] = (newposn, newcoef)
    if stype is None:
        stype = 'sf'
    return SpinfulFermionOper(newdata)


def Fermi_Hubbard_operator(L, J=1.0, U=5.0, cyclic=False) -> SpinfulFermionOper:
    r"""Fermi-Hubbard Model
    
    .. math::
        H = -J\sum_{i=0,\sigma}^{L-2} \left(c^\dagger_{i\sigma}c_{i+1,\sigma} - c_{i\sigma}c^\dagger_{i+1,\sigma}\right) +U\sum_{i=0}^{L-1} n_{i\uparrow }n_{i\downarrow }.
    
    Example:
    --------
    >>> op = qt.generate.operas.SpinfulFermionOper
    >>> L = 8
    >>> N = L // 2  # number of particles
    >>> N_up = N // 2 + N % 2  # number of fermions with spin up
    >>> N_down = N // 2  # number of fermions with spin down
    >>> basis = qt.generate.basis.spinful_fermion_basis(L=L, Nf=(N_up, N_down))
    >>> ham = op.Fermi_Hubbard_operator(L=L, cyclic=False)
    >>> mat = ham.to_matrix(basis, dtype=float)
    """
    data = {}
    posn1 = np.array([[i, i] for i in range(L)], dtype=np.int64)
    coef1 = np.ones(L, dtype=np.float64)
    if cyclic:
        posn2 = np.array([[i%L, (i+1)%L] for i in range(L)], dtype=np.int64)
        coef2 = np.ones(L, dtype=np.float64)
    else:
        posn2 = np.array([[i, i+1] for i in range(L-1)], dtype=np.int64)
        coef2 = np.ones(L-1, dtype=np.float64)
        
    data['+-|'] =  (posn2, -J*coef2)
    data['-+|'] =  (posn2, J*coef2)
    data['|+-'] =  (posn2, -J*coef2)
    data['|-+'] =  (posn2, J*coef2)
    data['n|n'] = (posn1, U*coef1)
    return SpinfulFermionOper(data)


class SpinfulFermionBuilder:
    def __init__(self):
        r"""初始化 SpinfulFermionBuilder 类

        算符名称具有 'xxx|xxx' 的形式，前一半表示自旋上，后一半表示自旋下。
        
        如
        .. math::
            c^\dagger_{i,\uparrow} c_{i+1,\downarrow}
        
        表示为 `"+|-", [i, i+1], 1.`
        
        而
        .. math::
            c^\dagger_{i,\downarrow} c_{i+1,\uparrow}
        
        表示为 `"-|+", [i+1, i], -1.`

        也是总要把自旋向上放在前面，向下放在后面。
        
        Examples
        --------
        >>> import quante as qt
        >>> op = qt.generate.operas.fermion
        >>> L = 8
        >>> J = 1.0
        >>> Δ = 0.5
        >>> γ = 0.1
        >>> builder = op.SpinfulFermionBuilder()
        >>> for l in range(L-1):
        ...     # hopping term
        ...     builder += '+-|', [l+1, l], -J/2
        ...     builder += '|+-', [l+1, l], -J/2
        ...     builder += '+-|', [l, l+1], -J/2
        ...     builder += '|+-', [l, l+1], -J/2
        ...     # non-hermitian term
        ...     builder += '+-|', [l+1, l], -γ/2
        ...     builder += '|+-', [l+1, l],  γ/2
        ...     builder += '+-|', [l, l+1],  γ/2
        ...     builder += '|+-', [l, l+1], -γ/2
        ...     # spin-orbit coupling term
        ...     builder += '+|-', [l+1, l], 1j * Δ/2
        ...     builder += '-|+', [l+1, l], 1j * Δ/2
        ...     builder += '-|+', [l, l+1], - 1j * Δ/2
        ...     builder += '+|-', [l, l+1], - 1j * Δ/2
        >>> spinfulham = builder.build()
        """
        self.terms = {}

    def __iadd__(self, term) -> 'SpinfulFermionBuilder':
        if isinstance(term, tuple):
            assert len(term) == 3 and len(term[0]) == len(term[1]) + 1, f"length wrong for term: {term}"
            for i in term[0]:
                assert i in ['I', '+', '-', '|', 'n'], "term must be a tuple of I, +, -"
            assert '|' in term[0], "term must contain |"
            
            if abs(term[2]) < 1e-10:
                return self
            
            posnlist, coeflist = self.terms.setdefault(term[0], [[], []])
            posnlist.append(term[1])
            coeflist.append(term[2])
            return self
        else:
            return super().__iadd__(term)
  
    def build(self):
        data = {}
        for name, (posnlist, coeflist) in self.terms.items():
            data[name] = _merge_poscoef(posnlist, coeflist)
        return SpinfulFermionOper(data, type='f')

def builder() -> SpinfulFermionBuilder:
    r"""返回一个 `SpinfulFermionBuilder` 实例
    
    该方法用于创建一个 `SpinfulFermionBuilder` 实例，方便构建自旋算符。
    
    Returns
    -------
    SpinfulFermionBuilder
        一个新的 `SpinfulFermionBuilder` 实例。
    """
    return SpinfulFermionBuilder()

