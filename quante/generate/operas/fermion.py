# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 19:13:08
# @Last Modified by:   hzhu
# @Last Modified time: 2025-02-18 15:24:34

import numpy as np
from .spin import Oper, _single_term, _merge_poscoef

class FermionOper(Oper):
    """
    算符类: 该类用于表示和操作量子系统的算符。
    
    提供最重要的功能包括:
    
    - 转化为 quspin 接受的格式 (`quspin_form`)

    - 算符生成矩阵 (`to_matrix`)

    .. code-block:: python
    
        data = {
            operator_1: (posn_ndarray, coef_ndarray),
            operator_2: (posn_ndarray, coef_ndarray), 
            ...
        }
    """
    def __init__(self, data:dict, type='f') -> None:
        assert type == 'f'
        super().__init__(data, stype='f')
   
    def sort(self):
        """
        对算符进行排序，使得所有的位置信息按照从小到大的顺序排列。
        """
        data = {}
        for oper, posn, coef in self.each_term():
            posn_sorted = _sort_pm([
                _sort_posn(oper, posn, coef)
            ])
            for new_oper, new_posn, new_coef in posn_sorted:
                posnlist, coeflist = data.setdefault(new_oper, ([], []))
                posnlist.append(new_posn)
                coeflist.append(new_coef)
        return FermionOper(data)
    
    def dtype(self):
        for _, (_, coef) in self.data.items():
            if np.iscomplexobj(coef):
                return complex
        return float
    
    def quspin_form(self):
        """
        返回 quspin 可以接受的格式
        
        Examples
        --------
        >>> from quspin.operators import hamiltonian
        >>> from quspin.basis import spin_basis_1d
        >>> ham = sum(xx(i,i+1) + yy(i,i+1) for i in range(5))
        >>> basis = spin_basis_1d(L=6)
        >>> mat = hamiltonian(ham.quspin_form(), [], basis=basis)
        """
        static = []
        for opnm, (posn, coef) in self.data.items():
            static_bond = []
            for i in range(len(coef)):
                static_bond.append([(coef[i]).item()] + [a.item() for a in posn[i]])
            static.append([opnm, static_bond])
        return static

    def to_matrix(self, basis, dtype=np.complex128, sparse=False):
        self._check_length(basis.L)
        from ..basis.quspin.quspin_basis.basis_1d.fermion import spinless_fermion_basis_1d
        if isinstance(basis, spinless_fermion_basis_1d):
            op_list = []
            for opstr, posn, coef in self.each_term():
                op_list.append([opstr, posn, coef])
            mat = basis._make_matrix(op_list, dtype=dtype)
            if sparse:
                return mat
            else:
                return mat.toarray()
        else:
            raise NotImplementedError("不支持的基矢类型")

    @classmethod
    def I(cls, i:int=0) -> "FermionOper":
        return cls({'I': _single_term((0,), 1.)})

    @classmethod
    def p(cls, i:int=0) -> "FermionOper":
        return cls({'+': _single_term((i,), 1.)})

    @classmethod
    def m(cls, i:int=0) -> "FermionOper":
        return cls({'-': _single_term((i,), 1.)})

    @classmethod
    def x(cls, i:int=0) -> "FermionOper":
        return cls.p(i) + cls.m(i)

    @classmethod
    def y(cls, i:int=0) -> "FermionOper":
        return (cls.p(i) - cls.m(i))*1j

    @classmethod
    def n(cls, i:int=0) -> "FermionOper":
        return cls({'n': _single_term((i,), 1.)})

    @classmethod
    def nn(cls, i:int, j:int) -> "FermionOper":
        return cls({'nn': _single_term((i, j), 1.)})

    @classmethod
    def mp(cls, i:int, j:int) -> "FermionOper":
        if i==j:
            return cls({'n': _single_term((i,), 1.)})
        return cls({'-+': _single_term((i, j), 1.)})

    @classmethod
    def pm(cls, i:int, j:int) -> "FermionOper":
        return cls({'+-': _single_term((i, j), 1.)})

    @classmethod
    def sum(cls, oper) -> "FermionOper":
        data = {}
        for opx in oper:
            if isinstance(opx, (int,float,complex)):
                iterterm = (('I', (np.array([[0]], dtype=int), np.array([opx]))),)
            else:
                assert isinstance(opx, FermionOper), "Operands must be instances of SpinfulFermionOper"
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
        return cls(newdata)
    
    @classmethod
    def ssh_operator(cls, L, J=1.0, deltaJ=0.1, Delta=0.5, cyclic=False) -> "FermionOper":
        r"""Su-Schrieffer-Heeger Model
        
        .. math::
            H = \sum_{j=0}^{L-1} -(J+(-1)^j\delta J)\left(c_jc^\dagger_{j+1} - c^\dagger_{j}c_{j+1}\right) + \Delta(-1)^jn_j.
            
        Examples
        --------
        >>> op = qt.generate.operas.FermionOper
        >>> L = 100
        >>> basis = qt.generate.basis.spinless_fermion_basis(L=L, Nf=1)
        >>> ham = op.ssh_operator(L=L, cyclic=True)
        >>> mat = ham.to_matrix(basis, dtype=float)
        >>> print(mat)
        
        等价于
        >>> ham = (op.sum(
                (-1) * (J + (-1)**j * DeltaJ) * (op.mp(j,j+1) - op.pm(j,j+1)) for j in range(L-1)) 
            + 
            op.sum(Delta * (-1)**j * op.n(j) for j in range(L)))
        """
        data = {}
        if cyclic:
            posn1 = np.array([[i%L, (i+1)%L] for i in range(L)], dtype=np.int64)
            coef1 = np.array([-(J + (-1)**j * deltaJ) for j in range(L)], dtype=np.float64)
        else:
            posn1 = np.array([[i, i+1] for i in range(L-1)], dtype=np.int64)
            coef1 = np.array([-(J + (-1)**j * deltaJ) for j in range(L-1)], dtype=np.float64)

        posn2 = np.arange(0,L, dtype=np.int64).reshape(L,1)
        coef2 = np.array([Delta * (-1)**j for j in range(L)], dtype=np.float64)
        data['-+'] =  (posn1, coef1)
        data['+-'] =  (posn1, -coef1)
        data['n'] = (posn2, coef2)
        return cls(data)


# from functools import lru_cache
def is_even_permutation(arr):
    """判断排列是否为偶排列。"""
    n = len(arr)
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            if arr[i] > arr[j]:
                cnt += 1
    return cnt % 2 == 0


def _sort_pm(tmp_res):
    res = []
    for oper, posn, coef in tmp_res:
        sorted_ones = [(oper, posn, coef)]
        for i in range(1, len(posn)):
            if posn[i-1] == posn[i]: 
                if oper[i-1:i+1] == '-+':
                    sorted_ones = [
                        (oper[:i-1] + '+-' + oper[i+1:], posn, -coef),
                    ]
                    if len(posn) == 2:
                        sorted_ones.append(('I', np.array([0]), coef))
                    else:
                        sorted_ones.append(
                            (
                                oper[:i-1] + oper[i+1:],
                                np.delete(posn, [i-1, i]),
                                coef
                            )
                        )
                    sorted_ones = _sort_pm(sorted_ones)
                elif oper[i-1:i+1] != '+-':
                    sorted_ones = []
        res += sorted_ones
    return res


def _sort_posn(oper, posn, coef):
    inc_indx = np.argsort(posn, stable=True)
    sign = 1 if is_even_permutation(inc_indx) else -1
    new_oper = ''.join(oper[i] for i in inc_indx)
    new_posn = np.array(posn)[inc_indx]
    new_coef = sign * coef
    return new_oper, new_posn, new_coef


class SpinlessFermionOperBuilder:
    def __init__(self):
        """
        可用的符号包括：I, p, m, x, y, z
        
        Example:
        --------
        >>> ham = SpinOperBuilder()
        >>> for i in range(10):
        >>>     ham += 1.0, 'x', i, 'x', i+1
        >>>     ham += 1.0, 'y', i, 'y', i+1
        >>>     ham += 1.0, 'z', i, 'z', i+1
        >>>     ham += 1.0, 'x', i
        >>> ham = ham.to_oper()
        """
        self.terms = {}

    def __iadd__(self, term) -> 'SpinlessFermionOperBuilder':
        assert isinstance(term, tuple) and len(term) % 2 == 1, "term must be a tuple of odd length"
        for i in range(1, len(term), 2):
            assert term[i] in ['I', '+', '-', 'n'], "term must be a tuple of I, p, m, n"
        
        opnm = "".join(term[1::2])
        posn = np.array(term[2::2])
        coef = term[0]
        if np.all(np.diff(posn) >= 0):
            posn_sorted = (opnm, posn, coef)
        else:
            posn_sorted = _sort_posn(opnm, posn, coef)
        posn_sorteds = _sort_pm([posn_sorted])
        for new_oper, new_posn, new_coef in posn_sorteds:
            posnlist, coeflist = self.terms.setdefault(new_oper, [[], []])
            posnlist.append(new_posn)
            coeflist.append(new_coef)
        return self
    
    def to_oper(self):
        data = {}
        for name, (posnlist, coeflist) in self.terms.items():
            data[name] = _merge_poscoef(posnlist, coeflist)
        return FermionOper(data)
    
    build = to_oper

#####################################################################
#####################################################################

class SpinfulFermionOper(Oper):
    
    def __init__(self, data, type = 'sf'):
        super().__init__(data, type)
    
    def quspin_form(self):
        static = []
        for opnm, (posn, coef) in self.data.items():
            static_bond = []
            for i in range(len(coef)):
                static_bond.append([(coef[i]).item()] + [a.item() for a in posn[i]])
            static.append([opnm, static_bond])
        return static

    def to_matrix(self, basis, dtype=np.complex128, sparse=False):
        self._check_length(basis.L)
        from ..basis.quspin.quspin_basis.basis_1d.fermion import spinful_fermion_basis_1d
        if isinstance(basis, spinful_fermion_basis_1d):
            op_list = []
            for opstr, posn, coef in self.each_term():
                op_list.append([opstr, posn, coef])
            mat = basis._make_matrix(op_list, dtype=dtype)
            if sparse:
                return mat
            else:
                return mat.toarray()
        else:
            raise NotImplementedError("不支持的基矢类型")

    @classmethod
    def p(cls, i:int=0, sigma='up') -> "SpinfulFermionOper":
        if sigma in [0, 'u', '+', 'up']:
            return cls({'+|': _single_term((i,), 1.)})
        elif sigma in [1, 'd', '-', 'down']:
            return cls({'|+': _single_term((i,), 1.)})
        else:
            raise ValueError(f"Invalid sigma {sigma}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")
    
    @classmethod
    def m(cls, i:int=0, sigma='up') -> "SpinfulFermionOper":
        if sigma in [0, 'u', '+', 'up']:
            return cls({'-|': _single_term((i,), 1.)})
        elif sigma in [1, 'd', '-', 'down']:
            return cls({'|-': _single_term((i,), 1.)})
        else:
            raise ValueError(f"Invalid sigma {sigma}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")

    @classmethod
    def n(cls, i:int=0, sigma='up') -> "SpinfulFermionOper":
        if sigma in [0, 'u', '+', 'up']:
            return cls({'n|': _single_term((i,), 1.)})
        elif sigma in [1, 'd', '-', 'down']:
            return cls({'|n': _single_term((i,), 1.)})
        else:
            raise ValueError(f"Invalid sigma {sigma}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")

    @classmethod
    def pm(cls, i:int=0, sigma_i='up', j:int=0, sigma_j='up') -> "SpinfulFermionOper":
        if sigma_i in [0, 'u', '+', 'up']:
            if sigma_j in [0, 'u', '+', 'up']:
                return cls({'+-|': _single_term((i, j), 1.)})
            elif sigma_j in [1, 'd', '-', 'down']:
                return cls({'+|-': _single_term((i, j), 1.)})
        elif sigma_i in [1, 'd', '-', 'down']:
            if sigma_j in [0, 'u', '+', 'up']:
                return cls({'-|+': _single_term((j, i), -1.)})
            elif sigma_j in [1, 'd', '-', 'down']:
                return cls({'|+-': _single_term((i, j), 1.)})
        else:
            raise ValueError(f"Invalid sigma {sigma_i, sigma_j}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")
    
    @classmethod
    def mp(cls, i:int=0, sigma_i='up', j:int=0, sigma_j='up') -> "SpinfulFermionOper":
        if sigma_i in [0, 'u', '+', 'up']:
            if sigma_j in [0, 'u', '+', 'up']:
                return cls({'-+|': _single_term((i, j), 1.)})
            elif sigma_j in [1, 'd', '-', 'down']:
                return cls({'-|+': _single_term((i, j), 1.)})
        elif sigma_i in [1, 'd', '-', 'down']:
            if sigma_j in [0, 'u', '+', 'up']:
                return cls({'+|-': _single_term((j, i), -1.)})
            elif sigma_j in [1, 'd', '-', 'down']:
                return cls({'|-+': _single_term((i, j), 1.)})
        else:
            raise ValueError(f"Invalid sigma {sigma_i, sigma_j}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")
    
    
    @classmethod
    def nn(cls, i:int=0, sigma_i='up', j:int=0, sigma_j='up') -> "SpinfulFermionOper":
        if sigma_i in [0, 'u', '+', 'up']:
            if sigma_j in [0, 'u', '+', 'up']:
                return cls({'nn|': _single_term((i, j), 1.)})
            elif sigma_j in [1, 'd', '-', 'down']:
                return cls({'n|n': _single_term((i, j), 1.)})
        elif sigma_i in [1, 'd', '-', 'down']:
            if sigma_j in [0, 'u', '+', 'up']:
                return cls({'n|n': _single_term((i, j), 1.)})
            elif sigma_j in [1, 'd', '-', 'down']:
                return cls({'|nn': _single_term((i, j), 1.)})
        else:
            raise ValueError(f"Invalid sigma {sigma_i, sigma_j}, should be in [0, 'u', 'up', '+'] for spin up and ['1, 'd', '-',' 'down'] for spin down")
    
    @classmethod
    def sum(cls, oper):
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
        return cls(newdata)

    
    @classmethod
    def Fermi_Hubbard_operator(cls, L, J=1.0, U=5.0, cyclic=False) -> "SpinfulFermionOper":
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

        等价于：
        >>> ham = (- J * op.sum(    
            op.pm(i,sigma,i+1,sigma) + op.mp(i,sigma,i+1,sigma) 
            for i in range(L-1) for sigma in [0,1]) 
            +  
            U * op.sum(op.nn(i,0,i,1) for i in range(L)))
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
        return cls(data)
