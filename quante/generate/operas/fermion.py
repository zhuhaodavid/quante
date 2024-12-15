# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 19:13:08
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-15 22:19:54

import numpy as np
from .operas import Oper, merge_poscoef, _single_term

class FermionOper(Oper):
    """
    算符类: 该类用于表示和操作量子系统的算符。
    
    提供最重要的功能包括:
    
    - 算符的加减、乘法、幂运算, 以及将 x,y 用 +,- 展开 (`expandxy`)

    - 转化为 quspin 接受的格式 (`quspin_form`)

    - 算符生成矩阵 (`to_matrix`) 、MPO (`automata`)

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
   
    def expandxy(self) -> 'FermionOper':
        """
        
        展开算符中的 `x`,`y`,`z`，
        只保留 `I`, `p`, `m`, `n` 四种算符
        
        展开之后，应当只包含 `p`, `m`, `I`, `Z` 这三种算符
       
        Examples
        --------
        >>> ham = op.heisenberg_operator(L=4)
        >>> ham = ham.expandxy()
        >>> ham.show_string_form()
        
        注:只是自旋
        """
        if self._has_expanded():
            return self.copy()
        if self.type == "f":
            res = FermionOper({})
            for name, (posn, coef) in _sort_subindx(self.data).items():
                expanded_names, expanded_coefs = _expand_term(name)
                for expanded_name, expanded_coef in zip(expanded_names, expanded_coefs):
                    res += FermionOper({expanded_name: (posn, coef * expanded_coef)})
            return res
        else:
            raise NotImplementedError()

    def _has_expanded(self) -> bool:
        for opnm in self.data.keys():
            if opnm not in ['I', "+", "-", "n", 'z']:
                return False
        return True
    
    def dtype(self):
        tmp = self if self._has_expanded() else self.expandxy()
        for _, (_, coef) in tmp.data.items():
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
        return cls({'x': _single_term((i,), 1.)})

    @classmethod
    def y(cls, i:int=0) -> "FermionOper":
        return cls({'y': _single_term((i,), 1.)})

    @classmethod
    def z(cls, i:int=0) -> "FermionOper":
        return cls({'z': _single_term((i,), 1.)})

    @classmethod
    def n(cls, i:int=0) -> "FermionOper":
        return cls({'n': _single_term((i,), 1.)})

    @classmethod
    def nn(cls, i:int, j:int) -> "FermionOper":
        return cls({'nn': _single_term((i, j), 1.)})

    @classmethod
    def zz(cls, i:int, j:int) -> "FermionOper":
        return cls({'zz': _single_term((i, j), 1.)})

    @classmethod
    def mp(cls, i:int, j:int) -> "FermionOper":
        if i==j:
            return cls({'n': _single_term((i,), 1.)})
        return cls({'-+': _single_term((i, j), 1.)})

    @classmethod
    def pm(cls, i:int, j:int) -> "FermionOper":
        return cls({'+-': _single_term((i, j), 1.)})

    @classmethod
    def xx(cls, i:int, j:int) -> "FermionOper":
        return cls({'xx': _single_term((i, j), 1.)})

    @classmethod
    def yy(cls, i:int, j:int) -> "FermionOper":
        return cls({'yy': _single_term((i, j), 1.)})

    @classmethod
    def xy(cls, i:int, j:int) -> "FermionOper":
        return cls({'xy': _single_term((i, j), 1.)})

    @classmethod
    def yx(cls, i:int, j:int) -> "FermionOper":
        return cls({'yx': _single_term((i, j), 1.)})
    
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
            newposn, newcoef = merge_poscoef(posnlist, coeflist)
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
            newposn, newcoef = merge_poscoef(posnlist, coeflist)
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
        

def is_even_permutation(arr):
    """判断排列是否为偶排列。"""
    indices = np.arange(arr.shape[1])
    inversions = np.sum(
        (arr[:, :, None] > arr[:, None, :]) & (indices[None, :, None] < indices[None, None, :]), axis=(1, 2)
    )
    return 1 - 2 * (inversions % 2)

def _sort_subindx(dic):
    """Sort the subindices of operators"""
    newdic = {}
    for opnm, (posn, coef) in dic.items():
        indx = np.argsort(posn, axis=1)
        percoef = is_even_permutation(indx)
        newposn = np.take_along_axis(posn, indx, axis=1)
        for i in range(len(indx)):
            curnewname = ''.join(opnm[j] for j in indx[i])
            tmp1, tmp2 = newdic.setdefault(curnewname, ([], []))
            tmp1.append(newposn[i])
            tmp2.append(coef[i] * percoef[i])
    
    keys_to_delete = []
    for key, (newposn, newcoef) in newdic.items():
        tmp1, tmp2 = merge_poscoef(newposn, newcoef)
        if len(tmp2) > 0:
            newdic[key] = (tmp1, np.real_if_close(tmp2))
        else:
            keys_to_delete.append(key)

    # 删除暂存的键
    for key in keys_to_delete:
        del newdic[key]
        
    return newdic

def _expand_term(name):
    """Expand the term based on the given name and coefficient."""
    # Initialize with base case
    expanded_names = ['']
    expanded_coefs = [1]

    for char in reversed(name):  # Process characters from the end to the start
        if char == 'x':
            prefixes = ['+', '-']
            factors = [1., 1.]
        elif char == 'y':
            prefixes = ['+', '-']
            factors = [-1j, 1j]
        else:
            prefixes = [char]
            factors = [1.]

        # Combine prefixes and coefficients with the current expansions
        new_names = []
        new_coefs = []
        for p, f in zip(prefixes, factors):
            for n, coef in zip(expanded_names, expanded_coefs):
                new_names.append(p + n)
                new_coefs.append(f * coef)

        # Update expanded terms
        expanded_names, expanded_coefs = new_names, new_coefs

    return expanded_names, expanded_coefs
