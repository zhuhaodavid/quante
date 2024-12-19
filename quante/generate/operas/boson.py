# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 22:14:57
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-15 22:30:48
import numpy as np
from .operas import Oper, merge_poscoef, _single_term

# todo spinful boson
# todo photon


class BosonOper(Oper):
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
    def __init__(self, data:dict, type='b') -> None:
        assert type == 'b'
        super().__init__(data, stype='b')
   
    def expandxy(self) -> 'BosonOper':
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
        if self.type == "b":
            res = BosonOper({})
            for name, (posn, coef) in self.data.items():
                expanded_names, expanded_coefs = _expand_term(name)
                for expanded_name, expanded_coef in zip(expanded_names, expanded_coefs):
                    res += BosonOper({expanded_name: (posn, coef * expanded_coef)})
            return res
        else:
            raise NotImplementedError()

    def _has_expanded(self) -> bool:
        for opnm in self.data.keys():
            if opnm not in ['I', "+", "-", "n", "z"]:
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
        from ..basis.quspin.quspin_basis.basis_1d.boson import boson_basis_1d
        if isinstance(basis, boson_basis_1d):
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
    def I(cls, i:int=0) -> "BosonOper":
        return cls({'I': _single_term((0,), 1.)})

    @classmethod
    def p(cls, i:int=0) -> "BosonOper":
        return cls({'+': _single_term((i,), 1.)})

    @classmethod
    def m(cls, i:int=0) -> "BosonOper":
        return cls({'-': _single_term((i,), 1.)})

    @classmethod
    def x(cls, i:int=0) -> "BosonOper":
        return cls({'x': _single_term((i,), 1.)})

    @classmethod
    def y(cls, i:int=0) -> "BosonOper":
        return cls({'y': _single_term((i,), 1.)})

    @classmethod
    def z(cls, i:int=0) -> "BosonOper":
        return cls({'z': _single_term((i,), 1.)})

    @classmethod
    def n(cls, i:int=0) -> "BosonOper":
        return cls({'n': _single_term((i,), 1.)})

    @classmethod
    def nn(cls, i:int, j:int) -> "BosonOper":
        return cls({'nn': _single_term((i, j), 1.)})

    @classmethod
    def zz(cls, i:int, j:int) -> "BosonOper":
        return cls({'zz': _single_term((i, j), 1.)})

    @classmethod
    def mp(cls, i:int, j:int) -> "BosonOper":
        if i==j:
            return cls({'n': _single_term((i,), 1.)})
        return cls({'-+': _single_term((i, j), 1.)})

    @classmethod
    def pm(cls, i:int, j:int) -> "BosonOper":
        return cls({'+-': _single_term((i, j), 1.)})

    @classmethod
    def xx(cls, i:int, j:int) -> "BosonOper":
        return cls({'xx': _single_term((i, j), 1.)})

    @classmethod
    def yy(cls, i:int, j:int) -> "BosonOper":
        return cls({'yy': _single_term((i, j), 1.)})

    @classmethod
    def xy(cls, i:int, j:int) -> "BosonOper":
        return cls({'xy': _single_term((i, j), 1.)})

    @classmethod
    def yx(cls, i:int, j:int) -> "BosonOper":
        return cls({'yx': _single_term((i, j), 1.)})
    
    @classmethod
    def sum(cls, oper) -> "BosonOper":
        data = {}
        for opx in oper:
            if isinstance(opx, (int,float,complex)):
                iterterm = (('I', (np.array([[0]], dtype=int), np.array([opx]))),)
            else:
                assert isinstance(opx, BosonOper), "Operands must be instances of SpinfulFermionOper"
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


