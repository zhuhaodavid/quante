# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 22:14:57
# @Last Modified by:   hzhu
# @Last Modified time: 2025-05-13 11:59:55
import numpy as np
from .spin import Oper, _merge_poscoef, _single_term
from .fermion import _sort_pm, _sort_posn


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
 

    def normal_ordering(self):
        """通过对易关系将算符转换为正则序列。

        .. math::
            -+ -> i + +-

        Returns
        -------
        FermionOper
            具有正则序列的 FermionOper 对象。
        """
        data = {}
        expandn = []
        for oper, (posnlist, coeflist) in self.data.items():
            coeflen = len(coeflist)
            if 'n' in oper:
                positions = [i for i, char in enumerate(oper) if char == 'n']
                for pos in sorted(positions, reverse=True):  # 从后往前插入，避免索引偏移
                    posnlist = np.insert(posnlist, pos + 1, posnlist[:, pos], axis=1)
                    for i in range(coeflen):
                        expandn.append(
                            (oper[:pos] + '+-' + oper[pos+1:], posnlist[i], coeflist[i])
                            )
            else:
                for i in range(coeflen):
                    expandn.append((oper, posnlist[i], coeflist[i]))
        for new_opnm, new_posn, new_coef in _sort_pm(expandn, 1.):
            new_posn, parity = _sort_posn(new_opnm, new_posn)
            posnlist, coeflist = data.setdefault(new_opnm, [[], []])
            posnlist.append(new_posn)
            coeflist.append(parity * new_coef)
        newdata = {}
        for name, (posnlist, coeflist) in data.items():
            newposn, newcoef = _merge_poscoef(posnlist, coeflist)
            if len(newposn) > 0:
                newdata[name] = (newposn, newcoef)
        return BosonOper(newdata)

  
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
        operator = self if self._has_expanded() else self.expandxy()
        from ..basis.quspin.quspin_basis.basis_1d.boson import boson_basis_1d
        if isinstance(basis, boson_basis_1d):
            op_list = []
            for opstr, posn, coef in operator.each_term():
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

    @classmethod
    def heisenberg_operator(cls, L, j=1.0, h=0.0, cyclic=False) -> "BosonOper":
        r"""
        生成 heisenberg 模型的哈密顿量，返回一个 'Oper' 的实例
        
        这个实例可以 automata, local_matrix, to_matrix 等方法

        .. math::
            \sum_{i=1}^{N-1} j * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1} + s^z_i s^z_{i+1}) + \sum_i^N h * s^z_i
        
        Parameters
        ----------
        L : int
            系统的长度
        j : float or tuple of float
            相互作用强度，可以是单个值，也可以是三个值表示 x, y, z 方向的相互作用强度
        h : float or tuple of float
            自旋场强度，可以是单个值，也可以是三个值表示 x, y, z 方向的自旋场强度
        cyclic : bool
            是否是周期性模型，默认是 False
        
        Examples
        --------
        >>> ham = qt.generate.operas.heisenberg_operator(L=10, j=1.0, h=0.0) # heisenberg model
        >>> ham = qt.generate.operas.heisenberg_operator(L=10, j=(1.0, 1.0, 0.0), h=0.0)  # xy model
        >>> ham = qt.generate.operas.heisenberg_operator(L=10, j=(0.0, 0.0, 1.0), h=(1.0, 0.0, 0.0))  # ising model
        """
        try:
            jx, jy, jz = j # type: ignore
        except TypeError:
            jx = jy = jz = j
        try:
            hx, hy, hz = h # type: ignore
        except TypeError:
            hz = h
            hx = hy = 0.0
        data = {}
        posn1 = np.arange(0,L, dtype=np.int32).reshape(L,1)
        coef1 = np.ones(L, dtype=np.float64)
        if cyclic:
            posn2 = np.array([[i%L, (i+1)%L] for i in range(L)], dtype=np.int32)
            coef2 = np.ones(L, dtype=np.float64)
        else:
            posn2 = np.array([[i, i+1] for i in range(L-1)], dtype=np.int32)
            coef2 = np.ones(L-1, dtype=np.float64)
        if jx != 0:
            data["xx"] = (posn2, jx*coef2)
        if jy != 0:
            data["yy"] = (posn2, jy*coef2)
        if jz != 0:
            data["zz"] = (posn2, jz*coef2)
        if hx != 0:
            data["x"] = (posn1, hx*coef1)
        if hy != 0:
            data["y"] = (posn1, hy*coef1)
        if hz != 0:
            data["z"] = (posn1, hz*coef1)
        return cls(data)
    
    @classmethod
    def builder(cls) -> "BosonBuilder":
        """
        返回一个 BosonBuilder 对象
        
        该对象可以用于构建哈密顿量
        
        Examples
        --------
        >>> builder = qt.generate.operas.BosonBuilder()
        >>> builder += '+-', [(0,1), (1,2)], -1.0
        >>> builder += '+-', [(2,3), (3,4)], -1.0
        >>> ham = builder.build()
        """
        return BosonBuilder()


def _expand_term(name):
    """Expand the term based on the given name and coefficient."""
    # Initialize with base case
    expanded_names = ['']
    expanded_coefs = [1]

    for char in reversed(name):  # Process characters from the end to the start
        if char == 'x':
            prefixes = ['+', '-']
            factors = [0.5, 0.5]
        elif char == 'y':
            prefixes = ['+', '-']
            factors = [-0.5j, 0.5j]
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


class BosonBuilder(BosonOper):
    def __init__(self):
        self.terms = {}

    def __iadd__(self, term) -> 'BosonBuilder':
        if isinstance(term, tuple):
            assert len(term) == 3 and len(term[0]) == len(term[1]), f"length wrong for term: {term}"
            for i in term[0]:
                assert i in ['I', '+', '-', 'n'], "term must be a tuple of I, +, -, n"
            
            if abs(term[2]) < 1e-10:
                return self
            
            posnlist, coeflist = self.terms.setdefault(term[0], [[], []])
            posnlist.append(term[1])
            coeflist.append(term[2])
            return self
        else:
            return super().__iadd__(term)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        data = {}
        for name, (posnlist, coeflist) in self.terms.items():
            data[name] = _merge_poscoef(posnlist, coeflist)
        super().__init__(data, type='f')
        
        if exc_type is not None:  # 检查是否发生错误
            tb.print_exc()  # 打印堆栈跟踪
    
    def build(self):
        data = {}
        for name, (posnlist, coeflist) in self.terms.items():
            data[name] = _merge_poscoef(posnlist, coeflist)
        return BosonOper(data, type='b')



