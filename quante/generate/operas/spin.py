# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-07 20:26:18
# @Last Modified by:   hzhu
# @Last Modified time: 2025-04-16 23:29:20

import warnings
import traceback as tb
import numpy as np
import scipy.sparse as sp
import copy 
import typing


if typing.TYPE_CHECKING:
    from .fermion import FermionOper

def _argsort_positions(pos_array):
    """
    对位置数组进行排序，返回排序后的索引数组。

    Parameters
    ----------
    pos_array : np.ndarray
        位置数组。

    Returns
    -------
    np.ndarray
        排序后的索引数组。 
    """
    n, m = pos_array.shape
    indices = np.arange(n)

    for col in range(m):  # Sort by columns from last to first
        indices = indices[np.argsort(pos_array[indices, col], kind='mergesort')]

    return indices


def _sort_positions(positions, coefficients):
    """
    对位置和系数数组进行排序，将相同位置的系数相加。

    Parameters
    ----------
    positions : np.ndarray
        位置数组。
    coefficients : np.ndarray
        系数数组。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        排序后的位置和系数数组。
    """
    # Sort positions using custom argsort
    sorted_indices = _argsort_positions(positions)
    res_pos = positions[sorted_indices]
    res_coef = coefficients[sorted_indices]
    
    cur_pos = 0
    cur_coef = res_coef[0]
    total_len = len(res_pos)
    for i in range(1, total_len):
        if np.all(res_pos[i] == res_pos[i-1]):
            cur_coef += res_coef[i]
        else:
            res_pos[cur_pos] = res_pos[i-1]
            res_coef[cur_pos] = cur_coef
            cur_pos += 1
            cur_coef = res_coef[i]
    
    res_pos[cur_pos] = res_pos[total_len-1]
    res_coef[cur_pos] = cur_coef
    
    mask = res_coef[:cur_pos + 1] != 0  # Remove zero coefficients
    return res_pos[:cur_pos+1][mask], res_coef[:cur_pos+1][mask]


def _merge_poscoef(poss, coefs):
    """
    合并位置和系数数组，将相同位置的系数相加。
    将会重新排序

    Parameters
    ----------
    poss : list[np.ndarray]
        位置数组的列表。
    coefs : list[np.ndarray]
        系数数组的列表。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        合并后的位置和系数数组。
    """
    res_pos, res_coef = np.vstack(poss), np.hstack(coefs)
    return _sort_positions(res_pos, res_coef)


class Oper:
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
    def __init__(self, data:dict, stype:str) -> None:  # todo 处理费米子系统
        self.data = data
        self.type = stype
    
    @property
    def L(self):
        return max([np.max(posn) for posn, _ in self.data.values()]) + 1 
    
    def __iadd__(self, oper, add_or_minus=1) -> 'Oper':
        """ self += oper """
        if isinstance(oper, (int, float, complex)) and oper == 0:  # a + 0 = a
            return self
        elif isinstance(oper, (int, float, complex)):  # 加单位阵
            old_pos, old_coef = self.data.get('I', (None,None))
            if old_coef is None and old_pos is None:
                self.data['I'] = (np.array([[0]],dtype=int), np.array([1.], dtype=float))
            else:
                old_coef[0] += add_or_minus*1.
            return self
        elif isinstance(oper, Oper):  # 两个算符相加
            assert self.type == oper.type, NotImplementedError("算符类型不相同")
            for name, (pos, coef) in oper.data.items():
                old_pos, old_coef = self.data.get(name, (None, None))
                if old_coef is None and old_pos is None:
                    self.data[name] = (pos, add_or_minus*coef)
                else:
                    newpos,newcoef = _merge_poscoef((old_pos, pos), (old_coef, add_or_minus*coef))
                    if len(newpos) > 0:
                        self.data[name] = (newpos, np.real_if_close(newcoef))
                    else:
                        del self.data[name]
            return self
        else:
            raise NotImplementedError(f"oper type {type(oper)} not supported")
    
    def __add__(self, oper) -> 'Oper':
        """ self + a * oper """
        self_copy = self.copy()
        self_copy.__iadd__(oper)
        return self_copy

    def copy(self) -> 'Oper':
        cls = self.__class__
        return cls(copy.deepcopy(self.data), self.type)
    
    def __radd__(self, oper) -> 'Oper':
        """ num + oper """
        return self.__add__(oper)  # a + b = b + a
    
    def __isub__(self, oper):
        """self -= oper"""
        self.__iadd__(oper, add_or_minus=-1.)
        return self
    
    def __sub__(self, oper) -> 'Oper':
        """self - oper"""
        self_copy = self.copy()
        self_copy.__iadd__(oper, add_or_minus=-1.)
        return self_copy
    
    def __imul__(self, oper):
        """self *= oper"""
        if isinstance(oper, (int, float, complex)):
            for name, (posn, coef) in self.data.items():
                try:
                    coef *= oper
                except TypeError:
                    self.data[name] = (posn, coef * oper)
            return self
        elif isinstance(oper, Oper):
            return self.__matmul__(oper)
    
    def __mul__(self, scale) -> 'Oper':
        """ oper * num """
        if isinstance(scale, (int, float, complex)):
            self_copy = self.copy()
            self_copy.__imul__(scale)
            return self_copy
        return self.__matmul__(scale)
        
    def __rmul__(self, scale) -> 'Oper':
        """ num * oper """
        if isinstance(scale, Oper):
            return self.__matmul__(scale)
        return self * scale

    def __rsub__(self, oper) -> 'Oper':
        """oper - self"""
        self_copy = self.copy()
        self_copy *= -1
        self_copy.__iadd__(oper)
        return self_copy
    
    def __neg__(self) -> 'Oper':
        """- self"""
        return (-1) * self
    
    def __truediv__(self, num) -> 'Oper':
        """self / num"""
        return (1 / num) * self

    def __matmul__(self, oper:'Oper') -> 'Oper':
        """ self * oper """
        if self.type == oper.type and len(oper.type) == 1:  # 相同类型
            cls = self.__class__
            newoper = cls({}, self.type)
            for opnm1, (posn1, coef1) in self.data.items():
                if opnm1 == "I":
                    newoper += np.sum(coef1) * oper
                    continue
                for opnm2, (posn2, coef2) in oper.data.items():
                    if opnm2 == "I":
                        newoper += np.sum(coef2) * cls({opnm1: (posn1, coef1)}, self.type)
                        continue
                    newopnm = opnm1 + opnm2
                    newposn, newcoef = catposcoef(posn1, coef1, posn2, coef2)
                    newoper += cls({newopnm : (newposn, newcoef)}, self.type)
            return newoper
        else:
            raise NotImplementedError("不同基矢相加")
    
    def __pow__(self, n, m=None) -> 'Oper':
        """ self ** n """
        if m is not None:
            raise NotImplementedError("modulo is not implemented")
        if n <= 0:
            raise NotImplementedError("inverse is not implemented")
        newoper = self
        for _ in range(n - 1):
            newoper = self * newoper
        return newoper
    
    def _check_length(self, L:int) -> None:
        assert L >= self.L

    def show_string_form(self, maxlen=80, form='v') -> None:
        """打印算符的字符串形式"""
        if form == 'v':
            print(self.table_form(maxlen=maxlen))
        elif form == 'h':
            print(self.table_form2(maxlen=maxlen))
    
    def table_form(self, maxlen=90) -> str:
        if len(self.data) == 0:
            return "0"
        pages = []
        first_line = "|"
        second_line = "|"
        data_list = []
        last_len = 0
        for i, (operator, (posn, coef)) in enumerate(self.data.items()):
            
            oper_len = len(operator)
            
            if (len(first_line) + 6 * oper_len  > maxlen 
                and i != 0):
                pages.append(first_line)
                pages.append(second_line)
                pages += data_list
                pages.append("="*len(first_line))
                first_line = "|"
                second_line = "|"
                data_list = []
                last_len = 0
            
            ml = 12 # max length
            for i in range(len(coef)):
                if len(data_list) <= i:
                    data_list.append("|")
                
                data_line = data_list[i]
                if len(data_line) < last_len:
                    data_line += " " * (last_len-2) + "|"
                for j in range(oper_len):
                    data_line += f"   {posn[i][j]:<3}"

                if 0.1 <= abs(coef[i]) < 100:
                    tmp = f"{coef[i]:.3f}".rjust(10) + " |"  # 普通浮点数格式
                else:
                    tmp = f"{coef[i]:.2e}".rjust(10) +" |"  # 科学计数法格式
                data_line += tmp
                if len(tmp) > ml:
                    ml = len(tmp)
                data_list[i] = data_line
            
            last_len = len(data_line)
        
            for i in operator:
                first_line += f"   {i:<3}"
                second_line += "-"*6
            makeup = ml - 12
            first_line += "  " + " "*makeup + "   coef. |"
            second_line += "-"*(11+makeup) + "|"
            
        pages.append(first_line)
        pages.append(second_line)
        pages += data_list

        # 增加 self 的名字和地址
        prefix = f"{self.__class__.__name__} at {hex(id(self))}, \n"
        return prefix + '\n'.join(pages) + '\n'
    
    def table_form2(self, maxlen=90) -> str:
        lines = []
        for operator, (posn, coef) in self.data.items():
            oper_len = len(operator)
            line = f"{operator}: "
            for i in range(len(coef)):
                if i > 0:
                    line += " + "
                line += f"{coef[i]:.3f} * " + " * ".join([f"({posn[i][j]})" for j in range(oper_len)])
            lines.append(line)
        print('\n'.join(lines))
    
    def __repr__(self) -> str:
        """
        返回算符的字符串形式
        """
        return self.table_form(maxlen=80)
    
    def each_term(self):
        """
        遍历每个算符的每个项
        
        Examples
        --------
        >>> ham = op.heisenberg_operator(L=10)
        >>> for opnm, posn, coef in ham.each_term():
        >>>    print(opnm, posn, coef)

        Yields
        ------
        Generator[tuple[str, tuple[int], Union[float, complex]], None, None]
        
        注: 逐项返回算符的名称、位置和系数,但是只能使用一次, 不可重复使用
        如果要重复使用,需要 list(ham.each_term())
        """
        for operator, (posn, coef) in self.data.items():
            for i in range(len(coef)):
                yield operator, tuple(posn[i]), coef[i]

    def show(self, whichonm=None) -> None:
        """
        使用 igraph 画出算符的图形表示

        Parameters
        ----------
        whichonm : str, optional
            指定要画哪个算符，如 'xx' 的图形, 如果为 None, 则画出所有算符，每个算符用线表示, by default None

        Raises
        ------
        ImportError
            没有安装 igraph 模块
        """
        import matplotlib.pyplot as plt
        try:
            import igraph as ig # type: ignore
        except ImportError:
            raise ImportError("需要安装: pip install igraph")
        
        dic = {}
        for onm, posn, coef in self.each_term():
            if whichonm is not None:
                if onm!= whichonm:
                    continue
            interact = "-".join([str(i) for i in posn])
            if whichonm is not None:
                tmp = dic.setdefault(interact, 0)
                dic[interact] = tmp + coef
            else:
                tmp = dic.setdefault(interact, [])
                tmp.append(onm)
        
        if whichonm is None:
            for interact, onms in dic.items():
                dic[interact] = "+".join(onms)

        fml = ",".join(dic.keys())
        g = ig.Graph.Formula(
            fml
        )
        
        g.es["weight"] = list(dic["-".join([ges.vertex_tuple[0]['name'], ges.vertex_tuple[1]['name']])]  for ges in g.es)
        # print(ges.vertex_tuples[0]['name'] for ges in g.es)
        # 如何表示单体作用？
        # 如何表示多体作用？

        fig, ax = plt.subplots(figsize=(10, 2))
        ig.plot(
            g,
            target=ax,
            layout=g.layout_kamada_kawai(),
            vertex_size=30,
            vertex_color="lightblue",
            vertex_label=g.vs["name"],
            edge_width=1.0,
            edge_color='gray',
            edge_label=g.es['weight']
        )

        plt.show()

    def save(self, filename:str = 'ham.h5') -> None:
        from ...basicfun import save_hdf5
        data_dict = {name: {"posn": posn, "coef": coef} for name, (posn, coef) in self.data.items()}
        data_dict.update({"type": self.type})
        save_hdf5(filename, "/", data_dict)
        
    @classmethod
    def load(cls, filename:str) -> 'Oper':
        from ...basicfun import load_hdf5
        data = load_hdf5(filename, '/', '/')
        dic = {}
        for key, val in data.items():
            if key == "type":
                continue
            dic[key] = (val["posn"], val["coef"])
        return cls(dic, data['type'].decode('utf-8'))
    
    def trotter_gates(self) -> None:
        raise NotImplementedError("Subclasses should implement this.")

    def _minimal_shift(self):
        l = min([np.min(posn) for posn, _ in self.data.values()])
        newdata = {}
        for oper, (posn, coef) in self.data.items():
            newdata[oper] = (posn-l, coef)
        return l, SpinOper(newdata)


class SpinOper(Oper):
    def __init__(self, data:dict, type='s') -> None:
        assert type == 's'
        super().__init__(data, stype='s')
 
    def expandxy(self, pauli:bool = False) -> 'SpinOper':
        """
        
        展开算符中的 `x`,`y`，同时将 `z` 替换为 `Z` 
        其中 `Z` = `pm`-`mp` = `sigma_z`，这是为了 `to_matrix` 方便
        
        展开之后，应当只包含 `p`, `m`, `i`, `Z` 这三种算符

        Parameters
        ----------
        pauli : bool, optional
            是否使用 Pauli 矩阵作为局部矩阵。默认为 False，即使用常规矩阵。
        
        Examples
        --------
        >>> ham = op.heisenberg_operator(L=4)
        >>> ham = ham.expandxy()
        >>> ham.show_string_form()
        
        注:只是自旋
        """
        if self._has_expanded():
            return self.copy()
        c = 1.0 if pauli else 0.5
        res = SpinOper({})
        for name, (posn, coef) in self.data.items():
            expanded_names, expanded_coefs = _expand_term(name, c)
            for expanded_name, expanded_coef in zip(expanded_names, expanded_coefs):
                res += SpinOper({expanded_name: (posn, coef * expanded_coef)})
        return res
    
    def _has_expanded(self) -> bool:
        for opnm in self.data.keys():
            if opnm not in ['I', "p", "m", "Z"]:
                return False
        return True

    def jw_transfer(self, pauli=False) -> 'FermionOper':
        from .fermion import FermionOper
        ham = self.expandxy(pauli=pauli)
        res = FermionOper({})
        for opnm, pos, coeff in ham.each_term():
            # # 判断 opnm 中有多少个 p,m
            if opnm == 'pm' and pos[0]+1 == pos[1]:
                fham = FermionOper({"+-": ([pos], [- coeff])})
                res += fham
            elif opnm == 'mp' and pos[0]+1 == pos[1]:
                fham = FermionOper({"+-": ([pos[::-1]], [-coeff])})
                res += fham
            else:
                pm_num = opnm.count('p') + opnm.count('m')
                if pm_num % 2 == 0:
                    fham = FermionOper._convert_from_spin(opnm[0], pos[0], coeff)
                    for i in range(1, len(opnm)):
                        for cur_pos in range(pos[i-1], pos[i]):
                            fham @= FermionOper._convert_from_spin('Z', cur_pos, 1.)
                        fham @= FermionOper._convert_from_spin(opnm[i], pos[i], 1.)
                res += fham.normal_ordering()
        return res

    @property
    def dtype(self):
        for name, (_, coef) in self.data.items():
            if np.iscomplexobj(coef) or 'y' in name:
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
            c = opnm.count('Z')
            for i in range(len(coef)):
                static_bond.append([coef[i]*2**c] + list(posn[i]))
            static.append([opnm.replace('m', '-').replace('p', '+').replace('Z', 'z'), static_bond])
        return static
    
    def to_matrix(self, basis, pauli=None, sparse=False, savememory=False):
        """
        生成哈密顿量在给定基矢下的矩阵，对于自旋 1/2 默认使用 symmetrize 的方法计算矩阵元
        
        .. 警告::
            这个函数不检查哈密顿量是否有对称性。如果哈密顿量没有对称性，那么这个函数会返回错误的结果，而不会报错。
        
        Parameters
        ----------
        basis : Basis
            基矢。
        pauli : bool, optional
            是否使用 Pauli 矩阵作为局部矩阵。默认为 False，即使用常规矩阵。如果哈密顿量已经用 expandxy 展开过，那么这个参数无效，同时给出警告。
        sparse : bool, optional
            是否返回稀疏矩阵。默认为 False，即返回 numpy 数组。

        Returns
        -------
        Union[_np.ndarray, _sp.sparse.csr_array]
            哈密顿量在给定基矢下的矩阵。
        
        Examples
        --------
        >>> L = 10
        >>> basis = qt.generate.basis.spin_basis(L=L, Nup=5)
        >>> ham = qt.generate.operas.heisenberg_operator(L=L)
        >>> mat = ham.to_matrix(basis)
        >>> print(mat)
        
        时间对比参考 example/exact_diagonalization.ipynb
        
        对于**没有对称性**的基矢，automata 收缩是最快的方法：
        
        >>> from ..tensor.automata import get_sparse_matrix
        >>> mat = get_sparse_matrix(L, *ham.split_data(), pauli=pauli, usecuda=True)
        
        对于**没有对称性**的基矢，也可以使用 `to_matrix_cuda` 来实现：
        
        >>> from ..torch_utils.symmetry import to_matrix_cuda
        >>> eachterm, hascomplex = ham.expandxy(False)._convert_to_quick_form()
        >>> mat = to_matrix_cuda(basis, eachterm, hascomplex)
        
        有对称性的,也可以这么使用,但是加速并不明显.
        
        如果反复生成也可以将 to_matrix 拆开来：
        
        >>> eachterm, hascomplex = ham.expandxy(pauli=pauli)._convert_to_quick_form()
        >>> mat = basis._sparse_matrix(eachterm, hascomplex)
        
        可以反复使用 `basis._sparse_matrix(eachterm, hascomplex)`
        """
        if self.data == {}:
            if not sparse:
                return np.zeros((basis.Ns, basis.Ns), dtype=float)
            return sp.csr_matrix((basis.Ns, basis.Ns), dtype=float)
        self._check_length(basis.L)
        from ..basis.symmetry.basis_class import SpinHalfBasis, SpinHighBasis
        if isinstance(basis, (SpinHalfBasis, SpinHighBasis)):
            if basis.S != 0.5 and pauli is True:
                raise KeyError("自旋不是 1/2，不能使用 Pauli 矩阵")
            if self._has_expanded():
                if pauli is not None:
                    warnings.warn("pauli in to_matrix is not used")
                eachterm, hascomplex = self._convert_to_quick_form()
            else:
                if pauli is None:
                    pauli = False
                eachterm, hascomplex = self.expandxy(pauli)._convert_to_quick_form()
            mat = basis._sparse_matrix(eachterm, hascomplex, savememory=savememory)
            if sparse:
                return mat
            else:
                return mat.toarray()
        else:
            from ..basis.quspin.quspin_basis.basis_1d.spin import spin_basis_1d
            if isinstance(basis, spin_basis_1d):
                qs_list = []
                for opnm, posncoefs in self.quspin_form():
                    for posn in posncoefs:
                        qs_list.append((opnm, posn[1:], posn[0]))
                mat = basis._make_matrix(qs_list, dtype=np.complex128)
                if sparse:
                    return mat
                else:
                    return mat.toarray()
            else:
                raise NotImplementedError(f"Spin Oper 不支持的 {type(basis).__name__} 作为基矢")
    
    def _convert_to_quick_form(self):
        """这个函数专门为 to_matrix 写的，其他函数不需要"""
        eachterm = []
        hascomplex = False
        for opnm, posn, coef in self.each_term():
            opnm_list = []
            for i in opnm:
                if i == 'm':
                    opnm_list.append(0)
                elif i == 'p':
                    opnm_list.append(1)
                elif i == 'I':
                    opnm_list.append(2)
                elif i == 'Z':
                    opnm_list.append(3)
                else:
                    raise ValueError(f"opnm {i} is not supported")
            opnm_list = np.array(opnm_list, dtype=np.int64)
            posn_list = np.array(posn, dtype=np.int64)
            coef_real = np.real_if_close(coef).item()
            if isinstance(coef_real, complex):
                hascomplex = True
            eachterm.append((opnm_list, posn_list, coef_real))
        return eachterm, hascomplex

    def automata(
        self,
        L: int | None = None,
        pauli: bool = False,
        S: str = '1/2',
    ):
        """
        生成算符的 mpo 形式

        Parameters
        ----------
        L : int
            系统的长度，即量子比特的数量。
        pauli : bool, optional
            是否使用 Pauli 矩阵作为局部矩阵。默认为 False，即使用常规矩阵。
        d : int, optional
            局部矩阵的维度。默认为 2，即二维矩阵。
        S : str, optional
            用于生成局部矩阵的函数。如果提供，该函数将根据字符串参数生成对应的局部矩阵。默认为 None，即使用默认的局部矩阵生成方式。

        Examples
        --------
        >>> L = 10
        >>> ham = op.heisenberg_operator(L)
        >>> basis = (L, pauli=False)
        >>> mpo = ham.automata(L, pauli=False)
        """
        if L is None:
            L = self.L
        else:
            assert L >= self.L

        from ..matrix import pauli_matrix
        local_matrix = lambda x: pauli_matrix(x.upper() if x in ['x', 'y', 'z'] else x, S=S) if pauli else pauli_matrix(x.upper() if x in ['X', 'Y', 'Z'] else x, S=S)

        if L == 1:
            tmp = np.sum(c*local_matrix(i) for i, _, c in self.each_term())
            return [tmp.reshape(1,*tmp.shape,1)]

        from ..automata import automata_mpo
        expanded = self.expandxy(pauli=pauli)
        return automata_mpo(expanded.each_term(), L, local_matrix, expanded.dtype)

        # 下面是使用 simple_automata_mpo 的调用方式
        # from ..automata import simple_automata_mpo
        # hlocals, positions, coefficients = self.expandxy(pauli=pauli).split_data()
        # coefficients = np.real_if_close(coefficients)
        # return simple_automata_mpo(L, hlocals, positions, coefficients, d=d, pauli=pauli, local_matrix_function=local_matrix, dtype=coefficients.dtype)

    def split_data(self):
        """这个函数是为 automata 写的，但 parallel_matrix 等函数可能会用到"""
        operators, positions, coefficients = [], [], []
        for operator, position, coefficient in self.each_term():
            operators.append(operator)
            positions.append(position)
            coefficients.append(coefficient)
        return operators, positions, coefficients
  
    def gate2_decomposition(self, L:int, tau:float, form="ladder", pauli:bool=True) -> tuple[list[int],list[np.ndarray]]:
        r"""
        用最简单的方法（ladder/brick）将哈密顿量的演化拆分成一些列局域两体门：
        
        .. code-block:: text
        
            ladder:
               |    |    |    |  ╭-┴----┴-╮
               |    |    |    |  ╰-┬----┬-╯
               |    |    |  ╭-┴----┴-╮  |  
               |    |    |  ╰-┬----┬-╯  |  
               |    |  ╭-┴----┴-╮  |    |  
               |    |  ╰-┬----┬-╯  |    |  
               |  ╭-┴----┴-╮  |    |    |  
               |  ╰-┬----┬-╯  |    |    |  
             ╭-┴----┴-╮  |    |    |    |  
             ╰-┬----┬-╯  |    |    |    |  

        .. code-block:: text
        
            brick:
               |  ╭-┴----┴-╮╭-┴----┴-╮╭-┴----┴-╮  |   
               |  ╰-┬----┬-╯╰-┬----┬-╯╰-┬----┬-╯  |   
            ╭--┴----┴-╮╭-┴----┴-╮╭-┴----┴-╮╭-┴----┴-╮ 
            ╰--┬----┬-╯╰-┬----┬-╯╰-┬----┬-╯╰-┬----┬-╯ 

        Parameters
        ----------
        L : int
            系统的长度，即量子比特的数量。
        tau : float
            时间步长，用于控制演化的速度。
        form : str, optional
            门的形式，可以是 "ladder" 或 "brick"。
        pauli : bool, optional
            是否使用 Pauli 矩阵作为局部矩阵。默认为 True，即使用 Pauli 矩阵。

        Returns
        -------
        tuple[list[int],list[_np.ndarray]]
            门的位置和对应的矩阵。
        
        Examples
        --------
        >>> import quante as qt
        >>> import numpy as np
        >>> import torch as tc
        >>> 
        >>> L = 10
        >>> tau = 0.01
        >>> ham = qt.generate.operas.heisenberg_operator(L=L)
        >>> gates = ham.gate2_decomposition(L, tau=tau, form='ladder')
        >>> U_tau = tcclass.MPO(tcfuncs.mpo_eye(L, [2]*L))  # 生成单位矩阵
        >>> for pos_cur, gate in zip(*gates):
        >>>     local_evolve = tc.tensor(gate).reshape(2,2,2,2)
        >>>     U_tau.apply_gate_2b(pos_cur, local_evolve)
        >>> # 验证程序:
        >>> basis = qt.generate.basis.spin_basis(L=L)
        >>> mat = ham.to_matrix(basis, sparse=False)
        >>> evolve_operator = qt.linalg.expm( -1j*tau*mat)
        >>> print(np.linalg.norm(evolve_operator - U_tau.full_contract().numpy()))
        """
        expandself = self.expandxy(pauli=pauli)
        
        gates = []  # 用于存储局部两体门
        positions = []  # 用于储存两体门作用的格点位置（两体门所作用的格点 其中最左侧格点的位置）
        
        if form == "ladder":
            increment = 1  # ladder 每增加一个门，格点位置向右移动 1，如：第一个门作用在 0,1 格点，第二个门作用在 1,2 格点
            layer_number = 1  # ladder 只需要一层，从左到右
        elif form == "brick":
            increment = 2  # brick 每增加一个门，格点位置向右移动 2，如：第一个门作用在 0,1 格点，第二个门作用在 2,3 格点
            layer_number = 2  # brick 需要两层，从左到右，再从右到左
        else:
            raise ValueError(f"form keyword form({form}) should be ladder of brock")
        
        for layer_index in range(layer_number):
            # 首先拿到哈密顿量中的局域算符对应的 4x4 矩阵
            layer = "even" if layer_index % 2 == 0 else "odd"
            
            gate_positions_in_layer_i, local_hamiltonian_matrices = expandself._get_local_hamiltonians(L, increment, layer=layer)
            
            if layer_index == 0:
                # 从左到右的门
                positions += gate_positions_in_layer_i
                # 将局域哈密顿量转换为局域演化门：expm
                gates += [sp.linalg.expm((-1j*tau)*matrix) for matrix in local_hamiltonian_matrices]
            else:
                # 从右到左的门，如果 layer_number = 1，那么这个部分不会执行
                positions += gate_positions_in_layer_i[::-1]
                # 将局域哈密顿量转换为局域演化门：expm
                gates += [sp.linalg.expm((-1j*tau)*matrix) for matrix in local_hamiltonian_matrices[::-1]]

        return positions, gates

    def trotter_gates(self, L:int, tau:float, order:str="2", evolve_type:str="time", N_step:int=1, pauli:bool=True) -> tuple[list[int], list[np.ndarray]]:
        """
        获取局域演化门

        使用 Trotter-Suzuki 分解将给定的哈密顿量分解为一系列局部操作，并按照给定的时间步长 tau 进行演化。
        
        .. 警告::
            该函数只支持最近邻相互作用的哈密顿量。

        Parameters
        ----------
        L : int
            系统的长度，即量子比特的数量。
        tau : float
            时间步长，用于控制演化的速度。
        order : str, optional
            Trotter 分解的阶数，决定了分解的精度。``1, 2, 4, '4_opt'``
            Order ``1`` approximation is simply :math:`e^A a^B`.
            Order ``2`` is the "leapfrog" `e^{A/2} e^B e^{A/2}`.
            Order ``4`` is the fourth-order from `[suzuki1991] <https://doi.org/10.1063/1.529425>`_
        evolve_type : str, optional
            演化的类型，可以是 "time" 或 "temporal"
        N_step : int, optional
            演化的总步数，总演化时间为 N_step * tau。
        pauli : bool, optional
            是否使用 Pauli 矩阵作为局部矩阵。默认为 True，即使用 Pauli 矩阵。

        Returns
        -------
        tuple[list[int], list[_np.ndarray]]
            位置列表, 对应的局部操作

        Examples
        --------
        >>> gates = ham.trotter_gates(L, tau=tau, order='2', evolve_type='time', pauli=False)
        >>> U_tau = MPO.eye(L)
        >>> for pos_cur, gate in zip(*gates):
        >>>     U_tau.apply_gate_(pos_cur, gate)
        """
        from ...linalg import expm
        expandself = self.expandxy(pauli=pauli)
        # 获取偶数位置和局域哈密顿量
        even_positions, even_hamiltonians = expandself._get_local_hamiltonians(L, increment=2, layer="even")
        # 获取奇数位置和局域哈密顿量
        odd_positions, odd_hamiltonians =  expandself._get_local_hamiltonians(L, increment=2, layer="odd")

        # 遍历Suzuki-Trotter时间步长 不同的阶数将 tau 拆分成了不同种 dt，每一种会给出一组演化算符，将它们都记录下来
        if evolve_type == "time":
            c = -1.j
        elif evolve_type == "temporal":
            c = -1.
        else:
            raise ValueError("type should be 'time' or 'temporal'")

        even_gates_dts = []
        odd_gates_dts = []
        
        for dt in SpinOper._trotter_suzuki_time_steps(order):
            # 计算偶数位置的演化门的单步演化门
            even_gates_dt = [expm(hamiltonian, c*tau*dt) for hamiltonian in even_hamiltonians]
            even_gates_dts.append(even_gates_dt)
            
            # 计算奇数位置的单步演化门
            odd_gates_dt = [expm(hamiltonian, c*tau*dt) for hamiltonian in odd_hamiltonians]
            odd_gates_dts.append(odd_gates_dt)
        
        # 初始化门和位置列表
        gates = []
        positions = []
        
        # 根据 trotter-suzuki 先后分别演化奇偶门
        for dt_index, evolve_even_gate in SpinOper._pseudo_trotter_suzuki_decomposition(order, N_step):
            
            direction = 1 if evolve_even_gate else -1  # 如果是偶数，方向是 1，表示从左向右演化，如果是奇数，方向是 -1，表示从右向左演化
            
            # 根据even的值选择演化门和位置列表
            single_step_gates = even_gates_dts if evolve_even_gate else odd_gates_dts
            gates += single_step_gates[dt_index][::direction]  # 演化 single_step_gates 中第 indx 个
            
            # 根据even的值选择演化门和位置列表
            single_step_positions = even_positions if evolve_even_gate else odd_positions
            positions += single_step_positions[::direction]

        return positions, gates

    def _get_local_hamiltonians(self, L:int, increment:int, layer:str) -> tuple[list[int], list[np.ndarray]]:
        """得到偶数层或奇数层的局域哈密顿量，以及局域哈密顿量的位置信息
        """
        if layer == "even":
            first_position = 0
        elif layer == "odd":
            first_position = 1
        else:
            raise ValueError("layer should be even or odd")
        
        site_positions = []  # 局域哈密顿量的格点位置
        local_hamiltonians = []  # 局域哈密顿量
        
        # * 逐个找到作用在 position 和 position+1 这两个格点上的局域哈密顿量
        for site_position in range(first_position, L-1, increment):
            local_hamiltonian, hasoper = self.local(site_position, L)
            # 如果 local_hamiltonian == 0，表示没有算符作用在 site_position 和 site_position+1 这两个格点上，跳过
            if not hasoper:
                continue
            site_positions.append(site_position)
            local_hamiltonians.append(local_hamiltonian)
        return site_positions, local_hamiltonians

    def local(self, site_position:int, L:int | None = None) -> np.ndarray:
        """
        根据 Oper 的实例得到作用在 position 和 position+1 这两个格点上的局域哈密顿量
        
        根据 self 的算符名称决定是否使用 pauli 矩阵。
        """
        if L is None:
            L = self.L
        else:
            assert L >= self.L
        assert site_position < L-1, "site_position should be less than L-1"
        from ..matrix import PAULI_MAT
        
        local_hamiltonian = np.zeros((4,4), dtype=self.dtype)  # 用来储存所有作用到 position 和 position+1 这两个格点上的局域哈密顿量的和
        hasoper = False  # 用来判断是否有算符作用在 position 和 position+1 这两个格点上

        # 遍历哈密顿量中的每一项，找到所有作用在 position 和 position+1 这两个格点上的局域哈密顿量
        for oper_operator, (posn, coef) in self.data.items():
            position_length = len(oper_operator)  # 判断局域哈密顿量是单体项还是两体项

            if position_length == 1:
                indices = np.where(posn[:, 0] == site_position)[0]
                if site_position == 0:
                    for i in indices:
                        local_hamiltonian += coef[i] * np.kron(PAULI_MAT[oper_operator], PAULI_MAT['i'])    
                        hasoper = True
                else:
                    for i in indices:
                        local_hamiltonian += (coef[i]/2) * np.kron(PAULI_MAT[oper_operator], PAULI_MAT['i'])
                        hasoper = True
                
                indices = np.where(posn[:, 0] == site_position+1)[0]
                if site_position+1 == L-1:
                    for i in indices:
                        local_hamiltonian += coef[i] * np.kron(PAULI_MAT['i'], PAULI_MAT[oper_operator])
                        hasoper = True
                else:
                    for i in indices:
                        local_hamiltonian += (coef[i]/2) * np.kron(PAULI_MAT['i'], PAULI_MAT[oper_operator])
                        hasoper = True
                    
            elif position_length == 2:
                assert all(np.abs(posn[:,1] - posn[:,0]) == 1), "非最最近邻模型不适合用局域门算法求解"
                indices = np.where(posn[:, 0] == site_position)[0]
                for i in indices:
                    if posn[i, 0] + 1 == posn[i, 1]:
                        local_hamiltonian += coef[i] * np.kron(PAULI_MAT[oper_operator[0]], PAULI_MAT[oper_operator[1]])
                        hasoper = True
                
                indices = np.where(posn[:, 1] == site_position)[0]
                for i in indices:
                    if posn[i, 0] == posn[i, 1] + 1:
                        local_hamiltonian += coef[i] * np.kron(PAULI_MAT[oper_operator[1]], PAULI_MAT[oper_operator[0]])
                        hasoper = True
                
            else:
                raise NotImplementedError("超两体相互作用模型不适合用局域门算法求解")
        return local_hamiltonian, hasoper

    @classmethod
    def _trotter_suzuki_time_steps(cls, order:str) -> list[float]:
        """Return time steps of U for the Suzuki Trotter decomposition of desired order."""
        if order == "1":
            return [1.]
        elif order == "2":
            return [0.5, 1.]
        elif order == "4":
            t1 = 1. / (4. - 4.**(1 / 3.))
            t3 = 1. - 4. * t1
            return [t1/2., t1, (t1+t3)/2., t3]
        elif order == "4_opt":
            # Eq (30a) of arXiv:1901.04974
            a1 = 0.095848502741203681182
            b1 = 0.42652466131587616168
            a2 = -0.078111158921637922695
            b2 = -0.12039526945509726545
            return [a1, b1, a2, b2, 0.5 - a1 - a2, 1. - 2 * (b1 + b2)]  # a1 b1 a2 b2 a3 b3
        # else
        raise ValueError("Unknown order %r for Suzuki Trotter decomposition" % order)

    @classmethod
    def _pseudo_trotter_suzuki_decomposition(cls, order:str, N_steps:int)->list[tuple[int, int]]:
        even, odd = 0, 1
        if N_steps == 0:
            return []
        if order == "1":
            a = (0, odd)
            b = (0, even)
            return [a, b] * N_steps
        elif order == "2":
            a = (0, odd)  # dt/2
            a2 = (1, odd)  # dt
            b = (1, even)  # dt
            # U = [a b a]*N
            #   = a b [a2 b]*(N-1) a
            return [a, b] + [a2, b] * (N_steps - 1) + [a]
        elif order == "4":
            a = (0, odd)  # t1/2
            a2 = (1, odd)  # t1
            b = (1, even)  # t1
            c = (2, odd)  # (t1 + t3) / 2 == (1 - 3 * t1)/2
            d = (3, even)  # t3 = 1 - 4 * t1
            # From Schollwoeck 2011 (:arxiv:`1008.3477`):
            # U = U(t1) U(t2) U(t3) U(t2) U(t1)
            # with U(dt) = U(dt/2, odd) U(dt, even) U(dt/2, odd) and t1 == t2
            # Using above definitions, we arrive at:
            # U = [a b a2 b c d c b a2 b a] * N
            #   = [a b a2 b c d c b a2 b] + [a2 b a2 b c d c b a2 b a] * (N-1) + [a]
            steps = [a, b, a2, b, c, d, c, b, a2, b]
            steps = steps + [a2, b, a2, b, c, d, c, b, a2, b] * (N_steps - 1)
            steps = steps + [a]
            return steps
        elif order == '4_opt':
            # symmetric: a1 b1 a2 b2 a3 b3 a2 b2 a2 b1 a1
            steps = [(0, odd), (1, even), (2, odd), (3, even), (4, odd),  (5, even),
                        (4, odd), (3, even), (2, odd), (1, even), (0, odd)]  # yapf: disable
            return steps * N_steps
        # else
        raise ValueError("Unknown order {0!r} for Suzuki Trotter decomposition".format(order))
    
    
    def to_mpo(self, L=None, pauli=False, backend='torch', device=None):
        if L is None:
            L = max([np.max(posn) for posn, _ in self.data.values()]) + 1 
        if backend == 'torch':
            from ...torch_utils.networks import MPO
            from ...torch_utils.utils import totc 
            import torch as tc # type: ignore
            tt = self.automata(L, pauli=pauli)
            dtype = tc.float64
            for i in tt:
                if np.iscomplexobj(i):
                    dtype = tc.complex128
            return MPO(totc(tt, dtype=dtype, device=device))
        elif backend == 'tensor':
            # todo 
            raise NotImplementedError("还没有实现")
        elif backend == 'quimb':
            import quimb.tensor as qtn
            builder = qtn.SpinHam1D()
            for oper, posn, coef in self.expandxy(pauli).each_term():
                conuntZ = 2**oper.count('Z') # quimb 总是使用 spin oper 需要调整 Z
                if len(posn) == 1:
                    builder[posn[0]] += tuple([coef*conuntZ] + list(oper))
                else:
                    builder[*posn] += tuple([coef*conuntZ] + list(oper))
            return builder.build_mpo(L=L)
        else:
            raise ValueError("backend should be 'torch' or 'tensor'")

    def energies(self, pauli=None, basis=None):
        L = max([np.max(posn) for posn, _ in self.data.values()]) + 1
        assert L < 14, "L should be less than 21, otherwise the matrix size will be too large"
        if basis is None:
            from ..basis import spin_basis
            basis = spin_basis(L)
        mat = self.to_matrix(basis, pauli=pauli, sparse=False)
        isherm = (mat == mat.T.conj()).all()
        return {True: np.linalg.eigvalsh,
                    False: np.linalg.eigvals}[isherm](mat)

    def gdenergy(self, pauli=None, k=1, return_eigenvectors=False, basis=None):
        L = max([np.max(posn) for posn, _ in self.data.values()]) + 1
        assert L < 21, "L should be less than 21, otherwise the matrix size will be too large"
        if basis is None:
            from ..basis import spin_basis
            basis = spin_basis(L)
        mat = self.to_matrix(basis, pauli=pauli, sparse=True)
        isherm = (mat != mat.T.conj()).nnz == 0
        if mat.shape[0] < 1000:
            if not return_eigenvectors:
                return {True: np.linalg.eigvalsh,
                        False: np.linalg.eigvals}[isherm](mat.todense())[:k]
            else:
                val, vec = {True: np.linalg.eigh,
                        False: np.linalg.eig}[isherm](mat.todense())
                return val[:k], vec[:, :k]
        else:
            if isherm:
                return sp.linalg.eigsh(mat, k=k, which='SA', return_eigenvectors=return_eigenvectors)
            else:
                return sp.linalg.eigs(mat, k=k, which='LM', return_eigenvectors=return_eigenvectors)

    def evolve(self, inistate:np.ndarray, tlist:np.ndarray, measure, basis=None, L=None, pauli=False):
        """计算观测量演化的示例

        Parameters
        ----------
        inistate : np.ndarray
            初始态
        tlist : np.ndarray
            时间
        measure : list[SpinOper] | Callable[[np.ndarray], np.ndarray]
            观测量列表
        basis : Basis, optional
            基矢, by default None
        L: int, optional
            系统的长度, by default None
        pauli : bool, optional
            是否使用 pauli 矩阵表示, by default False

        Returns
        -------
        np.ndarray
            对应观测量的演化
        
        Example
        -------
        >>> import quante as qt
        >>> import numpy as np
        >>> op = qt.generate.operas
        >>> L = 10
        >>> tlist = np.linspace(0, 10, 200)
        >>> # Model
        >>> J, γ = 1., 0.
        >>> builder = op.SpinOperBuilder()
        >>> for l in range(L-1):
        >>>     builder += 1/2 * (J + γ), 'p', l+1, 'm',   l
        >>>     builder += 1/2 * (J - γ), 'p',   l, 'm', l+1
        >>> ham = builder.build()
        >>> basis = qt.generate.basis.spin_basis(L=L, Nup=L//2)
        >>> obsoper = [op.z(i) for i in range(L)]
        >>> init_state = qt.generate.state.neel(L=L, down_first=True, basis=basis)
        >>> res = ham.evolve(init_state, tlist, obsoper, basis=basis)
        >>> res
        """
        if L is None:
            L = max([np.max(posn) for posn, _ in self.data.values()]) + 1

        if basis is None:
            from ..basis import spin_basis
            basis = spin_basis(L)
        assert basis.Ns == len(inistate), "inistate should be the same length as basis"

        # Method to get evolve expectation values
        if basis.Ns < 2**12:  # 小尺寸的做法
            ###################################################################################
            # 严格对角化的写法
            ###################################################################################
            from ...linalg import get_time_evolution_states_ED, observe_states
            engres = np.linalg.eigh(self.to_matrix(basis, pauli=pauli))
            evalstate = get_time_evolution_states_ED(inistate, *engres, tlist, failback_to_CPU=True)
            if isinstance(measure, list):
                return np.real_if_close([observe_states(evalstate, obs.to_matrix(basis, pauli=pauli)) for obs in measure]).T
            else:
                return np.real_if_close([measure(evalstate[:, i]) for i in range(len(tlist))])
        else:
            hammat = self.to_matrix(basis, pauli=pauli, sparse=True)
            try:
            ###################################################################################
            # GPU expm_multiply
            ###################################################################################
                from ...torch_utils.linalg.expm_multiply import EvolveEngine
                from ...torch_utils.linalg.sparse import to_csr
                from ...torch_utils.utils import totc
                from tqdm import tqdm
                import torch as tc # type: ignore
                assert tc.cuda.is_available(), "CUDA is not available"
                device = tc.device('cuda')
                hammat0 = to_csr(hammat, device=device)
                inistate = totc(inistate, device=device)
                evolve_engine = EvolveEngine(hammat0, inistate, ts=tlist, device=device)
                if isinstance(measure, list):
                    obsmatlist = [to_csr(o.to_matrix(basis, pauli=pauli, sparse=True), device=device, dtype=tc.complex128) for o in measure]
                    obs = lambda state: [state.conj().reshape(1,-1) @ (obsmat @ state).reshape(-1,1) for obsmat in obsmatlist]
                else:
                    obs = measure
                res = []
                for _ in tqdm(tlist, ascii=True):
                    state = evolve_engine.run()
                    res.append(obs(state))
                return np.real_if_close(res)
            except:
            ###################################################################################
            # CPU parallel expm_multiply
            ###################################################################################
                from ...linalg import EvolveEngine
                from tqdm import tqdm
                evolve_engine = EvolveEngine(hammat, inistate, ts=tlist)
                if isinstance(measure, list):
                    obsmatlist = [to_csr(o.to_matrix(basis, pauli=pauli, sparse=True), device=device, dtype=tc.complex128) for o in measure]
                    obs = lambda state: [state.conj().reshape(1,-1) @ (obsmat @ state).reshape(-1,1) for obsmat in obsmatlist]
                else:
                    obs = measure
                res = []
                for _ in tqdm(tlist, ascii=True):
                    state = evolve_engine.run()
                    res.append(measure)
                return np.real_if_close(res)    



def catposcoef(posn1, coef1, posn2, coef2):
    len1, len2 = len(coef1), len(coef2)
    operlen1, operlen2 = posn1.shape[1], posn2.shape[1]
    # 创建索引网格
    i_indices, j_indices = np.indices((len1, len2))
    # 使用广播生成结果位置
    res_pos = np.empty((len1 * len2, operlen1 + operlen2), dtype=np.int32)
    res_pos[:, :operlen1] = posn1[i_indices.ravel()]  # 直接使用索引赋值
    res_pos[:, operlen1:] = posn2[j_indices.ravel()]  # 直接赋值
    return res_pos, np.kron(coef1, coef2)


def _expand_term(name, c):
    """Expand the term based on the given name and coefficient."""
    # Initialize with base case
    expanded_names = ['']
    expanded_coefs = [1]

    for char in reversed(name):  # Process characters from the end to the start
        if char == 'x':
            prefixes = ['p', 'm']
            factors = [c, c]
        elif char == 'y':
            prefixes = ['p', 'm']
            factors = [-1j * c, 1j * c]
        elif char == 'z':
            prefixes = ['Z']
            factors = [c]
        else:
            prefixes = [char]
            factors = [1]

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


def _single_term(i, coef):
    return (np.array([list(i)], dtype=int), np.array([coef]))

def I(i:int=0) -> SpinOper:
    return SpinOper({'I': _single_term((0,), 1.)})

def p(i:int=0) -> SpinOper:
    return SpinOper({'p': _single_term((i,), 1.)})

def m(i:int=0) -> SpinOper:
    return SpinOper({'m': _single_term((i,), 1.)})

def x(i:int=0) -> SpinOper:
    return SpinOper({'x': _single_term((i,), 1.)})

def y(i:int=0) -> SpinOper:
    return SpinOper({'y': _single_term((i,), 1.)})

def z(i:int=0) -> SpinOper:
    return SpinOper({'z': _single_term((i,), 1.)})

def n(i:int=0) -> SpinOper:
    return SpinOper({'pm': _single_term((i, i), 1.)})

def nn(i:int, j:int) -> SpinOper:
    return SpinOper({'pmpm': _single_term((i, i, j, j), 1.)})

def zz(i:int, j:int) -> SpinOper:
    return SpinOper({'zz': _single_term((i, j), 1.)})

def mp(i:int, j:int) -> SpinOper:
    return SpinOper({'mp': _single_term((i, j), 1.)})

def pm(i:int, j:int) -> SpinOper:
    return SpinOper({'pm': _single_term((i, j), 1.)})

def xx(i:int, j:int) -> SpinOper:
    return SpinOper({'xx': _single_term((i, j), 1.)})

def yy(i:int, j:int) -> SpinOper:
    return SpinOper({'yy': _single_term((i, j), 1.)})

def xy(i:int, j:int) -> SpinOper:
    return SpinOper({'xy': _single_term((i, j), 1.)})

def yx(i:int, j:int) -> SpinOper:
    return SpinOper({'yx': _single_term((i, j), 1.)})

def sum(oper) -> Oper:
    # lazy sum
    data = {}
    stype = None
    for opx in oper:
        if isinstance(opx, (int,float,complex)):
            iterterm = (('I', (np.array([[0]], dtype=int), np.array([opx]))),)
        else:
            iterterm = opx.data.items()
            if stype is None:
                stype = opx.type
            else:
                assert stype == opx.type, "Operands must have the same stype"
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
        stype = 's'
    return SpinOper(newdata)


class SpinBuilder(SpinOper):
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

    def __iadd__(self, term) -> 'SpinBuilder':
        if isinstance(term, tuple):
            assert len(term) % 2 == 1, f"length wrong for term: {term}"
            for i in range(1, len(term), 2):
                assert term[i] in ['I', 'p', 'm', 'x', 'y', 'z', '+', '-'], "term must be a tuple of I, p, m, '+', '-', x, y, or z"
        
            posn = np.array(term[2::2], dtype=int)
            inc_indx = np.argsort(posn, kind='stable')
            posn = posn[inc_indx]
            
            opnm = "".join(term[2*i+1] for i in inc_indx)
            # 把字符串中的 + 和 - 替换成 p 和 m
            opnm = opnm.replace('+', 'p')
            opnm = opnm.replace('-', 'm')
        
            posnlist, coeflist = self.terms.setdefault(opnm, [[], []])
            posnlist.append(posn)
            coeflist.append(np.array([term[0]]))
            return self
        else:
            return super().__iadd__(term)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        data = {}
        for name, (posnlist, coeflist) in self.terms.items():
            data[name] = (np.vstack(posnlist), np.hstack(coeflist))
        super().__init__(data, type='s')

        if exc_type is not None:  # 检查是否发生错误
            tb.print_exc()  # 打印堆栈跟踪
    
    def build(self):
        data = {}
        for name, (posnlist, coeflist) in self.terms.items():
            data[name] = (np.vstack(posnlist), np.hstack(coeflist))
        return SpinOper(data, 's')


SpinOperBuilder = SpinBuilder  # 兼容性

class HeisenbergOper(SpinOper):
    def __init__(self, data, type='s'):
        super().__init__(data, type=type)
        
    @classmethod
    def _make_spinoper(cls, L, j=1.0, h=0.0, cyclic=False):
        cls.L = L
        cls.cyclic = cyclic
        try:
            jx, jy, jz = j # type: ignore
        except TypeError:
            jx = jy = jz = j
        try:
            hx, hy, hz = h # type: ignore
        except TypeError:
            hz = h
            hx = hy = 0.0
        cls.jx = jx
        cls.jy = jy
        cls.jz = jz
        cls.hx = hx
        cls.hy = hy
        cls.hz = hz
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
        return HeisenbergOper(data)

    def energies(self, isinf=False, pauli=False):
        from ...solvable_models.free_fermion import free_fermion as ff
        L = self.L if not isinf else np.inf
        if self.jz == self.hx == self.hy == 0 and not self.cyclic:
            # xy model
            return ff.XY_energies(L=L, jxx=self.jx, jyy=self.jy, hz=self.hz, pauli=pauli)
        
            # todo 其它的结论？
            
        if np.isinf(L):
            raise ValueError("Infinite system size is not supported")
        return super().energies(pauli=pauli)
        

    def gdenergy(self, isinf=False, pauli=False, *, k=1, return_eigenvectors=False):
        if not return_eigenvectors:
            from ...solvable_models.free_fermion import free_fermion as ff
            L = self.L if not isinf else np.inf
            if self.hx == self.hy == self.hz == 0 and self.jx == self.jy == self.jz and not np.isinf(L) and self.cyclic and k == 1:
                print("approximate: ", end='')
                return ff.XXX_gdenergy_pbc_approx(L) * (4 if pauli else 1)
            if self.jz == self.hx == self.hy == 0 and not self.cyclic and k == 1:
                # xy model
                return ff.XY_gdenergy(L=L, jxx=self.jx, jyy=self.jy, hz=self.hz, pauli=pauli)
            
            # todo 其它的结论？
            
        if np.isinf(L):
            raise ValueError("Infinite system size is not supported")
        return super().gdenergy(pauli=pauli, k=k, return_eigenvectors=return_eigenvectors)
    
        
def heisenberg_operator(L, j=1.0, h=0.0, cyclic=False) -> HeisenbergOper:
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
    return HeisenbergOper._make_spinoper(L=L, j=j, h=h, cyclic=cyclic)