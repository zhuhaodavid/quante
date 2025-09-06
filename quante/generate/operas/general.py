# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-05-17 22:07:46
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-07 00:13:03

import numpy as np
import copy 

def _isscale(i):
    return np.isscalar(i) or i.__class__.__module__.startswith('sympy.')

def _single_term(i, coef):
    return (np.array([list(i)], dtype=int), np.array([coef]))

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

    # Sort positions using custom argsort
    sorted_indices = _argsort_positions(res_pos)
    res_pos = res_pos[sorted_indices]
    res_coef = res_coef[sorted_indices]

    return _quick_merge_batch(res_pos, res_coef)

def _quick_merge_batch(res_pos, res_coef):
    # 如果 res_coef 是数字而不是 object 那么就可以调用 numba
    if res_coef.dtype != object:
        from .nbfuc.general_nb import _quick_merge
        return _quick_merge(res_pos, res_coef)

    total_len = len(res_pos)
    cur_coef = res_coef[0]
    cur_pos = 0
    prev_pos = res_pos[0]  # 引入局部变量存储上一个位置
    for i in range(1, total_len):
        tmp = res_pos[i]
        if (tmp == prev_pos).all():
            cur_coef += res_coef[i]
        else:
            res_pos[cur_pos] = prev_pos
            res_coef[cur_pos] = cur_coef
            cur_pos += 1
            cur_coef = res_coef[i]
            prev_pos = tmp
            
    res_pos[cur_pos] = res_pos[total_len-1]
    res_coef[cur_pos] = cur_coef
    mask = res_coef[:cur_pos + 1] != 0  # Remove zero coefficients
    return res_pos[:cur_pos+1][mask], res_coef[:cur_pos+1][mask]

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
    
    def __iadd__(self, oper, add_or_minus=1):
        """ self += oper """
        if _isscale(oper):
            if oper == 0:
                return self
            else:
                old_pos, old_coef = self.data.get('I', (None,None))
                if old_coef is None and old_pos is None:
                    self.data['I'] = (np.array([[0]],dtype=int), np.array([oper]))
                else:
                    old_coef[0] += add_or_minus*1
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
    
    def __add__(self, oper):
        """ self + a * oper """
        self_copy = self.copy()
        self_copy.__iadd__(oper)
        return self_copy

    def copy(self):
        return copy.deepcopy(self)
        # cls = self.__class__
        # return cls(copy.deepcopy(self.data), self.type)
    
    def __radd__(self, oper):
        """ num + oper """
        return self.__add__(oper)  # a + b = b + a
    
    def __isub__(self, oper):
        """self -= oper"""
        self.__iadd__(oper, add_or_minus=-1)
        return self
    
    def __sub__(self, oper):
        """self - oper"""
        self_copy = self.copy()
        self_copy.__iadd__(oper, add_or_minus=-1)
        return self_copy
    
    def __imul__(self, oper):
        """self *= oper"""
        if _isscale(oper):
            scale = oper
            for name, (posn, coef) in self.data.items():
                try:
                    coef *= scale
                except TypeError:
                    self.data[name] = (posn, coef * scale)
            return self
        elif isinstance(oper, Oper):
            return self.__matmul__(oper)
        else:
            raise NotImplementedError(f"oper type {type(oper)} not supported")
    
    def __mul__(self, scale):
        """ oper * num """
        if _isscale(scale):
            self_copy = self.copy()
            self_copy.__imul__(scale)
            return self_copy
        return self.__matmul__(scale)
        
    def __rmul__(self, scale):
        """ num * oper """
        if isinstance(scale, Oper):
            return self.__matmul__(scale)
        return self.__mul__(scale)

    def __rsub__(self, oper):
        """oper - self"""
        self_copy = self.copy()
        self_copy *= -1
        # self_copy.__imul__(-1)
        self_copy.__iadd__(oper)
        return self_copy
    
    def __neg__(self):
        """- self"""
        return self.__mul__(-1)
    
    def __truediv__(self, num):
        """self / num"""
        return self.__mul__(1/num)

    def __matmul__(self, oper:'Oper'):
        """ self * oper """
        assert isinstance(oper, Oper), NotImplementedError(f"oper type {type(oper)} not supported")
        if self.type == oper.type and len(oper.type) == 1:  # 相同类型
            cls = self.__class__
            newoper = cls({}, self.type)
            for opnm1, (posn1, coef1) in self.data.items():
                if opnm1 == "I":
                    newoper.__iadd__(np.sum(coef1) * oper)
                    continue
                for opnm2, (posn2, coef2) in oper.data.items():
                    if opnm2 == "I":
                        newoper.__iadd__(np.sum(coef2) * cls({opnm1: (posn1, coef1)}, self.type))
                        continue
                    newopnm = opnm1 + opnm2
                    newposn, newcoef = catposcoef(posn1, coef1, posn2, coef2)
                    newoper.__iadd__(cls({newopnm : (newposn, newcoef)}, self.type))
            return newoper
        else:
            raise NotImplementedError("不同基矢相加")
    
    def __pow__(self, n, m=None):
        """ self ** n """
        if m is not None:
            raise NotImplementedError("modulo is not implemented")
        if n <= 0:
            raise NotImplementedError("inverse is not implemented")
        newoper = self
        for _ in range(n - 1):
            newoper = self * newoper
        return newoper
    
    def _check_length(self, L:int):
        assert L >= self.L
    
    def show_string_form(self, maxlen=80, form='h'):
        """打印算符的字符串形式"""
        if form == 'v':
            print(self.table_form(maxlen=maxlen))
        elif form == 'h':
            print(self.table_form2(maxlen=maxlen))
        else:
            raise NotImplementedError("form should be 'v' or 'h'")
    
    def table_form(self, maxlen=90) -> str:
        _seperate_notion = self._seperate_notion()
        if len(self.data) == 0:
            return "0"
        pages = []
        first_line = "|"
        second_line = "|"
        data_list = []
        last_len = 0
        for i, (operator, (posn, coef)) in enumerate(self.data.items()):
            
            oper_len = len(operator) - operator.count('|')
            
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
                    data_line += " " * (last_len - len(data_line)-1) + "|"
                for j in range(oper_len):
                    data_line += f"   {posn[i][j]:<3}"

                if coef.dtype == object:
                    from sympy import nsimplify
                    tmp = f"{nsimplify(coef[i])}".rjust(10) + " |"
                elif np.iscomplexobj(coef):
                    if max(np.abs(coef)) < 1e3 and min(np.abs(coef)) > -1e3:
                        tmp = f"{coef[i]:.3f}".rjust(16) + " |"  # 普通浮点数格式
                    else:
                        tmp = f"{coef[i]:.2e}".rjust(20) +" |"  # 科学计数法格式
                else:
                    if max(coef) < 1e3 and min(coef) > -1e3:
                        tmp = f"{coef[i]:.3f}".rjust(10) + " |"
                    else:
                        tmp = f"{coef[i]:.2e}".rjust(12) +" |"  # 科学计数法格式
                
                data_line += tmp
                if len(tmp) > ml:
                    ml = len(tmp)
                data_list[i] = data_line
            
            last_len = len(data_line)
        
            hasvbar = False
            for i in operator:
                if i == '|':
                    hasvbar = True
                    continue
                first_line += f" {_seperate_notion} {i:<3}" if hasvbar else f"   {i:<3}"
                hasvbar = True
                second_line += "-"*6
                hasvbar = False
            makeup = ml - 12
            if hasvbar:
                first_line += f" {_seperate_notion}" + " "*makeup + "   coef. |"
            else:
                first_line += "  " + " "*makeup + "   coef. |"
            second_line += "-"*(11+makeup) + "|"
            
        pages.append(first_line)
        pages.append(second_line)
        pages += data_list

        # 增加 self 的名字和地址
        prefix = self._prefix()
        return prefix + '\n'.join(pages) + '\n'
    
    def _prefix(self) -> str:
        return f"{self.__class__.__name__} at {hex(id(self))}, \n"
    
    def _seperate_notion(self) -> str:
        return "|"
    
    def table_form2(self, maxlen=90) -> str:
        lines = []
        for operator, (posn, coef) in self.data.items():
            oper_len = len(operator) - operator.count('|')
            for i in range(len(coef)):
                # if i > 0:
                #     line += " + "
                line = f"{operator}, "
                if coef.dtype == object:
                    from sympy import nsimplify
                    coefstr = f"{nsimplify(coef[i])}"
                elif np.iscomplexobj(coef):
                    if max(np.abs(coef)) < 1e3 and min(np.abs(coef)) > -1e3:
                        coefstr = f"{coef[i]:.3f}"
                    else:
                        coefstr = f"{coef[i]:.2e}"
                else:
                    if max(coef) < 1e3 and min(coef) > -1e3:
                        coefstr = f"{coef[i]:.3f}"
                    else:
                        coefstr = f"{coef[i]:.2e}"

                line += "(" + ", ".join([f"{posn[i][j]}" for j in range(oper_len)]) + "), " + coefstr
                lines.append(line)
        return '\n'.join(lines)
        # print('\n'.join(lines))
    
    def __str__(self) -> str:
        """
        返回算符的字符串形式
        """
        return self.table_form2(maxlen=80)
    
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
    def load(cls, filename:str):
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
    
    def expandn(self):
        res = {}
        for oper, (posnlist, coeflist) in self.data.items():
            if 'n' in oper:
                positions = []
                newoper = ''
                for i, char in enumerate(oper):
                    if char == 'n':
                        newoper += '+-'
                        positions.append(i)
                    else:
                        newoper += char
                for pos in sorted(positions, reverse=True):  # 从后往前插入，避免索引偏移
                    posnlist = np.insert(posnlist, pos + 1, posnlist[:, pos], axis=1)
                posnlist_ = posnlist
            else:
                newoper = oper
                posnlist_ = posnlist.copy()
            res[newoper] = (posnlist_, coeflist.copy())
        return type(self)(res)

    def subs(self, dic):
        res = {}
        for oper, (posnlist, coeflist) in self.data.items():
            newcoeflist = np.array([
                coef.subs(dic).evalf() for coef in coeflist
            ])
            res[oper] = (posnlist, newcoeflist)
        return type(self)(res)