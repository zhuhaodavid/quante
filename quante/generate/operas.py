# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-08-19 12:52:13
# @Last Modified by:   hzhu
# @Last Modified time: 2024-11-21 18:31:05
   
#!! 包里的其他文件不要 import 这个 operas.py !!!!

import numpy as _np
import scipy as _sp
import copy as _copy
import warnings as _warnings
from typing import Union, Generator, Optional, Type, Callable, Iterable

number = Union[int, float, complex]
OperDataType = dict[str, dict[tuple[int,...], number]]

from ..basicfun import PrintLn # type: ignore

@PrintLn.add_object_print
class Oper:
    """
    算符类: 该类用于表示和操作量子系统的算符。
    
    提供最重要的功能包括:
    
    - 算符的加减、乘法、幂运算, 以及将 x,y 用 +,- 展开 (`expandxy`)

    - 转化为 quspin 接受的格式 (`quspin_form`)

    - 算符生成矩阵 (`to_matrix`) 、MPO (`automata`)

    - 最近邻相互作用的二体门分解 (`gate2_decomposition`,  `suzuki_trotter_decomposition`)

    - 其他方便的函数 (`copy`, `each_term`)

    算符的数据结构：
    
    .. code-block:: python
    
        data = {
            operator_1: {position1: coefficient1, 
                        position2: coefficient2,
                        ...},
            operator_2: {position1: coefficient3, 
                        position2: coefficient4,
                        ...}, 
            ...
        }
    """
    def __init__(self, data:OperDataType, type:str = "s") -> None:  # todo 处理费米子系统
        self.data = data
        self.type = type
        
    def copy(self) -> 'Oper':
        return Oper(_copy.deepcopy(self.data), self.type)

    def __add__(self, oper:Union['Oper', number], a=1.) -> 'Oper':
        """ self + a * oper """
        if isinstance(oper, (int, float, complex)) and oper == 0:  # a + 0 = a
            return self
        elif isinstance(oper, (int, float, complex)):  # 加单位阵
            new_oper = self.copy()
            new_oper._add_single_oper('I', (0,), oper)
            return new_oper
        elif isinstance(oper, Oper):  # 两个算符相加
            assert self.type == oper.type, NotImplementedError("算符类型不相同")
            new_oper = self.copy()
            for operator, position, coefficient in oper.each_term():
                new_oper._add_single_oper(operator, position, coefficient*a)  # 逐项添加
            return new_oper
        else:
            raise NotImplementedError(f"oper type {type(oper)} not supported")

    def _add_single_oper(self, operator: str, position: tuple[int, ...], coefficient: number) -> None:
        """添加 opnm, posn, coef"""
        operator_data = self.data.setdefault(operator, {})  # 找到算符对应的数据
        previous_coef = operator_data.setdefault(position, 0.)  # 找到相应位置的系数
        updated_coef = previous_coef + coefficient
        if _np.isclose(updated_coef, 0.0):
            del operator_data[position]  # 如果新系数为 0，则删除 posn 键
            if not operator_data:
                del self.data[operator]  # 如果 opnm_data 变为空字典，则删除 opnm 键
        else:
            operator_data[position] = _np.real_if_close(updated_coef).item()

    def each_term(self) -> Generator[tuple[str, tuple[int,...], number], None, None]:
        """
        
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
        for operator, operator_data in self.data.items():
            for position, coefficient in operator_data.items():
                yield operator, position, coefficient

    def __radd__(self, oper:Union['Oper', number]) -> 'Oper':
        """ num + oper """
        return self.__add__(oper)  # a + b = b + a
    
    def __mul__(self, scale:Union['Oper', number]) -> 'Oper':
        """ oper * num """
        if isinstance(scale, Oper): return self.__matmul__(scale)
        if scale == 0: return Oper({}, self.type)
        newdata = {}
        for key, opnmdata in self.data.items():
            newdata[key] = {k: v*scale for k, v in opnmdata.items()}
        return Oper(newdata, self.type)

    def __rmul__(self, scale) -> 'Oper':
        """ num * oper """
        if isinstance(scale, Oper):
            return self.__matmul__(scale)
        return self * scale
    
    def __sub__(self, oper:Union['Oper', number]) -> 'Oper':
        """self - oper"""
        return self.__add__(oper, a=-1.)

    def __rsub__(self, oper:Union['Oper', number]) -> 'Oper':
        """oper - self"""
        return ((-1.)*self).__add__(oper)

    def __neg__(self) -> 'Oper':
        """- self"""
        return (-1) * self

    def __truediv__(self, num:number) -> 'Oper':
        """self / num"""
        return (1 / num) * self
    
    def __matmul__(self, oper:'Oper') -> 'Oper':
        """ self * oper """
        if self.type == oper.type and len(oper.type) == 1:  # 相同类型
            newoper = Oper({}, self.type)
            for opnm1, opnmdata1 in self.data.items():
                if opnm1 == "I":
                    coef_sum = 0
                    for _, coef in opnmdata1.items():
                        coef_sum += coef
                    newoper += coef_sum * oper
                    continue
                for opnm2, opnmdata2 in oper.data.items():
                    if opnm2 == "I":
                        coef_sum = 0
                        for _, coef in opnmdata2.items():
                            coef_sum += coef
                        newoper += coef_sum * Oper({opnm1: opnmdata1}, self.type)
                        continue
                    newopnm = opnm1 + opnm2
                    newopnmdata = {}
                    for posn1, coef1 in opnmdata1.items():
                        for posn2, coef2 in opnmdata2.items():
                            newopnmdata[posn1 + posn2] = coef1 * coef2
                    newoper += Oper({newopnm : newopnmdata}, self.type)
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

    def show_string_form(self) -> None:
        operator_string = ""
        for operator, operator_data in self.data.items():
            operator_string += operator + "\n"
            for position, coefficient in operator_data.items():
                operator_string += "  " + position.__str__() + " " + coefficient.__repr__() + "\n"
        print(operator_string)
    
    def show(self, whichonm=None) -> None:
        import matplotlib.pyplot as plt
        try:
            import igraph as ig
        except ImportError:
            raise ImportError("igraph is not installed, please install it first: pip install igraph")
        
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
        from ..basicfun import save_hdf5
        save_hdf5(filename, '/', {"data": self.data, "type":self.type})

    @classmethod
    def load(cls, filename:str) -> 'Oper':
        from ..basicfun import load_hdf5
        data = load_hdf5(filename, '/', '/')
        dic = {}
        for key, val in data['data'].items():
            indic = {}
            for inkey, inval in val.items():
                indic[eval(inkey)] = inval
            dic[key] = indic
        return Oper(dic, data['type'].decode('utf-8'))

    def sort_posn(self) -> 'Oper':
        """将位置按照从小到大顺序排列
        
        Examples
        --------
        >>> res = op.xx(2,1).sort_posn()
        
        注, 对费米子会报错
        """
        if self.type == 'f':
            raise NotImplementedError("费米子排序没有实现")
        newoper = Oper({}, self.type)
        for opnm, posn, coef in self.each_term():
            zipstr = list(zip(opnm, posn))
            zipstr.sort(key=lambda x: x[1])
            opnm, posn = zip(*zipstr)
            posn = tuple(posn)
            opnm = "".join(opnm)
            newoper._add_single_oper(opnm, posn, coef)
        return newoper

    def expandxy(self, pauli:bool = False) -> 'Oper':
        """
        
        展开算符中的 `x`,`y`，同时将 `z` 替换为 `Z` 
        其中 `Z` = `pm`-`mp`，这是为了 `to_matrix` 方便
        
        展开之后，应当只包含 `p`, `m`, `i`, `Z` 这三种算符
        
        Examples
        --------
        >>> ham = op.heisenberg_operator(L=4)
        >>> ham = ham.expandxy()
        >>> ham.show_string_form()
        
        注:只是自旋
        """
        if self._has_expanded():
            return self.copy()
        if self.type == "s":
            c = 1.0 if pauli else 0.5
            
            # 并行的计算求和
            from ..linalg.usenumba.numba_settings import get_thread_pool, parallel_reduce
            pool = get_thread_pool()
            
            def gen_term(arg):
                opnm, posn, coef = arg
                newop_eachterm = 1.
                for opnm_i, posn_i in zip(opnm, posn):
                    if opnm_i == 'x':  # x -> (p + m)/2
                        x = Oper({"p":{(posn_i,):c}, "m":{(posn_i,):c}}, "s")
                        newop_eachterm = newop_eachterm * x
                    elif opnm_i == 'y':  # x -> (p - m)/2i
                        y = Oper({"p":{(posn_i,):-1j*c}, "m":{(posn_i,):1j*c}}, "s")
                        newop_eachterm = newop_eachterm * y
                    elif opnm_i == 'z':  # z -> Z
                        z = Oper({"Z":{(posn_i,): c}}, "s")
                        newop_eachterm = newop_eachterm * z
                    else:
                        o = Oper({opnm_i:{(posn_i,): 1.0}})
                        newop_eachterm = newop_eachterm * o
                return newop_eachterm * coef
            
            newop = parallel_reduce(lambda a,b : a+b, pool.map(gen_term, self.each_term()))
            return newop

        else:
            raise NotImplementedError()
    
    def _has_expanded(self) -> bool:
        for opnm, _, _ in self.each_term():
            if 'x' in opnm or 'y' in opnm or 'z' in opnm:
                return False
        return True
    
    def dtype(self) -> Type[Union[float, complex]]:
        tmp = self if self._has_expanded() else self.expandxy()
        for _, _, coef in tmp.each_term():
            if isinstance(coef, complex):
                return complex
        return float
    
    def quspin_form(self) -> list[list[list[number]]]:
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
        for opnm, opnm_data in self.data.items():
            static_bond = []
            c = opnm.count('Z')
            for posn, coef in opnm_data.items():
                static_bond.append([coef*2**c] + list(posn))
            static.append([opnm.replace('m', '-').replace('p', '+').replace('Z', 'z'), static_bond])
        return static
    
    def to_matrix(self, basis, pauli=None, sparse=False, savememory=False) -> Union[_np.ndarray, _sp.sparse.csr_array]:
        """
        生成哈密顿量在给定基矢下的矩阵，对于自旋 1/2 默认使用 symmetrize 的方法计算矩阵元
        
        pauli 默认使用的是 False
        
        .. 警告::
            这个函数不检查哈密顿量是否有对称性。如果哈密顿量没有对称性，那么这个函数会返回错误的结果，而不会报错。
        
        Examples
        --------
        >>> L = 10
        >>> basis = qt.generate.basis.spin_basis(L=L, Nup=5)
        >>> ham = qt.generate.operas.heisenberg_operator(L=L)
        >>> mat = ham.to_matrix(basis)
        >>> print(mat)
        
        时间对比参考 example/exact_diagonalization.ipynb
        
        对于**没有对称性**的基矢，automata 收缩是最快的方法：
        
        >>> from quante.tensor.automata import get_sparse_matrix
        >>> mat = get_sparse_matrix(L, *ham.split_data(), pauli=pauli, usecuda=True)
        
        对于**没有对称性**的基矢，也可以使用 `to_matrix_cuda` 来实现：
        
        >>> from quante.torch_utils.symmetry import to_matrix_cuda
        >>> eachterm, hascomplex = ham.expandxy(False)._convert_to_quick_form()
        >>> mat = to_matrix_cuda(basis, eachterm, hascomplex)
        
        有对称性的,也可以这么使用,但是加速并不明显.
        
        如果反复生成也可以将 to_matrix 拆开来：
        
        >>> eachterm, hascomplex = ham.expandxy(pauli=pauli)._convert_to_quick_form()
        >>> mat = basis._sparse_matrix(eachterm, hascomplex)
        
        可以反复使用 `basis._sparse_matrix(eachterm, hascomplex)`
        """
        if self.data == {}:
            return None
        self._check_length(basis.L)
        from .symmetry.basis_class import SpinHalfBasis, SpinHighBasis
        if isinstance(basis, (SpinHalfBasis, SpinHighBasis)):
            if basis.S != 0.5 and pauli is True:
                raise KeyError("自旋不是 1/2，不能使用 Pauli 矩阵")
            if self._has_expanded():
                if pauli is not None:
                    _warnings.warn("pauli in to_matrix is not used")
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
            raise NotImplementedError("不支持的基矢类型")
        
    def _check_length(self, L:int) -> None:
        for _, position, _ in self.each_term():
            for pos in position:
                if pos >= L:
                    raise ValueError(f"position {pos} is larger than L {L}")
    
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
            opnm_list = _np.array(opnm_list, dtype=_np.int64)
            posn_list = _np.array(posn, dtype=_np.int64)
            coef_real = _np.real_if_close(coef).item()
            if isinstance(coef_real, complex):
                hascomplex = True
            eachterm.append((opnm_list, posn_list, coef_real))
        return eachterm, hascomplex

    def automata(
        self,
        L: int,
        pauli: bool = False,
        d: int = 2,
        local_matrix: Optional[Callable[[str], _np.ndarray]] = None,
        dtype: Type[_np.complex128] = _np.complex128,
    ) -> list[_np.ndarray]:
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
        gen_matrix : Optional[Callable[[str], _np.ndarray]], optional
            用于生成局部矩阵的函数。如果提供，该函数将根据字符串参数生成对应的局部矩阵。默认为 None，即使用默认的局部矩阵生成方式。
        dtype : Type[_np.complex128], optional
            局部矩阵的数据类型。默认为 _np.complex128，即复数类型。

        Examples
        --------
        >>> L = 10
        >>> ham = op.heisenberg_operator(L)
        >>> basis = (L, pauli=False)
        >>> mpo = ham.automata(L, pauli=False)
        
        """
        from ..tensor.automata import automata_mpo
        hlocals, positions, coefficients = self.expandxy(pauli=pauli).split_data()
        coefficients = _np.real_if_close(coefficients)
        return automata_mpo(L, hlocals, positions, coefficients, d=d, pauli=pauli, local_matrix_function=local_matrix, dtype=coefficients.dtype)

    def split_data(self) -> tuple[list[str], list[tuple[int]], list[number]]:
        """这个函数是为 automata 写的，但 parallel_matrix 等函数可能会用到"""
        operators, positions, coefficients = [], [], []
        for operator, position, coefficient in self.each_term():
            operators.append(operator)
            positions.append(position)
            coefficients.append(coefficient)
        return operators, positions, coefficients
    
    def gate2_decomposition(self, L:int, tau:float, form="ladder", pauli:bool=True) -> tuple[list[int],list[_np.ndarray]]:
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
            
            gate_positions_in_layer_i, local_hamiltonian_matrices = self._get_local_hamiltonians(L, increment, layer=layer, pauli=pauli)
            
            if layer_index == 0:
                # 从左到右的门
                positions += gate_positions_in_layer_i
                # 将局域哈密顿量转换为局域演化门：expm
                gates += [_sp.linalg.expm((-1j*tau)*matrix) for matrix in local_hamiltonian_matrices]
            else:
                # 从右到左的门，如果 layer_number = 1，那么这个部分不会执行
                positions += gate_positions_in_layer_i[::-1]
                # 将局域哈密顿量转换为局域演化门：expm
                gates += [_sp.linalg.expm((-1j*tau)*matrix) for matrix in local_hamiltonian_matrices[::-1]]

        return positions, gates

    def trotter_gates(self, L:int, tau:float, order:str="2", evolve_type:str="time", N_step:int=1, pauli:bool=True) -> tuple[list[int], list[_np.ndarray]]:
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
        from ..linalg import expm
        # 获取偶数位置和局域哈密顿量
        even_positions, even_hamiltonians = self._get_local_hamiltonians(L, increment=2, layer="even", pauli=pauli)
        # 获取奇数位置和局域哈密顿量
        odd_positions, odd_hamiltonians =  self._get_local_hamiltonians(L, increment=2, layer="odd", pauli=pauli)

        # 遍历Suzuki-Trotter时间步长 不同的阶数将 tau 拆分成了不同种 dt，每一种会给出一组演化算符，将它们都记录下来
        if evolve_type == "time":
            c = -1.j
        elif evolve_type == "temporal":
            c = -1.
        else:
            raise ValueError("type should be 'time' or 'temporal'")

        even_gates_dts = []
        odd_gates_dts = []
        
        for dt in Oper._trotter_suzuki_time_steps(order):
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
        for dt_index, evolve_even_gate in Oper._pseudo_trotter_suzuki_decomposition(order, N_step):
            
            direction = 1 if evolve_even_gate else -1  # 如果是偶数，方向是 1，表示从左向右演化，如果是奇数，方向是 -1，表示从右向左演化
            
            # 根据even的值选择演化门和位置列表
            single_step_gates = even_gates_dts if evolve_even_gate else odd_gates_dts
            gates += single_step_gates[dt_index][::direction]  # 演化 single_step_gates 中第 indx 个
            
            # 根据even的值选择演化门和位置列表
            single_step_positions = even_positions if evolve_even_gate else odd_positions
            positions += single_step_positions[::direction]

        return positions, gates

    def _get_local_hamiltonians(self, L:int, increment:int, layer:str, pauli:bool=True) -> tuple[list[int], list[_np.ndarray]]:
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
            local_hamiltonian = self.local(site_position, L, pauli=pauli)

            # 如果 local_hamiltonian == 0，表示没有算符作用在 site_position 和 site_position+1 这两个格点上，跳过
            if _np.all(local_hamiltonian == 0):
                continue
            site_positions.append(site_position)
            local_hamiltonians.append(local_hamiltonian)
        return site_positions, local_hamiltonians

    def local(self, site_position:int, L:int, pauli:bool=True) -> _np.ndarray:
        """根据 Oper 的实例得到作用在 position 和 position+1 这两个格点上的局域哈密顿量"""
        assert site_position < L-1, "site_position should be less than L-1"
        from .matrix import pauli_matrix
        c_one = 2. if pauli else 1.
        c_two = 4. if pauli else 1.
        
        local_hamiltonian = _np.zeros((4,4), dtype=_np.float64)  # 用来储存所有作用到 position 和 position+1 这两个格点上的局域哈密顿量的和

        # todo 如果有需要，这里是可以并行加速的（利用 linalg 中的 parallel_reduce 工具），提高 5 ~ 8 倍的速度
        # 遍历哈密顿量中的每一项，找到所有作用在 position 和 position+1 这两个格点上的局域哈密顿量
        for oper_operator, oper_position, oper_coefficient in self.each_term():
            position_length = len(oper_position)  # 判断局域哈密顿量是单体项还是两体项

            if position_length == 1:
                if oper_position[0] == site_position == 0:
                    local_hamiltonian = local_hamiltonian + oper_coefficient * pauli_matrix(oper_operator + 'i') * c_one
                    
                elif oper_position[0] == site_position+1 == L-1:
                    local_hamiltonian = local_hamiltonian + oper_coefficient * pauli_matrix('i' + oper_operator) * c_one
                    
                elif oper_position[0] == site_position:
                    local_hamiltonian = local_hamiltonian + (oper_coefficient/2) * pauli_matrix(oper_operator + 'i') * c_one
                    
                elif oper_position[0]-1 == site_position:
                    local_hamiltonian = local_hamiltonian + (oper_coefficient/2) * pauli_matrix('i' + oper_operator) * c_one
                else:
                    pass  # 其他的情况不是当前的位置（position 和 position + 1），不做处理

            elif position_length == 2:
                
                # oper 中不一定有 this_gate_position_1 < this_gate_position_2 所以需要判断
                if oper_position[0]+1 == oper_position[1] and oper_position[0] == site_position:
                    local_hamiltonian = local_hamiltonian + oper_coefficient * pauli_matrix(oper_operator) * c_two

                elif oper_position[1]+1 == oper_position[0] and oper_position[1] == site_position:
                    # 不同格点上的自旋算符是可以交换的所以之间用 [::-1] 交换顺序
                    local_hamiltonian = local_hamiltonian + oper_coefficient * pauli_matrix(oper_operator[::-1]) * c_two
                else:
                    pass  # 其他的情况不是需要的位置（position_wanted 和 position_wanted + 1），不做处理
                    
                if _np.abs(oper_position[1] - oper_position[0]) != 1:
                    raise NotImplementedError("非最最近邻模型不适合用局域门算法求解")
            else:
                raise NotImplementedError("超两体相互作用模型不适合用局域门算法求解")
        return local_hamiltonian

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
    
    def to_mpo(self, L, pauli=False, backend='torch', device=None):
        if backend == 'torch':
            from ..torch_utils.tensor_network.tnclass import MPO
            from ..torch_utils.utils import convert_to_torch 
            import torch as tc
            tt = self.automata(L, pauli=pauli)
            dtype = tc.float64
            for i in tt:
                if _np.iscomplexobj(i):
                    dtype = tc.complex128
            return MPO(convert_to_torch(tt, dtype=dtype, device=device))
        elif backend == 'tensor':
            # todo 
            raise NotImplementedError("还没有实现")
        else:
            raise ValueError("backend should be 'torch' or 'tensor'")


def I(i:int=0) -> Oper:
    return Oper({'I': {(i,): 1.}}, 's')

def p(i:int=0) -> Oper:
    return Oper({'p': {(i,): 1.}}, 's')

def m(i:int=0) -> Oper:
    return Oper({'m': {(i,): 1.}}, 's')

def x(i:int=0) -> Oper:
    return Oper({'x': {(i,): 1.}}, 's')

def y(i:int=0) -> Oper:
    return Oper({'y': {(i,): 1.}}, 's')

def z(i:int=0) -> Oper:
    return Oper({'z': {(i,): 1.}}, 's')

def n(i:int=0) -> Oper:
    return Oper({'pm': {(i,i): 1.}}, 's')

def nn(i:int, j:int) -> Oper:
    return Oper({'pmpm': {(i,i,j,j): 1.}}, 's')

def zz(i:int, j:int) -> Oper:
    return Oper({'zz': {(i,j): 1.}}, 's')

def mp(i:int, j:int) -> Oper:
    return Oper({'mp': {(i,j): 1.}}, 's')

def pm(i:int, j:int) -> Oper:
    return Oper({'pm': {(i,j): 1.}}, 's')

def xx(i:int, j:int) -> Oper:
    return Oper({'xx': {(i,j): 1.}}, 's')

def yy(i:int, j:int) -> Oper:
    return Oper({'yy': {(i,j): 1.}}, 's')

def xy(i:int, j:int) -> Oper:
    return Oper({'xy': {(i,j): 1.}}, 's')

def yx(i:int, j:int) -> Oper:
    return Oper({'yx': {(i,j): 1.}}, 's')

def sum(oper: Iterable[Oper]) -> Oper:
    res = Oper({}, 's')
    for o in oper:
        res += o
    return res

def heisenberg_operator(L, j=1.0, h=0.0, cyclic=False) -> Oper:
    r"""
    生成 heisenberg 模型的哈密顿量，返回一个 'Oper' 的实例
    
    这个实例可以 automata, local_matrix, to_matrix 等方法

    .. math::
        \sum_{i=1}^{N-1} j * (s^x_i s^x_{i+1} + s^y_i s^y_{i+1} + s^z_i s^z_{i+1}) + \sum_i^N h * s^z_i
    
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
    Li = 0 if cyclic else 1
    data = {}
    if jx != 0:
        data["xx"] = {(i, (i + 1) % L): jx for i in range(L - Li)}
    if jy != 0:
        data["yy"] = {(i, (i + 1) % L): jy for i in range(L - Li)}
    if jz != 0:
        data["zz"] = {(i, (i + 1) % L): jz for i in range(L - Li)}
    if hx != 0:
        data["x"] = {(i,): hx for i in range(L)}
    if hy != 0:
        data["y"] = {(i,): hy for i in range(L)}
    if hz != 0:
        data["z"] = {(i,): hz for i in range(L)}
    return Oper(data, "s")
