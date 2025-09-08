# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 19:13:08
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-08 16:21:56

import numpy as np
import scipy.sparse as sp
from .general import Oper, _single_term, _merge_poscoef 

def _sort_pm(data:list, sign=-1.):
    r"""
    对给定的算符字符串、位置数组和系数进行排序和化简。

    该函数用于对费米算符进行排序，处理相邻的 `-+` 和 `+-` 操作符对，并根据反对易关系调整符号或化简算符。

    参数:
    --------
    oper : list[tuple[str, numpy.ndarray, float | complex]]
        包含算符字符串、位置数组和系数的列表，例如 `[('-+', np.array([0, 1]), 1.0)]`。
    posn : numpy.ndarray
        表示算符作用位置的数组，例如 `[0, 1]`。
    coef : float 或 complex
        表示算符的系数。

    返回:
    --------
    list
        返回一个包含排序和化简后的算符、位置数组和系数的列表。

    处理逻辑:
    --------
    1. 检查相邻的 `-+` 和 `+-` 操作符对。
    2. 如果相邻操作符的作用位置相同，则根据反对易关系调整符号或化简算符。
    3. 如果相邻操作符的作用位置不同，则交换它们的位置并调整符号。
    4. 返回排序和化简后的算符、位置数组和系数的列表。
    """
    maxiter = 100000
    iternum = 0
    tobeordered = [i for i in data]
    final_res = []
    while iternum < maxiter and len(tobeordered) != 0:
        oper, posn, coef = tobeordered.pop(0)
        for i in range(1, len(oper)):
            if oper[i-1:i+1] == '-+':
                if posn[i-1] == posn[i]:
                    if len(posn) == 2:
                        tobeordered.append(('I', [0], coef))
                    else:
                        tobeordered.append(
                            (
                                oper[:i-1] + oper[i+1:],
                                np.delete(posn, [i-1, i]),
                                coef
                            )
                        )
                tobeordered.append(
                    (
                        oper[:i-1] + '+-' + oper[i+1:],
                        list(posn[:i-1]) + [posn[i], posn[i-1]] + list(posn[i+1:]),
                        coef * sign
                    )
                )
                break
            elif (oper[i-1:i+1] == '--' or oper[i-1:i+1] == '++') and posn[i-1] == posn[i]:
                iternum += 1
                break
        else:
            final_res.append((oper, posn, coef))
        iternum += 1
    
        if iternum == maxiter:
            raise RuntimeError("Exceeded maximum iterations in _sort_pm. Possible infinite loop.")

    return final_res


def _sort_posn(ops, arr):
    r"""
    根据字符串和数组计算排序后的奇偶性。

    参数:
    ops (str): 包含操作符的字符串，如 "++++----"。
    arr (list): 包含整数的数组，如 [2,1,3,4,6,3,5,7]。

    返回:
    tuple: 排序后的数组和总的奇偶性 (1 或 -1)。
    """
    # 将数组按照字符串中的符号分类
    groups = {}
    for op, val in zip(ops, arr):
        if op not in groups:
            groups[op] = []
        groups[op].append(val)
    
    # 对每组进行排序并计算奇偶性
    sorted_groups = []
    parity = 0
    for key in sorted(groups.keys()):
        group = groups[key]
        sorted_group = sorted(group)
        sorted_groups.extend(sorted_group)
        
        # 计算排列的奇偶性
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i] > group[j]:
                    parity += 1
    
    # 计算总的奇偶性
    total_parity = (-1) ** parity
    return sorted_groups, total_parity


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
 
    def _has_expanded(self) -> bool:
        for opnm in self.data.keys():
            for o in opnm:
                if o not in ['I', "+", "-"]:
                    return False
        return True
    
    def normal_ordering(self):
        """通过对易关系将算符转换为正则序列。

        .. math::
            -+ -> i - +-

        Returns
        -------
        FermionOper
            具有正则序列的 FermionOper 对象。
        """
        data = {}
        if not self._has_expanded():
            expandn = list(self.expandn().each_term())
        else:
            expandn = list(self.each_term())
        for new_opnm, new_posn, new_coef in _sort_pm(expandn):
            new_posn, parity = _sort_posn(new_opnm, new_posn)
            posnlist, coeflist = data.setdefault(new_opnm, [[], []])
            posnlist.append(new_posn)
            coeflist.append(parity * new_coef)
        newdata = {}
        for name, (posnlist, coeflist) in data.items():
            newposn, newcoef = _merge_poscoef(posnlist, coeflist)
            if len(newposn) > 0:
                newdata[name] = (newposn, newcoef)
        return FermionOper(newdata)

    def single_particle_ham(self, L=None):
        r"""生成最简单的单粒子矩阵
        
        .. math::

            H = \sum_{i,j} h_{ij} c_i^\dagger c_j
        
        """
        if L is None:
            L = self.L
        h = np.zeros((L, L), dtype=self.dtype())
        coef_I = 0.
        for name, (posnlist, coeflist) in self.data.items():
            if name == '+-':
                for i, posn in enumerate(posnlist):
                    h[*posn] = coeflist[i]
            elif name == 'I':
                coef_I += coeflist[0]
            else:
                raise ValueError(f"not free fermion due to: {name}")
        return h, coef_I
    
    def BdG_ham(self, L=None):
        r"""
        Nambu spin form
        .. math:
            \Psi_i = \begin{bmatrix} c_i \\ c_i^\dagger \end{bmatrix}
        
        BdG hamiltonian:
        .. math:
            H = \frac{1}{2} \sum_{ij} \Psi_i^\dagger \mathcal{H}_{ij} \Psi_j + \text{Const.},
            = \frac{1}{2} \sum_{ij}^{} A_{ij} c_{i}^{\dagger} c_{j} + \frac{1}{2} \sum_{ij}^{} B_{ij} c_{i}^{\dagger} c_{j}^{\dagger} - \frac{1}{2} \sum_{ij}^{} B_{ij}^* c_{i} c_{j} - \frac{1}{2} \sum_{ij}^{} A_{ij}^* c_{i} c_{j}^{\dagger} + \text{Const.}
        
        where,
        .. math:
            \mathcal{H}_{ij} =  \begin{bmatrix}
                                    A_{ij} & B_{ij} \\
                                    -B_{ij}^* & -A_{ij}^*
                                \end{bmatrix},
        
        """
        if L is None:
            L = self.L
        H11 = np.zeros((L, L), dtype=self.dtype())
        H12 = np.zeros((L, L), dtype=self.dtype())
        H21 = np.zeros((L, L), dtype=self.dtype())
        # H22 = np.zeros((L, L), dtype=self.dtype())
        coef_I = 0.
        for name, (posnlist, coeflist) in self.data.items():
            if name == '+-':
                for i, posn in enumerate(posnlist):
                    H11[*posn] = coeflist[i]
                    if posn[0] == posn[1]:
                        coef_I += coeflist[i]/2
            elif name == '--':
                for i, posn in enumerate(posnlist):
                    H21[*posn] = coeflist[i]
            elif name == '++':
                for i, posn in enumerate(posnlist):
                    H12[*posn] = coeflist[i]
            elif name == 'I':
                coef_I += coeflist[0]
            elif name == '-+':
                raise ValueError("Normal ordering not done")
            else:
                raise ValueError(f"not BdG fermion due to: {name}")
        H12 = (H12 - H12.T)
        H21 = (H21 - H21.T)
        BdG = np.block([
            [H11, H12], 
            [H21, -H11.T]
        ])
        return BdG, coef_I


    def dtype(self):
        for _, (_, coef) in self.data.items():
            if np.iscomplexobj(coef):
                return complex
        return float


    def to_quspin(self):
        """
        返回 quspin 可以接受的格式
        """
        static = []
        for opnm, (posn, coef) in self.data.items():
            static_bond = []
            for i in range(len(coef)):
                static_bond.append([(coef[i]).item()] + [a.item() for a in posn[i]])
            static.append([opnm, static_bond])
        return static


    def to_matrix(self, basis, dtype=None, sparse=False):
        if self.data == {}:
            if not sparse:
                return np.zeros((basis.Ns, basis.Ns), dtype=float)
            return sp.csr_matrix((basis.Ns, basis.Ns), dtype=float)
        self._check_length(basis.L) 
        from ..basis.basis_class import FermionBasis
        if isinstance(basis, FermionBasis):
            eachterm, hascomplex = self._convert_to_quick_form()
            mat = basis._sparse_matrix(eachterm, hascomplex, savememory=False)
            if sparse:
                return mat
            else:
                return mat.toarray()
        else:
            from ...bridge.quspin_utils import hamiltonian
            return hamiltonian(self, basis, dtype=np.complex128 if dtype is None else dtype, sparse=sparse)


    def _convert_to_quick_form(self):
        """这个函数专门为 to_matrix 写的，其他函数不需要"""
        eachterm = []
        hascomplex = False
        for opnm, posn, coef in self.each_term():
            opnm_list = []
            for i in opnm:
                if i == '-':
                    opnm_list.append(0)
                elif i == '+':
                    opnm_list.append(1)
                elif i == 'I':
                    opnm_list.append(2)
                elif i == 'n':
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


    def jw_transfer(self):
        # !! todo: Z -> -Z
        from .spin import SpinBuilder
        builder = SpinBuilder()
        for opnm_, (poslist, coefflist) in self.data.items():
            for (pos_, coeff) in zip(poslist, coefflist):
                posn = list(pos_)
                opnm = opnm_
                newopnm = ''
                newposn = []
                newcoef = coeff
                
                next_pos = len(opnm)

                while True:
                    # 寻找下一个 +- 的位置
                    cur_pos = -1
                    for i in range(next_pos-1, -1, -1):
                        if opnm[i] in ['+', '-']:
                            cur_pos = i
                            break
                    if cur_pos == -1:
                        # 偶数个，直接退出
                        newopnm = opnm[:next_pos] + newopnm
                        newposn = posn[:next_pos] + newposn
                        break
                    
                    # 寻找下一个 +- 的位置，两个两个的处理可以避免出现太多的 Z
                    next_pos = -1
                    for i in range(cur_pos-1, -1, -1):
                        if opnm[i] in ['+', '-']:
                            next_pos = i
                            break
                    if next_pos == -1:
                        # 奇数个，补Z
                        newopnm = opnm[:cur_pos] + 'Z'*posn[cur_pos] + opnm[cur_pos] + newopnm
                        newposn = posn[:cur_pos] + list(range(posn[cur_pos]+1)) + newposn
                        break
                        
                    startindx = min(posn[next_pos:cur_pos+1])
                    
                    # 相邻的 +- 中间没有其他算符
                    if next_pos + 1 == cur_pos:
                        # 相同算符的简化
                        if posn[next_pos] == posn[cur_pos]:
                            newopnm = opnm[next_pos:cur_pos+1] + newopnm
                            newposn = posn[next_pos:cur_pos+1] + newposn
                        # 正序的
                        elif posn[next_pos] == startindx:
                            newopnm = opnm[next_pos:cur_pos] + 'Z'*(posn[cur_pos] - startindx - 1) + opnm[cur_pos] + newopnm
                            newposn = posn[next_pos:cur_pos] + list(range(startindx+1, posn[cur_pos])) + [posn[cur_pos]] + newposn
                            newcoef *= (-1 if opnm[next_pos] == '+' else 1)
                        # 倒序的
                        else:
                            newopnm = 'Z'*(posn[next_pos] - startindx - 1) + opnm[next_pos:cur_pos] + opnm[cur_pos] + newopnm
                            newposn = list(range(startindx+1, posn[next_pos])) + posn[next_pos:cur_pos] + [posn[cur_pos]] + newposn
                            newcoef *= (-1 if opnm[cur_pos] == '-' else 1)
                    else:
                        # 最一般的情况
                        newopnm = 'Z'*(posn[next_pos] - startindx) + opnm[next_pos:cur_pos] + 'Z'*(posn[cur_pos] - startindx) + opnm[cur_pos] + newopnm
                        newposn = list(range(startindx, posn[next_pos])) + posn[next_pos:cur_pos] + list(range(startindx, posn[cur_pos])) + [posn[cur_pos]] + newposn
                    
                newposn = np.array(newposn)
                builder += opnm, newposn, newcoef
        return builder.build()
    
    def automata(self, L=None):
        spinoper = self.jw_transfer()
        return spinoper.automata(L=L)
    
    def to_mpo(self, *args, **kwargs):
        spinoper = self.jw_transfer()
        return spinoper.to_mpo(*args, **kwargs)
    
    @classmethod
    def _convert_from_spin(cls, opnm, i, coef):
        if opnm == 'p':
            return FermionOper({'+': _single_term((i,), coef)})
        elif opnm == 'm':
            return FermionOper({'-': _single_term((i,), coef)})
        elif opnm == 'I':
            return FermionOper({'I': _single_term((0,), coef)})
        elif opnm == 'Z':
            return FermionOper({'+-': _single_term((i,i,), 2.*coef), 'I': _single_term((0,), -coef)})
        else:
            raise ValueError(f"Invalid operator name {opnm}, should be in ['p', 'm', 'I', 'Z']")

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
    
    @classmethod
    def builder(cls) -> 'FermionBuilder':
        return FermionBuilder()
    

class FermionBuilder:
    def __init__(self):
        self.terms = {}

    def __iadd__(self, term) -> 'FermionBuilder':
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
   
    def build(self):
        data = {}
        for name, (posnlist, coeflist) in self.terms.items():
            data[name] = _merge_poscoef(posnlist, coeflist)
        return FermionOper(data, type='f')

def builder() -> FermionBuilder:
    """返回 FermionBuilder 对象"""
    return FermionBuilder()

def sum(oper) -> FermionOper:
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
        newpos, newcoef = np.vstack(posnlist), np.hstack(coeflist)
        if len(newpos) > 0:
            newdata[name] = (newpos, newcoef)
    if stype is None:
        stype = 'f'
    return FermionOper(newdata)

def _make_oper(name: str, posn: tuple[int], coef: float, L:None|int) -> "SpinOper":
    """Helper function to create a SpinOper with a single term."""
    if L is not None:
        posn = [i % L for i in posn]  # Ensure positions are within bounds
    return FermionOper({name: _single_term(posn, coef)})

def f(i:int=0, L=None) -> FermionOper:
    return _make_oper('-', (i,), 1., L)

def fdag(i:int=0, L=None) -> FermionOper:
    return _make_oper('+', (i,), 1., L)

def p(i:int=0, L=None) -> FermionOper:
    return _make_oper('+', (i,), 1., L)

def m(i:int=0, L=None) -> FermionOper:
    return _make_oper('-', (i,), 1., L)

def pm(i:int=0, j:int=0, L=None) -> FermionOper:
    return _make_oper('+-', (i,j), 1., L)

def mp(i:int=0, j:int=0, L=None) -> FermionOper:
    return _make_oper('-+', (i,j), 1., L)


def syk4_dirac(L, Jmat=None):
    if Jmat is None:
        from ..matrix import _syk4_dirac_Jmat
        Jmat = _syk4_dirac_Jmat(L, J=1.0)
    bd = builder()
    for i1 in range(L):
        for i2 in range(L):
            for j1 in range(L):
                for j2 in range(L):
                    bd += "++--", [i1, i2, j1, j2], Jmat[i1 * L + i2, j1 * L + j2]
    return bd.build()

