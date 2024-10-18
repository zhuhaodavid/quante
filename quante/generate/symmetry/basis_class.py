# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-05 09:31:36
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-14 23:22:13

from typing import Union
import inspect
import numpy as np
import scipy.sparse as _sp
from ...basicfun import PrintLn

@PrintLn.add_object_print
class SpinBasis:
    """
    自旋基矢，目前包含：
        具体的方法参见子类
    
    SpinBasis\n
    |__ SpinHalfBasis
        |__ SpinHalfBasisNoBlock: 
        |           [noblock](file://./spin_half/noblock/defclass.py)
        |
        |__ SpinHalfBasisNup: 
        |           [Nup](file://./spin_half/Nup/defclass.py)
        |
        |__ SpinHalfBasisKBlock: 
        |           [kblock](file://./spin_half/kblock/defclass.py)
        |
        |__ SpinHalfBasisPBlock: 
        |           [pblock](file://./spin_half/pblock/defclass.py)
        |
        |__ SpinHalfBasisZBlock: 
        |           [zblock](file://./spin_half/zblock/defclass.py)
        |
        |__ SpinHalfBasisNupKBlock: 
        |           [Nup_kblock](file://./spin_half/Nup_kblock/defclass.py)
        |
        |__ SpinHalfBasisNupPBlock: 
        |           [Nup_pblock](file://./spin_half/Nup_pblock/defclass.py)
        |
        |__ SpinHalfBasisNupZBlock: 
        |           [Nup_zblock](file://./spin_half/Nup_zblock/defclass.py) 
        |
        |__ SpinHalfBasisNupKPBlock: 
        |           [Nup_kpblock](file://./spin_half/Nup_kblock_pblock/defclass.py)
        |
        |__ SpinHalfBasisNupKPZBlock:
        |           [Nup_kpzblock](file://./spin_half/Nup_kblock_pblock_zblock/defclass.py)
        |
        |__ SpinHalfBasisSU2: 
                    [SU2](file://./spin_half/su2/defclass.py)
    
    |__ SpinHighBasis
    """
    def __init__(self, L:int, S:Union[int, float]=0.5) -> None:
        self.L: int = L
        self.S: float = float(S)
        self.local_dim = int(S * 2 + 1)
        self.Ns: int
        self.other_params: dict = {}
        self.s_list: Union[list, range]
        self.default_complex: bool = True
        self._double_Ns = 1

    def where_codes(self, method_name: str = None):
        """
        返回某个实例所属，以及其方法，所在文件的路径和行号。
        
        Example:
        >>> basis = qt.generate.basis.spin_basis(L=10, zblock=1)
        >>> basis.where_codes()
        >>> basis.where_codes("_matrix")
        """
        # 获取当前实例的类
        cls = self.__class__
        # 获取类的定义文件路径
        file_path = inspect.getfile(cls)
        
        if method_name:
            # 获取方法对象
            method = getattr(cls, method_name, None)
            if method:
                # 获取方法的行号
                line_number = inspect.getsourcelines(method)[1]
                print(f"{cls.__name__}.{method_name}:\n   \"{file_path}\", line {line_number}")
            else:
                raise ValueError(f"Method {method_name} not found in class {cls.__name__}")
        else:
            print(f'{cls.__name__}:\n   "'+file_path+'"')
    
    def _sparse_matrix(self, op_list, hascomplex, savememory=False):
        """ takes list of operator strings and couplings to create matrix.
        
        # todo nobloc, Nup 之外的基矢，能否利用 _isdiag 函数判断是否是对角矩阵，然后直接生成对角矩阵?
        
        # todo 并行化程度仍然不高，如何更高效的利用 cpu?
        """
        from .basis_class_nb import _is_diagonal, _update_diag, add_, _get_index_type, coodiaglists2csr, coolists2csr2
        off_diag = None
        diag = None
        
        dtype = np.complex128 if hascomplex or self.default_complex else np.float64

        # 预设内存，避免反复分配内存
        real_Ns = self.Ns * self._double_Ns
        index_type = _get_index_type(real_Ns)
        
        row_init = np.empty(real_Ns, dtype=index_type)
        col_init = np.empty(real_Ns, dtype=index_type)
        ele_int = np.empty(real_Ns, dtype=dtype)
        
        if not savememory:
            row_result = []
            col_result = []
            ele_result = []
        
        for opnm, posn, coef in op_list:
            row, col, ele = self._Op(opnm, posn, coef, row_init, col_init, ele_int)  # 主要的时间花费，一半时间花在这里
            if(len(ele)>0):
                if row is None:
                    if diag is None:
                        diag = ele
                    else:
                        add_(diag, ele)
                        # diag += ele
                elif _is_diagonal(row,col):
                    if diag is None:
                        diag = np.zeros(self.Ns,dtype=dtype)
                    _update_diag(diag,row,ele)
                else:
                    if not savememory:
                        ele_result.append(ele)
                        row_result.append(row)
                        col_result.append(col)
                    else:
                        # todo, csr 如何并行相加？
                        tmp = _sp.csr_array((ele,(row,col)),shape=(self.Ns,self.Ns),dtype=dtype) 
                        off_diag = tmp if off_diag is None else off_diag + tmp
        
        if not savememory:
            if len(ele_result) > 0:
                return coodiaglists2csr(row_result, col_result, ele_result, diag, self.Ns, index_type, dtype)
                # return coolists2csr2(row_result, col_result, ele_result, self.Ns, dtype)
        else:
            if off_diag is not None and diag is None:
                return off_diag
            elif off_diag is not None and diag is not None:
                indptr = np.arange(self.Ns+1)
                return off_diag + _sp.csr_array((diag,indptr[:self.Ns],indptr),shape=(self.Ns,self.Ns),dtype=dtype)

        if diag is not None:
            return _sp.dia_array((np.atleast_2d(diag),[0]),shape=(self.Ns,self.Ns),dtype=dtype)
        else:
            return _sp.dia_array((self.Ns,self.Ns),dtype=dtype)
    
    def _isdiag(self, opnm:list[int], pos:list[int]):
        if 0 not in opnm and 1 not in opnm:
            return True
        else:
            zero_positions = np.sort([pos[index] for index, value in enumerate(opnm) if value == 0])
            one_positions = np.sort([pos[index] for index, value in enumerate(opnm) if value == 1])
            return len(zero_positions) == len(one_positions) and np.all(zero_positions == one_positions)

        
    def __getitem__(self, index:int):
        """生成基矢的向量"""
        pass  # 通过 .where_codes() 查看具体实现的位置
        
    def to_full_space(self, index:int) -> np.ndarray:
        """生成基矢的第 i 个向量在全空间中的表示
        
        Example:
        >>> L = 10
        >>> basis = qt.generate.basis.spin_basis(L=10, Nup=2)
        >>> for i in range(basis.Ns):
        >>>     qt.generate.show_spin_basis(basis.to_full_space(i))
        """
        pass  # 通过 .where_codes() 查看具体实现的位置
    
    def recover(self, state:np.ndarray) -> np.ndarray:
        """
        将全空间/Nup空间的态投影到某个子空间 #!! 核心代码，get_state, projection_matrix 都是基于 recover 实现的
        
        
        Example:
        >>> L = 10
        >>> basis = qt.generate.basis.spin_basis(L=10, Nup=2)
        >>> state = np.random.rand(basis.Ns)
        >>> state_in_full_space = basis.recover(state)
        """
        pass  # 通过 .where_codes() 查看具体实现的位置
    
    def projection_matrix(self) -> np.ndarray:
        """
        投影算符，（2^L, Ns) 维的矩阵，将全空间的态投影到某个子空间的矩阵
        
        Example:
        >>> L = 10
        >>> ham = qt.generate.operas.heisenberg_operator(L, cyclic=True)
        >>> basis = qt.generate.basis.spin_basis(L=L, kblock=1)
        >>> mat1 = ham.to_matrix(basis)
        >>> 
        >>> basis = qt.generate.basis.spin_basis(L=L)
        >>> mat1 = ham.to_matrix(basis)
        >>> 
        >>> proj = basis.projection_matrix()
        >>> mat2 = proj.conj().T @ mat1 @ proj
        >>> print(mat2 - mat1)
        """
        pass  # 通过 .where_codes() 查看具体实现的位置

    def project(self, state:np.ndarray) -> np.ndarray:
        """
        将全空间/Nup空间的态投影到某个子空间
        """
        pass  # 通过 .where_codes() 查看具体实现的位置
    
    
    @classmethod
    def print_dims(cls, L:int):
        """
        输出不同子空间的维数
        
        Example:
        >>> basis = qt.generate.basis.spin_basis(L=10)
        >>> basis.print_dims(L=10)
        """
        pass  # 通过 .where_codes() 查看具体实现的位置
    
    # todo: action_on(ham, state) 作用算符到态上，返回新的态，可以节约内存但是不如存下来速度快

class SpinHalfBasis(SpinBasis):
    """
    自旋 1/2，特指用 0 和 1 二进制数表示的自旋基矢来实现的类
    
    其中 0 表示自旋 up，1 表示自旋 down
    """
    def __init__(self, L: int) -> None:
        assert L < 63, "N should be less than 63, otherwise the int type will overflow"
        super().__init__(L, S=0.5)
        self.Ns = 1 << L


class SpinHighBasis(SpinBasis):
    """
    一般的自旋，速度可能会较慢
    
    其中数字表示激发数，例如 0 表示基态，1 表示一个激发，2 表示两个激发
    
    如果 S = 0.5，那么 0 表示自旋 down，1 表示自旋 up
    """
    def __init__(self, L: int, S:Union[float, int]) -> None:
        super().__init__(L, S=S)
        self.Ns = self.local_dim ** L
