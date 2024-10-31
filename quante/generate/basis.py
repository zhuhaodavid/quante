# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 16:51:39
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 18:51:48

"""
生成有对称性的基矢(`SpinBasis`类）：
- `spin_basis`
- `show_spin_basis`
"""

import numpy as _np
from typing import Union, Optional

# !! 这个文件提供到 symmetry 中给个基矢的接口，涉及的 basis 类都在 symmetry 中

__all__ = [
    "spin_basis",
    "show_spin_basis"
]


def _check_spin_number(value:Union[str, float, int]) -> Union[float, int]:
    """检查 value 是不是整数或者半整数"""
    try:
        from fractions import Fraction
        fraction = Fraction(value)  # 尝试将字符串转换为分数
        float_value = float(fraction)  # 转换为浮点数
        if float_value.is_integer():  # 检查是否是整数
            return int(float_value)
        elif (float_value * 2).is_integer():  # 检查是否是半整数
            return float_value
        else:
            raise ValueError("输入的值既不是整数也不是半整数")
    except ValueError:
        raise ValueError("输入的值格式不正确")

from .symmetry.basis_class import SpinBasis

def spin_basis(L:int, S:Union[str, int, float]=1/2, Nup: Optional[int] = None, kblock: Optional[int] = None, pblock: Optional[int] = None, zblock: Optional[int] = None, pzblock: Optional[int] = None,jmblock: Optional[Union[int, tuple[int,int]]] = None) -> SpinBasis:
    """
    参数
    -----------
    L: int
        链的长度/站点数量，需要小于 63。

    Nup: int, 可选
        总磁化强度，:math:`\\sum_j S^z_j`，投影。
        
    S: {str, int}, 可选
        局部自旋自由度的大小。可以是任何（半）整数：
        "1/2","1","3/2",...
        
    kblock: int, 可选
        指定动量块。可以从 0 到 L-1 的整数。
        
    pblock: int, 可选，可取 [-1, 1]
        指定奇偶块。此对称性变换的物理表现是关于链的中点反射。
        
    zblock: int, 可选，可取 [-1, 1]
        指定自旋反转对称性块。此对称性变换的物理表现是翻转自旋-z 分量的符号。
        
    pzblock: int, 可选，可取 [-1, 1]
        指定自旋反转和奇偶块。此对称性变换的物理表现是翻转自旋-z 分量的符号和关于链的中点反射。
        
    jmblock: {int, tuple[int, int]}, 可选
        指定 SU(2) 对称性块。第一个参数是 J，第二个参数是 m。J 取值从 L/2 到 1/2，m 取值从 J 到 -J。
        如果只传入一个参数，则默认 m = J。
        
    示例:
    --------
    >>> basis = qt.generate.basis.spin_basis(L=10, Nup=5, kblock=1)
    >>> print(basis.Ns)
    
    注：
    ---------
    目前支持的组合有：
    - spin-1/2
    
        - noblock

        - Nup

        - kblock

        - pblock

        - zblock

        - pzblock

        - Nup + kblock

        - Nup + pblock

        - Nup + zblock: 这个组合下 Nup 只能从 0 取到 N//2 （基矢为 Nup 与 N - Nup 的叠加态）

        - Nup + pzblock: 同上

        - Nup + kblock + pblock: 这个组合下 kblock 只能从 0 到 L//2,(基矢为 k 和 -k 的叠加)；与 quspin 中的代表元取法不同，因此得到矩阵相差一个排列，但本征值是相同的！

        - Nup + kblock + pblock + zblock: 同上，并且此时 Nup 只能取 N//2，且 N 为偶数；与 quspin 中的代表元取法不同，因此得到矩阵相差一个排列，但本征值是相同的！

        - jmblock

    - spin-high
        - noblock
        
        - Nup
    
    SpinBasis
    ----------
    返回结果的代码提示总是 `SpinBasis` 类型，如果需要查看具体代码，可以通过 `where_codes` 方法定位代码位置。
    其功能包括：
    - 获得基矢：`get_state`，返回基矢在全空间中的表示
    
    - 恢复态：`recover`，将子空间态恢复为全空间态
    
    - 投影算符：`projection_matrix`，返回投影矩阵
    
    - 态投影：`project`，将态投影到子空间
    
    - 打印维度：`print_dims`，打印不同子空间的维数
    
    新增对称性类需要做的事情：
    ------------------------------
    如果需要增加新的对称性，或者新的组合，需要：
    
    - 在 ./symmetry 中增加新的对称性类文件夹，文件夹中需要包含 `general_hamiltonian`, `xxx_basis.py` 和 `xxx_basis_class.py` 三个文件，如果有可以简化生成基的模型，也可以增加单独的文件，如 `heisenberg.py`。
    
    - 在 `xxx_class.py` 中实现对称性的方法（需要的 numba 函数在 `xxx_basis.py` 中）：
    
        - 生成矩阵：`_matrix`
        
        - 获得基矢：`get_state`
        
        - 投影算符：`projection_matrix`
        
        - 打印维度：`print_dims`
        
        - 态投影：`project`
        
        - 态恢复：`recover`
    
    - 在 `spin_basis` 函数中的参数增加对称性参数，如 `kblock`
    
    - 修改：`block_name_list`, `block_value_list`, `block_combinations`，增加对应的参数名和参数值
    
    - 增加对应的处理函数，如 `_process_kblock`
    
    - 记得同步这里的注释和 .matrix.heisenberg 中的代码
    """
    S = _check_spin_number(S)
    
    block_name_list = ["Nup", "kblock", "pblock", "zblock", "pzblock", "jmblock"]
    block_value_list = [Nup, kblock, pblock, zblock, pzblock, jmblock]
    # 将块的状态转换为元组
    blocks_tuple = tuple(block is not None for block in block_value_list)
    
    
    if S == 0.5 and L < 63:
        
        # 定义处理函数的映射
        block_combinations = {
            (False, False, False, False, False, False): _process_spin_half_full_basis,
            ( True, False, False, False, False, False): _process_spin_half_Nup_block,
            (False,  True, False, False, False, False): _process_spin_half_kblock,
            (False, False,  True, False, False, False): _process_spin_half_pblock,
            (False, False, False,  True, False, False): _process_spin_half_zblock,
            ( True,  True, False, False, False, False): _process_spin_half_Nup_kblock,
            ( True, False,  True, False, False, False): _process_spin_half_Nup_pblock,
            ( True, False, False,  True, False, False): _process_spin_half_Nup_zblock,
            ( True, False, False, False,  True, False): _process_spin_half_Nup_pzblock,
            ( True,  True,  True, False, False, False): _process_spin_half_Nup_kblock_pblock,
            ( True,  True,  True,  True, False, False): _process_spin_half_Nup_kblock_pblock_zblock,
            (False, False, False, False,  True, False): _process_spin_half_pzblock,
            (False, False, False, False, False,  True): _process_spin_half_jmblock,
        }
        
        
        # 查找对应的处理函数
        _process_func = block_combinations.get(blocks_tuple, None)

        if _process_func is not None:
            return _process_func(L, {name: value for name, value in zip(block_name_list, block_value_list)})
        else:
            wanted_blocks = [item for item, include in zip(block_name_list, blocks_tuple) if include is not None]
            raise NotImplementedError(f"The combination of blocks: {wanted_blocks} is not supported yet for spin-1/2")
    else:
        
        # 定义处理函数的映射
        block_combinations = {
            (False, False, False, False, False, False): _process_spin_high_full_basis,
            ( True, False, False, False, False, False): _process_spin_high_Nup_block,
        }
        
        
        # 查找对应的处理函数
        _process_func = block_combinations.get(blocks_tuple, None)

        if _process_func is not None:
            return _process_func(L, S, {name: value for name, value in zip(block_name_list, block_value_list)})
        else:
            wanted_blocks = [item for item, include in zip(block_name_list, blocks_tuple) if include is not None]
            raise NotImplementedError(f"The combination of blocks: {wanted_blocks} is not supported yet for spin-high")
        

# 处理不同block的函数部分
def _process_spin_half_full_basis(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.noblock.defclass import SpinHalfBasisNoBlock
    return SpinHalfBasisNoBlock(L)

def _process_spin_half_Nup_block(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.Nup.defclass import SpinHalfBasisNup
    return SpinHalfBasisNup(L, block_dic['Nup'])

def _process_spin_half_kblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.kblock.defclass import SpinHalfBasisKBlock
    return SpinHalfBasisKBlock(L, block_dic['kblock'])
    
def _process_spin_half_pblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.pblock.defclass import SpinHalfBasisPBlock
    return SpinHalfBasisPBlock(L, block_dic['pblock'])

def _process_spin_half_zblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.zblock.defclass import SpinHalfBasisZBlock
    return SpinHalfBasisZBlock(L, block_dic['zblock'])

def _process_spin_half_pzblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.pzblock.defclass import SpinHalfBasisPZBlock
    return SpinHalfBasisPZBlock(L, block_dic['pzblock'])

def _process_spin_half_Nup_kblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.Nup_kblock.defclass import SpinHalfBasisNupKBlock
    return SpinHalfBasisNupKBlock(L, block_dic['Nup'], block_dic['kblock'])

def _process_spin_half_Nup_pblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.Nup_pblock.defclass import SpinHalfBasisNupPBlock
    return SpinHalfBasisNupPBlock(L, block_dic['Nup'], block_dic['pblock'])

def _process_spin_half_Nup_zblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.Nup_zblock.defclass import SpinHalfBasisNupZBlock
    return SpinHalfBasisNupZBlock(L, block_dic['Nup'], block_dic['zblock'])

def _process_spin_half_Nup_pzblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.Nup_pzblock.defclass import SpinHalfBasisNupPZBlock
    return SpinHalfBasisNupPZBlock(L, block_dic['Nup'], block_dic['pzblock'])

def _process_spin_half_Nup_kblock_pblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.Nup_kblock_pblock.defclass import SpinHalfBasisNupKPBlock
    return SpinHalfBasisNupKPBlock(L, block_dic['Nup'], block_dic['kblock'], block_dic['pblock'])

def _process_spin_half_Nup_kblock_pblock_zblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.Nup_kblock_pblock_zblock.defclass import SpinHalfBasisNupKPZBlock
    return SpinHalfBasisNupKPZBlock(L, block_dic['Nup'], block_dic['kblock'], block_dic['pblock'], block_dic['zblock'])

def _process_spin_half_jmblock(L:int, block_dic:dict) -> SpinBasis:
    from .symmetry.spin_half.su2.defclass import SpinHalfBasisSU2
    jm = block_dic['jmblock']
    if isinstance(jm, tuple):
        j, m = jm
    else:
        j = m = jm
    return SpinHalfBasisSU2(L, j, m)

def _process_spin_high_full_basis(L:int, S:Union[int, float], block_dic:dict) -> SpinBasis:
    from .symmetry.spin_high.noblock.defclass import SpinHighBasisNoBlock
    return SpinHighBasisNoBlock(L, S)

def _process_spin_high_Nup_block(L:int, S:Union[int, float], block_dic:dict) -> SpinBasis:
    from .symmetry.spin_high.Nup.defclass import SpinHighBasisNup
    return SpinHighBasisNup(L, S, block_dic['Nup'])

def show_spin_basis(vector:_np.ndarray)->tuple[list[_np.ndarray], list[str]]:
    """向量转换为spin-1/2直积态求和形式 
    
    0 -> ↑ = (1, 0), 1 -> ↓ = (0, 1)
    
    [a, b, c, d] = a|00> + .. + d|11>

    Args: quantum state
    Returns: coefficients, basiss
    """
    size = vector.size
    assert (size & (size - 1))==0, "Only can calculate spin-1/2 state: (2^N,)"
    element_index = _np.nonzero(vector)[0]  # elemenet is the non-zero element
    coefficients = [vector[i] for i in element_index]
    basiss = [_np.binary_repr(i, int(_np.log2(size))) for i in element_index]
    for basis, coef in zip(basiss, coefficients):
        if _np.abs(coef) > 1.e-12:
            b = basis.replace('0', '↑').replace('1', '↓') + ":"
            print(b, coef)

