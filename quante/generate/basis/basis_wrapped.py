# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 16:51:39
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-13 23:44:07

"""
生成有对称性的基矢(`SpinBasis`类）：
- `spin_basis`
"""

import numpy as np
from typing import Union, Optional
from warnings import warn


# !! 这个文件提供到 symmetry 中给个基矢的接口，涉及的 basis 类都在 symmetry 中

__all__ = [
    "spin_basis",
    "spin_ladder_basis"
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

from .basis_class import SpinBasis, FermionBasis

def spin_basis(L:int, S:Union[str, int, float]=1/2, Nup: Optional[int] = None, kblock: Optional[int] = None, pblock: Optional[int] = None, zblock: Optional[int] = None, pzblock: Optional[int] = None,jmblock: Optional[Union[int, tuple[int,int]]] = None) -> SpinBasis:
    """计算自旋基矢，这个基矢生成速度较快，如果需要未实现的对称性，可以使用 quspin_spin_basis 中的函数。
    
    Parameters
    ----------
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

    Examples
    --------
    >>> basis = qt.generate.basis.spin_basis(L=10, Nup=5, kblock=1)
    >>> print(basis.Ns)
    
    Notes
    -----
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
    
    todos
    -----------------------
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
            (False,  True,  True, False, False, False): _process_spin_half_kpblock,
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
        _process_func = block_combinations.get(blocks_tuple, None) # type: ignore

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
        _process_func = block_combinations.get(blocks_tuple, None) # type: ignore

        if _process_func is not None:
            return _process_func(L, S, {name: value for name, value in zip(block_name_list, block_value_list)})
        else:
            wanted_blocks = [item for item, include in zip(block_name_list, blocks_tuple) if include is not None]
            raise NotImplementedError(f"The combination of blocks: {wanted_blocks} is not supported yet for spin-high")
        

# 处理不同block的函数部分
def _process_spin_half_full_basis(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.noblock.defclass import SpinHalfBasisNoBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisNoBlock
    return SpinHalfBasisNoBlock(L)

def _process_spin_half_Nup_block(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.Nup.defclass import SpinHalfBasisNup
    from .spin_half.spin_1d.basis import SpinHalfBasisNup
    return SpinHalfBasisNup(L, block_dic['Nup'])

def _process_spin_half_kblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.kblock.defclass import SpinHalfBasisKBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisKBlock
    return SpinHalfBasisKBlock(L, block_dic['kblock'])
    
def _process_spin_half_pblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.pblock.defclass import SpinHalfBasisPBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisPBlock
    return SpinHalfBasisPBlock(L, block_dic['pblock'])

def _process_spin_half_zblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.zblock.defclass import SpinHalfBasisZBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisZBlock
    return SpinHalfBasisZBlock(L, block_dic['zblock'])

def _process_spin_half_pzblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.pzblock.defclass import SpinHalfBasisPZBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisPZBlock
    return SpinHalfBasisPZBlock(L, block_dic['pzblock'])

def _process_spin_half_kpblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.kblock_pblock.defclass import SpinHalfBasisKPBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisKPBlock
    return SpinHalfBasisKPBlock(L, block_dic['kblock'], block_dic['pblock'])

def _process_spin_half_Nup_kblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.Nup_kblock.defclass import SpinHalfBasisNupKBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisNupKBlock
    return SpinHalfBasisNupKBlock(L, block_dic['Nup'], block_dic['kblock'])

def _process_spin_half_Nup_pblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.Nup_pblock.defclass import SpinHalfBasisNupPBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisNupPBlock
    return SpinHalfBasisNupPBlock(L, block_dic['Nup'], block_dic['pblock'])

def _process_spin_half_Nup_zblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.Nup_zblock.defclass import SpinHalfBasisNupZBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisNupZBlock
    return SpinHalfBasisNupZBlock(L, block_dic['Nup'], block_dic['zblock'])

def _process_spin_half_Nup_pzblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.Nup_pzblock.defclass import SpinHalfBasisNupPZBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisNupPZBlock
    return SpinHalfBasisNupPZBlock(L, block_dic['Nup'], block_dic['pzblock'])

def _process_spin_half_Nup_kblock_pblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.Nup_kblock_pblock.defclass import SpinHalfBasisNupKPBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisNupKPBlock
    return SpinHalfBasisNupKPBlock(L, block_dic['Nup'], block_dic['kblock'], block_dic['pblock'])

def _process_spin_half_Nup_kblock_pblock_zblock(L:int, block_dic:dict) -> SpinBasis:
    # from .spin_half.Nup_kblock_pblock_zblock.defclass import SpinHalfBasisNupKPZBlock
    from .spin_half.spin_1d.basis import SpinHalfBasisNupKPZBlock
    return SpinHalfBasisNupKPZBlock(L, block_dic['Nup'], block_dic['kblock'], block_dic['pblock'], block_dic['zblock'])

def _process_spin_half_jmblock(L:int, block_dic:dict) -> SpinBasis:
    from .spin_half.su2.defclass import SpinHalfBasisSU2
    jm = block_dic['jmblock']
    if isinstance(jm, tuple):
        j, m = jm
    else:
        j = m = jm
    return SpinHalfBasisSU2(L, j, m)

def _process_spin_high_full_basis(L:int, S:Union[int, float], block_dic:dict) -> SpinBasis:
    from .spin_high.noblock.defclass import SpinHighBasisNoBlock
    return SpinHighBasisNoBlock(L, S)

def _process_spin_high_Nup_block(L:int, S:Union[int, float], block_dic:dict) -> SpinBasis:
    if S == 0.5 and block_dic['Nup'] == 1:
        # For this single excitation, we can accelerate the calculation
        # by using the SpinHalfSingleExcitation basis
        from .spin_half.spin_excitation.defclass import SpinHalfSingleExcitation
        return SpinHalfSingleExcitation(L)
    from .spin_high.Nup.defclass import SpinHighBasisNup
    return SpinHighBasisNup(L, S, block_dic['Nup'])

def fermion_basis(L:int, Nf:int|None=None) -> SpinBasis:
    """Generate fermion basis for fermion systems.

    !!! This function may not be high efficiency, if you need high efficiency, please use quspin_fermion_basis instead.

    Parameters
    ----------
    L : int
        The length of the chain, must be less than 63.
    Nf : int | None, optional
        The total number of fermions, :math:`\\sum_j N_f^j`, projection. If None, the full basis will be generated.

    Returns
    -------
    SpinBasis
        The generated fermion basis.

    Raises
    ------
    NotImplementedError
        If the combination of blocks is not supported yet.
    """
    block_name_list = ["Nf"]
    block_value_list = [Nf]
    # 将块的状态转换为元组
    blocks_tuple = tuple(block is not None for block in block_value_list)

    assert L < 63, "L must be less than 63 for fermion basis generation."
    # 定义处理函数的映射
    block_combinations = {
        (False, ): _process_fermionbit_full_basis,
        ( True, ): _process_fermionbit_Nf_block,
    }
    # 查找对应的处理函数
    _process_func = block_combinations.get(blocks_tuple, None) # type: ignore

    if _process_func is not None:
        return _process_func(L, {name: value for name, value in zip(block_name_list, block_value_list)})
    else:
        wanted_blocks = [item for item, include in zip(block_name_list, blocks_tuple) if include is not None]
        raise NotImplementedError(f"The combination of blocks: {wanted_blocks} is not supported yet for spin-1/2")
    
def _process_fermionbit_full_basis(L:int, block_dic:dict) -> FermionBasis:
    from .fermion.noblock.defclass import FermionBitBasisNoBlock
    return FermionBitBasisNoBlock(L)

def _process_fermionbit_Nf_block(L:int, block_dic:dict) -> FermionBasis:
    from .fermion.Nup.defclass import FermionBitBasisNup
    return FermionBitBasisNup(L, block_dic['Nf'])


def spin_basis_general(
    L, S:Union[str, int, float]="1/2", 
    Nup=None, Ndiff=None, check_commute=True, **blocks
):
    """General spin-1/2 basis constructor with optional Z2 symmetries and Ndiff constraint.

    Parameters
    ----------
    L : int
        Number of sites.
    S : str|int|float, default '1/2'
        Spin quantum number (only 1/2 supported currently).
    Nup : int|None
        Particle number difference (or magnetization style) constraint.
    Ndiff : tuple[array_like, int]|None
        Particle number difference constraint. Must be a tuple ``(flipset, Ndiff)``,
    **blocks : dict[str, tuple[array_like,int]]
        Z2 symmetry specification(s). Each value must be a tuple ``(perm, sector)``.
    """
    assert L < 64, "L must be less than 64 for spin_basis_general generation."
    S = _check_spin_number(S)
    if S != 0.5:
        raise NotImplementedError("spin_basis_general is only implemented for spin-1/2 now.")
    

    from .spin_half.spin_general.basis import get_permute_number, _check_commute
    # Validate each provided symmetry block is Z2
    allz2 = True
    for _name, (_perm, _sector) in blocks.items():  # type: ignore
        _perm = np.array([
                [L+i, L-a-1, 1] if i < 0 else [L-i-1, L-a-1, 0]
                for a,i in enumerate(_perm)
        ])
        n = get_permute_number(L, _perm)
        blocks[_name] = (_perm, _sector, n)
        if n != 2:
            allz2 = False
    if check_commute:
        _check_commute(L, blocks)
    
    from .spin_half.spin_general.basis import BasisZ2N, BasisZNN
    BasisZN = BasisZ2N if allz2 else BasisZNN
    # BasisZN = BasisZNN
    if Ndiff is None:
        flipset = None
        _Ndiff = None
    else:
        flipset, _Ndiff = Ndiff
    return BasisZN(L, flipset, _Ndiff, Nup, **blocks)



def spin_basis_2d(
    Lx:int, Ly:int,
    Nup:int|None=None, 
    Ndiff:int|None=None,
    pxblock:int|None=None,
    pyblock:int|None=None,
    kxblock:int|None=None,
    kyblock:int|None=None,
    pzxblock:int|None=None,
    pzyblock:int|None=None,
    zblock:int|None=None,
    **blocks
):
    """Generate 2D spin basis with optional Z2 symmetries and Ndiff constraint.

    Parameters
    ----------
    Lx : int
        the length of x direction
    Ly : int
        the length of y direction
    Nup : int | None, optional
        the number of spin up, by default None
    Ndiff : tuple[array_like, int]|None
        Particle number difference constraint. Must be a tuple ``(flipset, Ndiff)``,
    pxblock : int | None, optional
        the x parity block, by default None
    pyblock : int | None, optional
        the y parity block, by default None
    pzxblock : int | None, optional
        the x parity and z parity block, by default None
    pzyblock : int | None, optional
        the y parity and z parity block, by default None
    zblock : int | None, optional
        the z parity block, by default None
    kxblock : int | None, optional
        the x momentum block, by default None
    kyblock : int | None, optional
        the y momentum block, by default None
    """
    N_2d = Lx * Ly  # total number of sites
    s = np.arange(N_2d)  # sites [0,1,2,..]
    x = s % Lx  # x positions for sites
    y = s // Lx  # y positions for sites

    for i,j in blocks:
        assert len(i) == N_2d
        assert isinstance(j, int)
        
    if pxblock is not None and 'pxblock' not in blocks:
        P_x = (Lx - x - 1) + Lx * y
        blocks['pxblock'] = (P_x, pxblock)

    if pyblock is not None and 'pyblock' not in blocks:
        P_y = x + Lx * (Ly - y - 1)
        blocks['pyblock'] = (P_y, pyblock)
    
    if kxblock is not None and 'kxblock' not in blocks:
        T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
        blocks['kxblock'] = (T_x, kxblock)

    if kyblock is not None and 'kyblock' not in blocks:
        T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
        blocks['kyblock'] = (T_y, kyblock)

    if pzxblock is not None and 'pzxblock' not in blocks:
        PZ_x = - ((Lx - x - 1) + Lx * y + 1)
        blocks['pzxblock'] = (PZ_x, pzxblock)

    if pzyblock is not None and 'pzyblock' not in blocks:
        PZ_y = - (x + Lx * (Ly - y - 1) + 1)
        blocks['pzyblock'] = (PZ_y, pzyblock)

    if zblock is not None and 'zblock' not in blocks:
        Z = -(s + 1)
        blocks['zblock'] = (Z, zblock)
    
    return spin_basis_general(
        N_2d, "1/2", Nup=Nup, Ndiff=Ndiff, **blocks
    )


def spin_super_basis(
    L, 
    Nup=None, 
    Ndiff=None, 
    Nup2=None,
    pblock=None, 
    zblock=None, 
    indx_order='stacked', 
    check_commute=True,
    **blocks
):
    assert 2*L < 64, "L must be less than 64 for spin_super_basis generation."
    if pblock is not None:
        if indx_order == 'stacked':
            Px = [i+j*L for j in [0,1] for i in range(L-1,-1,-1)]
        elif indx_order == 'snake':
            Px = [i-j for i in range(2*L-1,-1,-2) for j in [1,0]]
        else:
            raise ValueError(f"indx_order must be 'stacked' or 'snake', but got {indx_order}.")
        blocks.update({'pblock': (Px, pblock)})
    if zblock is not None:
        Pz = [-i-1 for i in range(2*L)]
        blocks.update({'zblock': (Pz, zblock)})
    if Nup2 is not None:
        if isinstance(Nup2, int):
            Nup2 = [Nup2]
        Ndiff = 0
        Nup = [2*n for n in Nup2]
    _Ndiff = [Ndiff] if isinstance(Ndiff, int) else Ndiff
    if _Ndiff is not None:
        if len(_Ndiff) != len(set(_Ndiff + [-nd for nd in _Ndiff])):
            warn(f"Ndiff does not contain both +nd and -nd, cannot realify :(")
    
    from .spin_half.spin_general.basis import get_permute_number, _check_commute
    # Validate each provided symmetry block is Z2
    allz2 = True
    for _name, (_perm, _sector) in blocks.items():  # type: ignore
        _perm = np.array([
                [2*L+i, 2*L-a-1, 1] if i < 0 else [2*L-i-1, 2*L-a-1, 0]
                for a,i in enumerate(_perm)
        ])
        n = get_permute_number(L, _perm)
        blocks[_name] = (_perm, _sector)
        if n != 2:
            allz2 = False
    if check_commute:
        _check_commute(L, blocks)

    if allz2:
        from .spin_half.spin_super.basis import BasisZ2N
        return BasisZ2N(L, Ndiff, Nup, indx_order, **blocks)
    else:
        raise NotImplementedError("" \
            "Only one Z2 is supported for spin_super_basis now.\n" \
            f"not support: {list(blocks.keys())}"
        )

