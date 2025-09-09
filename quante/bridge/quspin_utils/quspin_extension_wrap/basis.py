# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 18:08:18
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-09 00:41:30

# quspin_basis 是 quspin 库中的源码，这里提供一个简单的封装，使得用户可以更方便地使用 quspin_basis 中的 basis 类。

import numpy as np

def fermion_basis(L:int, Nf:int|list|None = None, nf:float|None = None, a:int|None = None, kblock:int|None=None, pblock:int|None=None):
    """使用 quspin 生成无自旋费米子基组。

    Parameters
    ----------
    L : int
        链长
    Nf : int | list | None, optional
        链中费米子的个数/激发数，可以是整数或者列表，列表表示不同激发数构成的子空间, by default None
    nf : float | None, optional
        费米子的密度，即链中费米子的个数/激发数占总数的比例，by default None
    a : int | None, optional
        平移的单位长度，by default None
    kblock : int | None, optional
        动量子空间，单位长度由 a 指定, by default None
    pblock : int | None, optional
        宇称子空间，反射对称性, by default None, by default None
    """
    from quspin.basis import spinless_fermion_basis_1d
    blocks = {'a': a, "kblock": kblock, "pblock": pblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return spinless_fermion_basis_1d(L=L, Nf=Nf, nf=nf, **blocks)

def spinful_fermion_basis(L:int, Nf:tuple[int, list]|None = None, nf: tuple[float]|None = None, double_occupancy:bool=True, a: int|None = None, kblock:int|None = None, pblock:int|None = None, sblock: int|None = None, psblock:int|None = None):
    """使用 quspin 生成有自旋费米子基组。

    Parameters
    ----------
    L : int
        链长
    Nf : tuple[int, list] | None, optional
        链中费米子的个数，第一个数表示自旋向上的费米子个数，第二个数表示自旋向下的费米子个数，可以是整数或者列表，列表表示不同激发数构成的子空间, by default None
    nf : tuple[float] | None, optional
        费米子的密度，即链中费米子的个数占总数的比例，第一个数表示自旋向上的费米子密度，第二个数表示自旋向下的费米子密度，by default None
    double_occupancy : bool, optional
        是否允许双占据，即一个格点上同时存在一个自旋向上的和一个自旋向下的费米子，by default True
    a : int | None, optional
        平移的单位长度，by default None
    kblock : int | None, optional
        动量子空间，单位长度由 a 指定, by default None
    pblock : int | None, optional
        宇称子空间，反射对称性, by default None, by default None
    sblock : int | None, optional
        费米子的上下自旋交换对称性, by default None
    psblock : int | None, optional
        同时考虑自旋交换和反射对称性, by default None
    """
    from quspin.basis import spinful_fermion_basis_1d
    blocks = {'a': a, "kblock": kblock, "pblock": pblock, "sblock": sblock, "psblock": psblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return spinful_fermion_basis_1d(L=L, Nf=Nf, nf=nf, double_occupancy=double_occupancy, **blocks)

def boson_basis(L:int, Nb:int|list|None = None, nb:float|None = None, sps:int|None=2, a:int|None = None, kblock:int|None=None, pblock:int|None=None, cblock:int|None = None, pcblock:int|None = None, cAblock:int|None = None, cBblock:int|None=None):
    """使用 quspin 生成玻色子基组。

    Parameters
    ----------
    L : int
        链长
    Nb : int | list | None, optional
        玻色子的个数，可以是整数或者列表，列表表示不同激发数构成的子空间, by default None
    nb : float | None, optional
        每个格点上的玻色子的密度，即玻色子的个数占总数的比例，by default None
    sps: int | None, optional
        每个格点上运行的状态数, or on-site Hilbert space dimension, by default None
    a : int | None, optional
        平移的单位长度，by default None
    kblock : int | None, optional
        动量子空间，单位长度由 a 指定, by default None
    pblock : int | None, optional
        宇称子空间，反射对称性, by default None, by default None
    cblock : int | None, optional
        (hard-core) 粒子空穴对称性
    pcblock : int | None, optional
        粒子空虚和反射对称性
    cAblock : int | None, optional
        偶数格点的粒子空穴对称性
    cBblock : int | None, optional
        奇数格点上的粒子空穴对称性
    """
    from quspin.basis import boson_basis_1d
    blocks = {'a': a, "kblock": kblock, "pblock": pblock, "cblock":cblock, "pcblock":pcblock, "cAblock": cAblock, "cBblock": cBblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return boson_basis_1d(L=L, Nb=Nb, nb=nb, sps=sps, **blocks)

def spin_basis(L:int, pauli, Nup:int=None, S:str="1/2", m:float|None = None, a:int|None = None, 
                      kblock:int|None=None, pblock:int|None=None, zblock:int|None=None, 
                      pzblock:int|None=None, zAblock:int|None=None, zBblock:int|None=None):
    """使用 quspin 生成自旋基组。

    Parameters
    ----------
    L : int
        链长
    S : int
        总自旋量子数
    m : float, optional
        链中自旋向上的密度，即链中自旋向上的个数占总数的比例, by default None
    a : int | None, optional
        平移的单位长度，by default None
    kblock : int | None, optional
        动量子空间，单位长度由 a 指定, by default None
    pblock : int | None, optional
        宇称子空间，反射对称性, by default None, by default None
    zblock : int | None, optional
        自旋翻转对称性, by default None
    pzblock : int | None, optional
        自旋翻转和反射对称性, by default None
    zAblock : int | None, optional
        A 子格点上的自旋翻转对称性, by default None
    zBblock : int | None, optional
        B 子格点上的自旋翻转对称性, by default None
    """
    from quspin.basis import spin_basis_1d
    blocks = {'a': a,
              "kblock": kblock, 
              "pblock": pblock, 
              "zblock": zblock, 
              "pzblock": pzblock, 
              "zAblock": zAblock, 
              "zBblock": zBblock}
    return spin_basis_1d(L=L, S=S, Nup=Nup, m=m, pauli=pauli, **blocks)


def spin_basis_2d(Lx, Ly, pauli=0, Nup=None, kxblock=None, kyblock=None, pxblock=None, pyblock=None, zblock=None):
    """使用 quspin 生成自旋梯子基组。

    采用编号方案：
    .. code-block:: text
        |    ---> y
        |   |
        | x V  1     3      5               2L-1
        |      o --- o  --- o  ---  ...  --- o
        |      o --- o  --- o  ---  ...  --- o
        |      2     4      6                2L

    Parameters
    ----------
    L : int
        梯子的长度
    pauli : int, optional
        保留的泡利矩阵, by default 0
    Nup : int, optional
        自旋向上的粒子数, by default None
    kxblock : int, optional
        动量子空间，x方向, by default None
    kyblock : int, optional
        动量子空间，y方向, by default None
    pxblock : int, optional
        宇称子空间，x方向, by default None
    pyblock : int, optional
        宇称子空间，y方向, by default None
    zblock : int, optional
        自旋翻转对称性, by default None
    """
    from quspin.basis import spin_basis_general
    N_2d = Lx * Ly  # total number of sites
    s = np.arange(N_2d)  # sites [0,1,2,..]
    x = s % Lx  # x positions for sites
    y = s // Lx  # y positions for sites
    
    _kxblock = None
    if kxblock is not None:
        T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
        _kxblock = (T_x, kxblock)
    
    _kyblock = None
    if kyblock is not None:
        T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
        _kyblock = (T_y, kyblock)
    
    _pxblock = None
    if pxblock is not None:
        P_x = (Lx - x - 1) + Lx * y
        _pxblock = (P_x, pxblock)
    
    _pyblock = None
    if pyblock is not None:
        P_y = x + Lx * (Ly - y - 1)
        print(P_y)
        _pyblock = (P_y, pyblock)
    
    _zblock = None
    if zblock is not None:
        Z = -(s + 1)
        _zblock = (Z, zblock)
    
    basis = spin_basis_general(
        N_2d,
        S='1/2',
        pauli=pauli,
        Nup=Nup,
        kxblock=_kxblock,
        kyblock=_kyblock,
        pxblock=_pxblock,
        pyblock=_pyblock,
        zblock=_zblock,
    )
    return basis

