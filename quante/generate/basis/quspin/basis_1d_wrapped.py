# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 18:08:18
# @Last Modified by:   hzhu
# @Last Modified time: 2025-02-16 18:12:28

# quspin_basis 是 quspin 库中的源码，这里提供一个简单的封装，使得用户可以更方便地使用 quspin_basis 中的 basis 类。

def quspin_spinless_fermion_basis(L:int, Nf:int|list|None = None, nf:float|None = None, a:int|None = None, kblock:int|None=None, pblock:int|None=None):
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
    from .quspin_basis.basis_1d.fermion import spinless_fermion_basis_1d
    blocks = {'a': a, "kblock": kblock, "pblock": pblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return spinless_fermion_basis_1d(L=L, Nf=Nf, nf=nf, **blocks)

def quspin_spinful_fermion_basis(L:int, Nf:tuple[int, list]|None = None, nf: tuple[float]|None = None, double_occupancy:bool=True, a: int|None = None, kblock:int|None = None, pblock:int|None = None, sblock: int|None = None, psblock:int|None = None):
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
    from .quspin_basis.basis_1d.fermion import spinful_fermion_basis_1d
    blocks = {'a': a, "kblock": kblock, "pblock": pblock, "sblock": sblock, "psblock": psblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return spinful_fermion_basis_1d(L=L, Nf=Nf, nf=nf, double_occupancy=double_occupancy, **blocks)

def quspin_boson_basis(L:int, Nb:int|list|None = None, nb:float|None = None, sps:int|None=2, a:int|None = None, kblock:int|None=None, pblock:int|None=None, cblock:int|None = None, pcblock:int|None = None, cAblock:int|None = None, cBblock:int|None=None):
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
    from .quspin_basis.basis_1d.boson import boson_basis_1d
    blocks = {'a': a, "kblock": kblock, "pblock": pblock, "cblock":cblock, "pcblock":pcblock, "cAblock": cAblock, "cBblock": cBblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return boson_basis_1d(L=L, Nb=Nb, nb=nb, sps=sps, **blocks)

def quspin_spin_basis(L:int, Nup:int=None, pauli=0, S:str="1/2", m:float|None = None, a:int|None = None, 
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
    from .quspin_basis.basis_1d.spin import spin_basis_1d
    blocks = {'a': a,
              "kblock": kblock, 
              "pblock": pblock, 
              "zblock": zblock, 
              "pzblock": pzblock, 
              "zAblock": zAblock, 
              "zBblock": zBblock}
    return spin_basis_1d(L=L, S=S, Nup=Nup, m=m, pauli=pauli, **blocks)