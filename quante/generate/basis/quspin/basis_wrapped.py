# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-12-15 18:08:18
# @Last Modified by:   hzhu
# @Last Modified time: 2024-12-15 22:02:45

# quspin_basis 是 quspin 库中的源码，这里提供一个简单的封装，使得用户可以更方便地使用 quspin_basis 中的 basis 类。

# todo 文档中文化
# todo boson-spin 类

from .quspin_basis.basis_1d.fermion import spinless_fermion_basis_1d, spinful_fermion_basis_1d
from .quspin_basis.basis_1d.boson import boson_basis_1d

def spinless_fermion_basis(L:int, Nf:int|list|None = None, nf:float|None = None, a:int|None = None, kblock:int|None=None, pblock:int|None=None):
    """quspin spinless fermion basis

    Parameters
    ----------
    L : int
        Length of chain/number of sites.
    Nf : int | list | None, optional
        Number of fermions in chain. Can be integer or list to specify one or more particle sectors, by default None
    nf : float | None, optional
        Density of fermions in chain (fermions per site), by default None
    a : int | None, optional
        specifies unit cell size for translation, by default None
    kblock : int | None, optional
        specifies momentum block. The physical manifestation of this symmetry transformation is translation by `a` lattice sites, by default None
    pblock : int | None, optional
        specifies parity block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain, by default None
    """
    blocks = {'a': a, "kblock": kblock, "pblock": pblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return spinless_fermion_basis_1d(L=L, Nf=Nf, nf=nf, **blocks)

def spinful_fermion_basis(L:int, Nf:tuple[int, list]|None = None, nf: tuple[float]|None = None, double_occupancy:bool=True, a: int|None = None, kblock:int|None = None, pblock:int|None = None, sblock: int|None = None, psblock:int|None = None):
    """quspin spinful fermion basis

    Parameters
    ----------
    L : int
        Length of chain/number of sites.
    Nf : tuple[int, list] | None, optional
        Number of fermions in chain. First (left) entry refers to spin-up and second (right) entry refers
        to spin-down. Each of the two entries can be integer or list to specify one or more particle sectors, by default None
    nf : tuple[float] | None, optional
        Density of fermions in chain (fermions per site). First (left) entry refers to spin-up. Second (right)
        entry refers to spin-down, by default None
    double_occupancy : bool, optional
        Boolean to toggle the presence of doubly-occupied sites (both a spin up and a spin-down fermion present on the same lattice site) in the basis. by default True, for which doubly-occupied states are present
    a : int | None, optional
        specifies unit cell size for translation, by default None
    kblock : int | None, optional
        specifies momentum block. The physical manifestation of this symmetry transformation is translation by `a` lattice sites, by default None
    pblock : int | None, optional
        specifies parity block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain, by default None
    sblock : int | None, optional
        specifies fermion spin inversion block. The physical manifestation of this symmetry transformation is the exchange of a spin-up and a spin-down fermion on a fixed lattice site, by default None
    psblock : int | None, optional
        specifies parity followed by fermion spin inversion symmetry block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain, and a simultaneous exchange of a spin-up and a spin-down fermion on a fixed lattice site, by default None
    """
    blocks = {'a': a, "kblock": kblock, "pblock": pblock, "sblock": sblock, "psblock": psblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return spinful_fermion_basis_1d(L=L, Nf=Nf, nf=nf, double_occupancy=double_occupancy, **blocks)

def boson_basis(L:int, Nb:int|list|None = None, nb:float|None = None, sps:int|None=None, a:int|None = None, kblock:int|None=None, pblock:int|None=None, cblock:int|None = None, pcblock:int|None = None, cAblock:int|None = None, cBblock:int|None=None):
    """quspin boson basis

    Parameters
    ----------
    L : int
        Length of chain/number of sites.
    Nb : int | list | None, optional
        Number of bosons in chain. Can be integer or list to specify one or more particle sectors, by default None
    nb : float | None, optional
        Density of bosons in chain (bosons per site), by default None
    sps: int | None, optional
        Number of states per site (including zero bosons), or on-site Hilbert space dimension, by default None
    a : int | None, optional
        specifies unit cell size for translation, by default None
    kblock : int | None, optional
        specifies momentum block. The physical manifestation of this symmetry transformation is translation by `a` lattice sites, by default None
    pblock : int | None, optional
        specifies parity block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain, by default None
    cblock : int | None, optional
        specifies particle-hole symmetry block. The physical manifestation of this symmetry transformation is the exchange of a hard-core boson for a hole (i.e. no particle).
    pcblock : int | None, optional
        specifies parity followed by particle-hole symmetry block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain, and a simultaneous exchange of a hard-core boson for a hole (i.e. no particle).
    cAblock : int | None, optional
        specifies particle-hole symmetry block for sublattice A (defined as all even lattice sites). The physical manifestation of this symmetry transformation is the exchange of a hard-core boson for a hole (i.e. no particle) on all even sites, by default None
    cBblock : int | None, optional
        specifies particle-hole symmetry block for sublattice B (defined as all odd lattice sites). The physical manifestation of this symmetry transformation is the exchange of a hard-core boson for a hole (i.e. no particle) on all odd sites, by default None
    """
    blocks = {'a': a, "kblock": kblock, "pblock": pblock, "cblock":cblock, "pcblock":pcblock, "cAblock": cAblock, "cBblock": cBblock}
    blocks = {k: v for k, v in blocks.items() if v is not None}
    return boson_basis_1d(L=L, Nb=Nb, nb=nb, sps=sps, **blocks)