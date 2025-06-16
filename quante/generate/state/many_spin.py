# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-16 16:07:40
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 18:05:36

import itertools
import numpy as _np
import math as _math
from .general import onehot, assemble

def _get_position(num, L, Nup):
    mask = 1 << L
    zeros_left = Nup - 1
    pos = 0
    remain = L - 1
    while mask >= 0 and zeros_left >= 0:
        mask >>= 1
        if num & mask:  # 当前位是1
            pos += _math.comb(remain, zeros_left)
        else: # 当前位是0
            zeros_left -= 1
        remain -= 1
    if not ((mask-1) & num == mask - 1):
        raise ValueError(f"{num} = {_np.binary_repr(num, L)} does not have {Nup} spin up.")
    return pos


def product_state(updns:list[str], dtype=float, Nup=None, sparse=False):
    """generate a product state from a binary string of up and down spins.
    
    Parameters
    ----------
    updns : list of str
        A list of "up" or "dn" strings.
    """
    assert isinstance(updns, list), "updns must be a list of 'up' or 'dn' strings."
    tmp = ""
    for b in updns:
        if b == "up":
            tmp += "0"
        elif b == "dn":
            tmp += "1"
        else:
            raise ValueError(f"Invalid value {b}, must be 'up' or 'dn'.")
    pos = int(tmp, 2)
    L = len(updns)
    if Nup is None:
        dim = 1<<L
    else:
        dim = _math.comb(L, Nup)
        pos = _get_position(pos, len(updns), Nup)
    return onehot(pos, dim, dtype=dtype, sparse=sparse)


def neel(L:int, down_first=False, dtype=float, Nup=None, sparse=False):
    r""" generate a Neel state, which is an alternating up and down state.

    .. math::
        |010101...\rangle 
    or
    .. math::
        |101010...\rangle

    Parameters
    ----------
    L : int
        Length of the chain.
    down_first : bool, optional
        If True, the first spin is down, otherwise it is up.
    dtype : type, optional
        Data type of the output state, default is float.
    Nup : int, optional
        Number of up spins in the state. If specified, the generated state will
    """
    updns = "01" * (L // 2) + (L % 2 == 1) * "0"
    if down_first:
        updns = "1" + updns[:-1]
    pos = int(updns, 2)
    if Nup is None:
        dim = 1<<L
    else:
        dim = _math.comb(L, Nup)
        pos = _get_position(pos, len(updns), Nup)
    return onehot(pos, dim, dtype=dtype, sparse=sparse)


def ghz(L, dtype=float, sparse=False):
    r"""Generate a GHZ state, which is a superposition of all spins being up and all spins being down.
    
    .. math::
        (|000...0\rangle + |11...1\rangle) / \sqrt{2}
    """
    c = 1 / 2.**.5
    return assemble([c, c], [0, 2**L - 1], 2**L, sparse=sparse).astype(dtype)


def w(L, dtype=complex, Nup=None):
    r"""generate a W state, which is a superposition of all spins being usp and one spin being down.
    
    .. math::
        |w\rangle = \frac{1}{\sqrt{L}} \sum_{i=0}^{L-1} |0...01...0\rangle
    """
    assert Nup in [None, 1], "W state only supports Nup=1 or None"

    c = 1.0 / L**0.5
    if Nup is None:
        x = _np.zeros((2**L, 1), dtype=dtype)
        for i in range(L):
            x[1<<i, 0] = c
        return x
    else:
        x = _np.ones((L, 1), dtype=dtype) * c
        return x


def _levi_civita(perm):
    """Compute the generalised levi-civita coefficient for a permutation.

    Parameters
    ----------
    perm : sequence of int
        The permutation, a re-arrangement of ``range(n)``.

    Returns
    -------
    int
        Either -1, 0 or 1.
    """
    n = len(perm)
    if n != len(set(perm)):  # infer there are repeated elements
        return 0
    mat = _np.zeros((n, n), dtype=_np.int32)
    for i, j in zip(range(n), perm):
        mat[i, j] = 1
    return int(_np.linalg.det(mat))


def antisymmetric(*states):
    """Construct the anti-symmetric state which is the +- sum of all
    tensored permutations of states ``ps``.

    Parameters
    ----------
    *states : _np.ndarray
        The states to combine.

    Returns
    -------
    vector or operator
        The permutation state, dimension same as ``kron(*ps)``.

    Examples
    --------
    A singlet is the ``perm_state`` of up and down.

    >>> states = [up(), down()]
    >>> pstate = perm_state(states)
    >>> expec(pstate, singlet())
    1.0
    """
    n = len(states)
    vec_perm = itertools.permutations(states)
    ind_perm = itertools.permutations(range(n))

    def terms():
        for vec, ind in zip(vec_perm, ind_perm):
            yield _levi_civita(ind) * kron(*vec) # type: ignore

    return sum(terms()) / _math.factorial(n)**0.5

