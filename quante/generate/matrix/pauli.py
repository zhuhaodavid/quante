# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2023-10-22 16:50:19
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-17 10:43:33

import functools

import re
import numpy as _np

from ...linalg.matops import kron
from ..basis.symmetry.basis_wrapped import _check_spin_number # type: ignore

from typing import Callable, Union 

__all__ = [
    "pauli_matrix",
]

PAULI_MAT = {
    "X": _np.array([[0.,1.],[1.,0.]]),
    "Y": _np.array([[0.,-1.j],[1.j,0.]]),
    "Z": _np.array([[1.,0.],[0.,-1.]]),
    "P": _np.array([[0.,1.],[0.,0.]]),
    "M": _np.array([[0.,0.],[1.,0.]]),
    "I": _np.array([[1.,0.],[0.,1.]]),
    "x": _np.array([[0.,0.5],[0.5,0.]]),
    "y": _np.array([[0.,-0.5j],[0.5j,0.]]),
    "z": _np.array([[0.5,0.],[0.,-0.5]]),
    "p": _np.array([[0.,1.],[0.,0.]]),
    "m": _np.array([[0.,0.],[1.,0.]]),
    "+": _np.array([[0.,1.],[0.,0.]]),
    "-": _np.array([[0.,0.],[1.,0.]]),
    "i": _np.array([[1.,0.],[0.,1.]]),
    "u": _np.array([[1.], [0.]]),
    "d": _np.array([[0.], [1.]]),
}

for k, v in PAULI_MAT.items():
    v.setflags(write=False)

def pauli_matrix(
    stri: str,
    S: Union[str, float, int] = '1/2',
    view: bool = False,
) -> _np.ndarray:
    """
    !! 多个算符的时候 `+`, `-`, `u`, `d` 这四个算符一定要十分小心，**最好不要用这四个算符**
    
    - `xx` -> kron(pauli_x, pauli_x)
    - `x3y3` -> `xxxyyy`
    - `x2y3!5` -> `iixyi`
    - `xy!5` -> `iiixy+iixyi+ixyii+xyiii`
    
    Examples
    --------
    >>> op.pauli_matrix("x")
    >>> op.pauli_matrix("xx+yy")
    >>> op.pauli_matrix("xx+3*yy")

    Generates a Pauli operator matrix based on the input string.

    Parameters
    -----------
        - `stri`: String specifying the Pauli operator configuration.
        - `dtype`: Data type of the resulting matrix (default: np.complex128).
        - `view`: If True, prints the intermediate result (default: False).
    """
    iS: Union[int, float] = _check_spin_number(S)

    # 定义获得单个算符的方法
    if iS == 0.5:
        _single_oper = functools.partial(_pauli_matrix_single)
    else:
        _single_oper = functools.partial(_spin_oper_single, S=iS)

    # 如果长度是 1, 直接返回结果
    if len(stri) == 1:
        res = _single_oper(stri).copy()
    else:
        # 用正则表达式转化为可识别的形式
        evalable_string = _standardlize_string(stri)  
        if view:
            print(evalable_string)
        res = _evaluate_string(evalable_string, _single_oper, kron)  # 运行字符串
        
    return _np.real_if_close(res)
        
def _pauli_matrix_single(stri):
    return PAULI_MAT[stri]

def _spin_oper_single(
    label:str,
    S:Union[float, int]=0.5,
):
    D = 2 * S + 1
    assert D == int(D), f"{S} 不是整数或半整数"
    D = int(D)
    op = _np.zeros((D, D), dtype=complex)
    ms = _np.linspace(S, -S, D)
    label = label.lower()
    if label in {'x', 'y'}:
        for i in range(D - 1):
            c = 0.5 * (S * (S + 1) - (ms[i] * ms[i + 1]))**0.5
            op[i, i + 1] = -1.0j * c if (label == 'y') else c
            op[i + 1, i] = 1.0j * c if (label == 'y') else c
    elif label == 'z':
        for i in range(D):
            op[i, i] = ms[i]
    elif label in {'+', 'p', '-', 'm'}:
        for i in range(D - 1):
            c = (S * (S + 1) - (ms[i] * ms[i + 1]))**0.5
            if label in {'+', 'p'}:
                op[i, i + 1] = c
            else:
                op[i + 1, i] = c
    elif label in {'i', 'I'}:
        _np.fill_diagonal(op, 1.0)
    else:
        raise ValueError(f"Label '{label}'' not understood, should be one of "
                         "``['X', 'Y', 'Z', '+', '-', 'I']``.")

    return op

def _standardlize_string(stri):
    new = re.sub(r"([xyzpmiXYZPMIud]+[0-9]+)+\![0-9]+", _term, stri)  # x1y2!4 -> ixyi
    new = re.sub(r"[xyzpmiXYZPMIud]+\![0-9]+", _shift, new)  # xy!4 -> xy!5 -> iiixy+iixyi+ixyii+xyiii
    new = re.sub(r"[xyzpmiXYZPMIud]+[0-9]+", _duplicate, new)  # x3y3 -> xxxyyy
    new = re.sub("[xyzpmiXYZPMIud]+", _rpmethod, new)
    return new

def _term(match):
    res = match.group()
    res = re.split(r'(\d+)', res)
    n = eval(res[-2])
    out = ['i']*n
    for i in range(0, len(res)-3, 2):
        ind = eval(res[i+1])
        out[ind] = res[i]
    return "".join(out)

def _shift(match):
    res = match.group()
    stri, n = res.split("!")
    n = eval(n)
    res = ""
    for i in range(n-len(stri), -1, -1):
        res += (stri + "i"*i).rjust(n, 'i')
        res += "+"
    return res[:-1]

def _duplicate(match):
    res = match.group()
    res = res[0] * eval(res[1:])
    return res

def _rpmethod(match):
    res = "_kron("
    for xi in match.group():
        res += "_single_oper('" +  xi + "'),"
    return res[:-1] + ")"

def _evaluate_string(evalable_string, _single_oper, _kron):
    assert isinstance(_single_oper, Callable)
    assert isinstance(_kron, Callable)
    return eval(evalable_string)
