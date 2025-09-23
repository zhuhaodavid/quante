# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-08 14:20:14
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-08 14:48:51


def hamiltonian(
    oper, basis, dtype, 
    *, 
    sparse=True, check_symm=False, check_herm=False, check_pcon=False,
    **kwargs
):
    from quspin.operators import hamiltonian
    _kwargs = {
        "dynamic_list": []
    }
    _kwargs.update(kwargs)
    if not isinstance(oper, list):
        oper = oper.to_quspin()
    ham = hamiltonian(
        static_list=oper,
        basis=basis,
        dtype=dtype,
        check_symm=check_symm,
        check_herm=check_herm,
        check_pcon=check_pcon,
        **_kwargs
    )
    if sparse:
        return ham.tocsr()
    else:
        return ham.toarray()








