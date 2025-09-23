# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-22 13:10:02
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-22 13:15:55

import numpy as np
from typing import Literal

def liouvillian(L, ham, lindblad_ops, format:Literal['chain', 'ladder']='chain'):
    r"""Create a Liouvillian operator in a operator format on vectorized space.

    Notes
    -----
    The Liouvillian is given by the following equation:
    .. math::
        \mathcal{L}(\rho) = -i [H, \rho] + \sum_{l} L_l \rho L_l^{\dagger} - \frac{1}{2} \sum_{l} (L_l^{\dagger} L_l \rho + \rho L_l^{\dagger} L_l)
    
    where :math:`H` is the Hamiltonian, :math:`L_l` are the Lindblad operators, and :math:`\rho` is the density matrix.

    In vectorized space, the Liouvillian can be represented as:
    .. math::
        \mathcal{L} = -i (H_{eff} \otimes I - I \otimes H_{eff}^*) + \sum_{l} L_l \otimes L_l^* 
    
    where :math:`\otimes` denotes the Kronecker product, :math:`I` is the identity operator and 
    :math:`H_{eff} = H - \frac{i}{2} \sum_{l} L_l^{\dagger} L_l` is the effective non-Hermitian Hamiltonian.

    The space the operator acts on is :math:`\mathcal{H} \otimes \mathcal{H}`, where :math:`\mathcal{H}` is the Hilbert space of the system. We denote the left and right parts of the space as the first and second :math:`\mathcal{H}` respectively.

    The space is ordered as follows:
    - 'chain' format, the default order by kronecker product,
        .. code-block:: text
            |            left sites                         right sites
            |    o --- o  --- o  ---  ...  --- o      o --- o  --- o  ---  ...  --- o
            |    1     2      3                L     L+1   L+2    L+3               2L
            
    - 'ladder' format, the order is like a ladder,
        .. code-block:: text
            |
            |    1     3      5               2L-1   left sites
            |    o --- o  --- o  ---  ...  --- o
            |    o --- o  --- o  ---  ...  --- o     right sites
            |    2     4      6                2L

    Parameters
    ----------
    L : int
        The total number of sites.
    ham : Operator
        The Hamiltonian operator.
    lindblad_ops : list of Operators
        The Lindblad operators.
    format : str, optional
        The format of the Liouvillian operator, by default 'chain'.

    Returns
    -------
    Operator
        The Liouvillian operator.
    """
    res = ham.builder()

    if format == 'chain':
        # transpose ham
        for opstr, posn, coef in ham.each_term():
            # H oxx I
            res += opstr, posn, coef * (-1j)
            # I oxx H^*
            num_y = opstr.count('y')
            res += opstr, [p+L for p in posn], (-1)**num_y * np.conj(coef) * (1j)
        
        for lo in lindblad_ops:
            for opstr, posn, coef in (lo.hc() @ lo).each_term():
                # Ldag L oxx I
                res += opstr, posn, coef * (-0.5)
                # I oxx (Ldag L)^*
                num_y = opstr.count('y')
                res += opstr, [p+L for p in posn], (-1)**num_y * np.conj(coef) * (-0.5)
            
            for opstr1, posn1, coef1 in lo.each_term():
                for opstr2, posn2, coef2 in lo.each_term():
                    # L oxx L^*
                    num_y = opstr2.count('y')
                    res += (
                        opstr1 + opstr2, 
                        list(posn1) + [p+L for p in posn2], 
                        (-1)**num_y * coef1 * np.conj(coef2)
                    )
    
    elif format == 'ladder':
        for opstr, posn, coef in ham.each_term():
            # H oxx I
            res += opstr, [2*p for p in posn], coef * (-1j)
            # I oxx H^*
            num_y = opstr.count('y')
            res += opstr, [2*p+1 for p in posn], (-1)**num_y * np.conj(coef) * (1j)
        
        for lo in lindblad_ops:
            for opstr, posn, coef in (lo.hc() @ lo).each_term():
                # Ldag L oxx I
                res += opstr, [2*p for p in posn], coef * (-0.5)
                # I oxx (Ldag L)^*
                num_y = opstr.count('y')
                res += opstr, [2*p+1 for p in posn], (-1)**num_y * np.conj(coef) * (-0.5)
            
            for opstr1, posn1, coef1 in lo.each_term():
                for opstr2, posn2, coef2 in lo.each_term():
                    # L oxx L^*
                    num_y = opstr2.count('y')
                    newoper = ''.join(
                        opstr1[i] if j == 0 else opstr2[i] 
                        for i in range(len(opstr1))
                        for j in range(2)
                    )
                    newposn = [
                        2*posn1[i] if j == 0 else 2*posn2[i] + 1
                        for i in range(len(opstr1))
                        for j in range(2)
                    ]
                    res += (
                        newoper,
                        newposn,
                        (-1)**num_y * coef1 * np.conj(coef2)
                    )
    else:
        raise ValueError("format should be 'chain' or 'ladder'")

    return res.build()

