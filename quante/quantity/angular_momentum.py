# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:41:02
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-11 22:43:31


def cg_coef(j1:float, j2:float, j3:float, m1:float, m2:float, m3:float) -> float:
    """Clebsch-Gordon coefficient 系数.
    
    (j1,m1) 与 (j2,m2) 耦合成为 (j3,m3) 的系数
    
    Parameters
    ----------
    j1 : float
        1 的总角动量
    j2 : float
        2 的总角动量
    j3 : float
        3 的总角动量
    m1 : float
        1 角动量的 z 分量
    m2 : float
        2 角动量的 z 分量
    m3 : float
        3 角动量的 z 分量

    Returns
    -------
    cg_coeff : float
        相应的 cg 系数
    
    Examples
    --------
    计算 [1] 中第一个数: (1/2,1/2) + (1/2,1/2) => (1,1)
    
    >>> import quante.quantity as qq
    >>> qq.cg_coef(1/2, 1/2, 1, 1/2, 1/2, 1)
    1.0
    
    References
    ----------
    [1]. https://github.com/BrandonHenke/phy803/blob/main/clebrpp.pdf
    """
    from .nbfuc.am_nb import clebsch
    return clebsch(j1, j2, j3, m1, m2, m3)

