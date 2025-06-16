# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-04-19 17:58:24
# @Last Modified by:   hzhu
# @Last Modified time: 2025-06-16 19:12:42

import numpy as np
from functools import lru_cache
from ....linalg import expm


def ising_evolve(L, model, init_state:str, tlist:np.ndarray):
    """Example of ising evolution (measure particle number)

    Parameters
    ----------
    model : FermionOper, SpinOper
        the model to evolve, must be a fermion operator
    init_state : str
        initial state string, e.g. '0000000000000000', 
        length must match the model
    tlist : np.ndarray
        time list, e.g. np.linspace(0, 60, 500)

    Returns
    -------
    np.ndarray
        measure the number of particles in the system,
        the measurement result, shape (len(tlist), len(init_state)), 
        each column is the corresponding measurement value

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import quante as qt
    >>> import numpy as np
    >>> from quante.solvable_models import gaussian_state
    >>> L = 100
    >>> builder = qt.generate.operas.SpinBuilder()
    >>> for i in range(L-1):
    ...     builder += "xx", [i+1, i], 1.
    >>> for i in range(L):
    ...     builder += "z", [i], 1.
    >>> ham = builder.build()
    >>> tlist = np.linspace(0, 60, 1000)
    >>> result = gaussian_state.ising_evolve(
    ...     L, ham, '0'*L, tlist
    ... )
    >>> plt.plot(tlist, result)
    >>> plt.show()
    """
    # convert to fermion operator
    from ...operas import spin, fermion
    if isinstance(model, spin.SpinOper):
        model = model.jw_transfer()
    assert isinstance(model, fermion.FermionOper), "model must be a fermion operator"

    # evolve the model
    h, coef_I = model.BdG_ham(L)
    state = PairingState.from_product_state(init_state)
    result = []
    for s in state.evolve(h, tlist):
        result.append(np.sum(s.particle_number()))
    return np.abs(result)


class PairingState:
    r"""配对态 又称 BCS态 (Bardeen-Cooper-Schrieffer state)

    .. math:
        |\psi\rangle = \mathcal{N} \exp\left( \frac{1}{2} \sum_{i,j} Z_{ij} c_i^\dagger c_j^\dagger \right) |0\rangle,
    
    其中 `Z_{ij}` 是一个反对称矩阵, 称为配对关联矩阵。

    但为了数值方便，用协方差矩阵`\Gamma`编码配对态, 协方差矩阵的（反对称的）定义为：
    .. math:
        \Gamma_{ab} = \frac{1}{2} \langle [\Psi_a, \Psi_b^\dagger] \rangle,
    其中 `\Psi` 是 Nambu 自旋表示
    .. math:
        \Psi = \begin{bmatrix}
                    c \\
                    c^\dagger
                \end{bmatrix},
    数值上更常用的形式是：
    .. math:
        \Gamma_{ab} = \langle \Psi_a\Psi_b^\dagger \rangle,
    
    协方差矩阵按分块形式可以定义粒子数矩阵 `G` 和配对矩阵 `F`:
    .. math:
        \Gamma = \begin{bmatrix}
                    G & F \\
                    F^\dagger & I - G^T
                \end{bmatrix},
    通过它们，协方差矩阵与配对关联矩阵之间的关系是：
    .. math:
        Z = F (I - G)^{-1}.
    """
    def __init__(self, Gamma):
        assert Gamma.shape[0] % 2 == 0, "协方差矩阵的行数必须是偶数"
        # ? save Gamma or G/F?
        self.Gamma = Gamma  # 协方差矩阵
        self.L = Gamma.shape[0] // 2
    
    @property
    def G(self):
        return self.Gamma[:self.L, :self.L]  # 粒子数矩阵

    @property
    def F(self):
        return self.Gamma[:self.L, self.L:]

    @classmethod
    def from_product_state(cls, state:str):  # todo spin 对应的应该是什么样子?
        L = len(state)
        G = np.zeros((L, L), dtype=complex)
        F = np.zeros((L, L), dtype=complex)
        for i, s in enumerate(state):
            if s == '1':
                G[i, i] = 1
        Gamma = np.block([[G, F], [F.conj().T, np.eye(G.shape[0]) - G.T]])
        return cls(Gamma)

    def evolve(self, BdG, tlist):
        r"""协方差矩阵的演化遵循动力学方程
        .. math:
            \frac{d\Gamma}{dt} = -i \left( \mathcal{H} \Gamma - \Gamma \mathcal{H}^\dagger \right),
        
        也就是
        .. math:
            \Gamma(t) = e^{-i\mathcal{H}t} \Gamma(0) e^{i\mathcal{H}^\dagger t}.
        """
        dtlist = [tlist[0]] + list(np.diff(tlist))

        if isinstance(BdG, tuple):
            BdG, coef_I = BdG
            # 如果 BdG() 返回的是一个 tuple,
            # 说明包含一个常数项
            # 那么严格来说，vector 将会相差一个相位因子 `exp(-i*coef_I*t)`
            # 但是这个相位因子不会影响到物理量的计算
            # 所以这里就不考虑了
            # （也可以考虑给 SlaterState 添加一个 coef_I 属性，需要的时候再改吧）

        isherm = np.allclose(BdG, BdG.conj().T)
        @lru_cache
        def exph(dt):
            return expm(BdG, -1j * dt, isherm=isherm)
        
        cur_state = PairingState(self.Gamma.copy())
        for dt in dtlist:
            if dt != 0:
                U = exph(round(dt, 12))  # 增加缓存命中率
                cur_state.Gamma =  U @ cur_state.Gamma @ U.conj().T
                if not isherm:
                    pass # todo 如何归一
            yield cur_state

    def particle_number(self, pos:int|list=None):
        r"""粒子数算符的期望值
        .. math:
            \langle \hat{N} \rangle = \text{Tr}(G).
        """
        if pos is None:
            return np.diag(self.G)
        elif isinstance(pos, int):
            return self.G[pos, pos]
        elif isinstance(pos, list):
            return np.sum(self.G[pos, pos])
        else:
            raise ValueError("pos must be None, int or list")

# todo 验证 PairingState 的正确性
