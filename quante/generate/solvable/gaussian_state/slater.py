# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-04-19 10:10:28
# @Last Modified by:   hzhu
# @Last Modified time: 2025-09-30 18:42:28

import numpy as np
from functools import lru_cache
from typing import Literal
from ....linalg import expm
from tqdm import tqdm

def xx_evolve(L, model, init_state:str, tlist:np.ndarray):
    """Example of free fermion evolution (measure particle number)

    Parameters
    ----------
    model : FermionOper, SpinOper
        the model to evolve, must be a fermion operator
    init_state : str
        initial state string, e.g. ['dn', 'up'] * (L//2), 
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
    >>> import quante as qt
    >>> from quante.solvable_models import gaussian_state
    >>> L = 100
    >>> J, γ = 1, 0.8
    >>> builder = qt.generate.operas.SpinBuilder()
    >>> for i in range(L-1):
    ...     builder += "+-", [i+1, i], (J+γ)/2
    ...     builder += "+-", [i, i+1], (J-γ)/2
    >>> ham = builder.build()
    >>> result = gaussian_state.xx_evolve(
    ...     L, ham, '01'*(L//2), np.linspace(0, 60, 100)
    ... )
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(result, aspect='auto', cmap='Blues_r', origin='lower', extent=(0, 100, 0, 60))
    >>> plt.clim(0,1.0)
    >>> plt.colorbar()
    >>> plt.show()
    """
    # convert to fermion operator
    from ...operas import spin, fermion
    if isinstance(model, spin.SpinOper):
        model = model.jw_transfer()
    assert isinstance(model, fermion.FermionOper), "model must be a fermion operator"

    # evolve the model
    h, coef_I = model.single_particle_ham(L)
    state = SlaterState.from_product_state(init_state)
    result = []
    for s in state.evolve(h, tlist):
        result.append(s.particle_number())
    return np.array(result)


class SlaterState:
    r"""
    .. math:
        |ψ(t)\rangle = \prod_{i=1}^N (\sum_{j=1}^L U_{ji} c_j^\dagger) |0\rangle

    """
    def __init__(self, U:np.ndarray):
        self.U = U
        self.L = U.shape[0]
        self.M = U.shape[1]
    
    @classmethod
    def from_product_state(cls, state:list[str], spin=False):
        r"""生成自由费米子态的 U 矩阵"""
        for i in state:
            if i not in ['up', 'dn']:
                raise ValueError(f"state must be a list of 'up' and 'dn', got {i}")
        L = len(state)
        inds = np.flatnonzero(np.array(list(state)) == 'up')
        M = len(inds)
        assert 0 < M < L, "vacuum state or full state is not allowed"
        U = np.zeros((L, M), dtype=complex)
        U[inds, np.arange(M)] = 1.0
        # for spin need to add a minus sign
        # e.g. '0101' -> (-1) ** (1+3) = 1
        #      '0100' -> (-1) ** 1 = -1
        if spin:
            U[:, 0] *= (-1)**sum(inds)
        return cls(U)
    
    @classmethod
    def from_spinful_product_state(cls, state:str, mode:Literal['near','extend']='near'):
        """生成自由费米子态的 U 矩阵

        Parameters
        ----------
        state : str
            需要生成的态，格式为 '0101|1010'，前半部分为上自旋，
            后半部分为下自旋
        mode : Literal[&#39;near&#39;,&#39;extend&#39;], optional
            编号方案, by default 'near'

        Returns
        -------
        SlaterState
            SlaterState 对象

        Raises
        ------
        ValueError
            if mode is not 'near' or 'extend'
        
        Example
        -------
        >>> state = '10' * (L//2) + '|' + '01' * (L//2)
        >>> state = np.real(slater.SlaterState.from_spinful_product_state(state))
        >>> for i in range(state.shape[0]):
        ...    print(i//2, ('up' if i%2==0 else 'dn'), state[i])
        >>> state = np.real(slater.SlaterState.from_spinful_product_state(state, mode='extend'))
        >>> for i in range(state.shape[0]):
        ...    print(i%L, ('up' if i//L==0 else 'dn'), state[i])
        """
        spinup, spindown = state.split('|')
        L = len(spinup)
        assert len(spindown) == L, "spinup and spindown must have the same length"
        inds1 = np.flatnonzero(np.array(list(spinup)) == '1') 
        inds2 = np.flatnonzero(np.array(list(spindown)) == '1') 
        M = len(inds1) + len(inds2)
        U = np.zeros((2*L, M), dtype=complex)
        if mode == 'near':
            U[2*inds1, np.arange(0,M,2)] = 1.0    
            U[2*inds2+1, np.arange(1,M,2)] = 1.0    
        elif mode == 'extend':
            U[inds1, np.arange(len(inds1))] = 1.0
            U[L+inds2, np.arange(len(inds1),M)] = 1.0
        else:
            raise ValueError("mode must be 'near' or 'extend'")
        return cls(U)

    def orthogonalize(self):
        r"""正交化 U 矩阵

        通过 QR 分解实现正交化
        """
        self.U, _ = np.linalg.qr(self.U)

    def evolve(self, h, tlist):
        dtlist = [tlist[0]] + list(np.diff(tlist))

        if isinstance(h, tuple):
            h, coef_I = h
            # 如果 single_particle_ham() 返回的是一个 tuple,
            # 说明包含一个常数项
            # 那么严格来说，vector 将会相差一个相位因子 `exp(-i*coef_I*t)`
            # 但是这个相位因子不会影响到物理量的计算
            # 所以这里就不考虑了
            # （也可以考虑给 SlaterState 添加一个 coef_I 属性，需要的时候再改吧）

        isherm = np.allclose(h, h.conj().T)
        @lru_cache
        def exph(dt):
            return expm(h, -1j * dt, isherm=isherm)
        
        # add tqdm
        if len(dtlist) > 1:
            dtlist = tqdm(dtlist, desc="Evolving", unit="step", ascii=True)
        else:
            dtlist = tqdm(dtlist, desc="Evolving", unit="step", disable=True, ascii=True)
        
        cur_state = SlaterState(self.U.copy())
        for dt in dtlist:
            # update tqdm
            dtlist.set_postfix_str(f"dt={dt:.2f}")
            dtlist.refresh()
            
            if dt != 0:
                dtp = round(dt, 12)  # 增加缓存命中率
                cur_state.U = exph(dtp) @ cur_state.U
                if not isherm:
                    cur_state.orthogonalize()
            yield cur_state

    def particle_number(self, pos:int|list=None):
        r"""measure the particle number at position 
        
        .. math:
            \langle c_i^\dagger c_i \rangle 

        if pos is None, return the particle number at all positions

        Parameters
        ----------
        pos : int | list, optional
            position or positions, by default None

        Returns
        -------
        np.ndarray | float
            the particle number at position pos, shape (L,) if pos is None, else scalar

        Raises
        ------
        TypeError
            if pos is not int or None
        """
        if pos is None:
            return np.linalg.norm(self.U, axis=1)**2
        elif isinstance(pos, int):
            return np.linalg.norm(self.U[pos, :])**2
        elif isinstance(pos, (list, np.ndarray, tuple, range)):
            pos = np.array(pos, dtype=int)
            return np.linalg.norm(self.U[pos, :], axis=1)**2
        else:
            raise TypeError("pos must be None, int, list, tuple, or np.ndarray")
    
    def correlation(self, pos1:int=None, pos2:int=None):
        r"""the correlation between two positions
        
        .. math:
            \lange c_i^\dagger c_j \rangle
           
        Parameters
        ----------
        pos1 : int, optional
            first position, if is None, set to 
            all positions, by default None
        pos2 : int, optional
            second position, if pos2 is None, set to 
            L//2 - 1, by default None

        Returns
        -------
        np.ndarray
            the correlation value, shape (L,) if pos1 is None, else scalar

        Raises
        ------
        TypeError
            if pos1 and pos2 are not int or None
        """
        if isinstance(pos1, int) and isinstance(pos2, int):
            return np.sum(self.U[pos1, :] * self.U[pos2, :].conj())
        elif isinstance(pos2, int) or pos2 is None:
            if pos2 is None:
                pos2 = self.L//2 - 1 
            if pos1 is None:
                return np.sum(self.U * self.U[pos2,:].conj(), axis=1)
            else:
                pos1 = np.array(pos1, dtype=int)
                return np.sum(self.U[pos1, :] * self.U[pos2, :].conj(), axis=1)
        else:
            raise TypeError("pos1 and pos2 must be int or None")
    
    def correlation_matrix(self):
        return self.U @ self.U.conj().T
    
    def _reduced_cormat(self, pos:list[int]):
        return self.U[pos, :] @ self.U[pos, :].conj().T

    def entanglement(self, pos:list=None):
        r""" the entanglement entropy of the system

        .. math:
            S = -\sum_i (\lambda_i \log(\lambda_i) + (1-\lambda_i) \log(1-\lambda_i))
        
        if pos is None, return the entanglement entropy of the whole system,
        else return the entanglement entropy of the subsystem pos.

        Parameters
        ----------
        pos : int, optional
            positions, by default None

        Returns
        -------
        np.ndarray | float
            the entanglement entropy, shape (L-2,) if pos is None, else scalar
        """
        from ....measure import entropy

        if pos is None:
            C = self.correlation_matrix()
            res = np.zeros(self.L-1, dtype=float)
            for i in range(1, self.L):
                lambda_s = np.linalg.eigvalsh(C[:i, :i])
                res[i-1] = entropy(lambda_s) + entropy(1 - lambda_s)
            return res

        CA = self._reduced_cormat(pos)
        lambda_s = np.linalg.eigvalsh(CA)
        return entropy(lambda_s) + entropy(1 - lambda_s)
    
    def _tovector(self, cs:list[np.ndarray] = None):
        r"""验证程序，通过定义重构态向量，没有效率上的优势

        .. math:
            |ψ(t)\rangle = \prod_{i=1}^N (\sum_{j=1}^L U_{ji} c_j^\dagger) |0\rangle
        
        Notes
        -----
        如果 single_particle_ham() 返回的是一个 tuple,
        说明包含一个常数项
        那么严格来说，vector 将会相差一个相位因子 `exp(-i*coef_I*t)`
        但是这个相位因子不会影响到物理量的计算
        
        """
        L, M = self.L, self.M
        if cs is None:
            from ...basis import spin_basis
            from ...operas import SpinOper
            basis = spin_basis(L)
            cs = [SpinOper({  # jordan-wigner string
                'Z'*i + 'm': ([list(range(i+1))], [1.])
                }).to_matrix(basis,sparse=True) 
                  for i in range(L)]

        psi = np.zeros((2**L, 1), dtype=np.complex128)
        psi[-1] = 1.0 # vacuum state

        for n in range(M-1,-1,-1):
            psi = sum(
                self.U[i,n] * (cs[i].conj().T @ psi) 
                for i in range(L) if abs(self.U[i,n]) > 1e-12)
   
        return psi
    
    @classmethod
    def cor2cov(cls, C:np.ndarray):
        r""" convert correlation matrix to covariance matrix
        
        .. math::
            H = \log (C^{-1} - 1)
        
        """
        val, vec = np.linalg.eigh(C)
        epsilon_k = []
        for v in val:
            if abs(v) < 1e-15:
                epsilon_k.append(100)
            elif 1 - v < 1e-15:
                epsilon_k.append(-100)
            else:
                epsilon_k.append(np.log(1/v - 1))
        epsilon_k = np.real_if_close(epsilon_k)
        return (vec * epsilon_k) @ vec.conj().T
        
    def _todensirtmatrix(self, cs:list[np.ndarray] = None):
        r"""验证程序，没有效率上的优势

        .. math::
            \rho = \exp(-H) / Z
            
        .. math::
            H = \sum_{i,j} H_{ij} c_i^\dagger c_j
        
        下面的程序可以验证两种方法得到的态是一样的
        >>> L = 4
        >>> U = np.random.randn(L, 2)
        >>> ss = SlaterState(U)
        >>> ss.orthogonalize()
        >>> psi = ss._2fermionstate()
        >>> rho = ss._2densirtmatrix()
        >>> print(np.linalg.norm(
        ...     rho - np.real_if_close(psi @ psi.conj().T)
        ... ))
        """
        L = self.L
        if cs is None:
            from ...basis import spin_basis
            from ...operas import SpinOper
            basis = spin_basis(L)
            cs = [SpinOper({  # jordan-wigner string
                'Z'*i + 'm': ([list(range(i+1))], [1.])
                }).to_matrix(basis,sparse=True) 
                  for i in range(L)]
        H = SlaterState.cor2cov(self.correlation_matrix())
        Hmscr = sum(
            H[i,j] * (cs[i].conj().T @ cs[j]) 
            for i in range(L) for j in range(L) 
            if abs(H[i,j]) > 1e-12)
        rho = expm(-Hmscr.toarray())
        rho /= np.trace(rho)
        return rho

    def reduced_density_matrix(self, pos:list[int], cs:list[np.ndarray] = None):
        r"""the reduced density matrix of the subsystem pos

        .. math::
            \rho_A = \exp(-H) / Z
        
        Note that, this is also a Gaussian state, but not a Slater state.

        Parameters
        ----------
        pos : list[int]
            positions of the subsystem, e.g. [0, 1, 2]
        cs : list[np.ndarray], optional
            the fermion operators, by default None

        Returns
        -------
        np.ndarray
            the reduced density matrix, shape (2**len(pos), 2**len(pos))
        """
        Lsub = len(pos)
        if cs is None:
            from ...basis import spin_basis
            from ...operas import SpinOper
            basis = spin_basis(Lsub)
            cs = [SpinOper({  # jordan-wigner string
                'Z'*i + 'm': ([list(range(i+1))], [1.])
                }).to_matrix(basis,pauli=False,sparse=True) 
                  for i in range(Lsub)]
        CA = self._reduced_cormat(pos)
        H = SlaterState.cor2cov(CA)
        Hmscr = sum(
            H[i,j] * (cs[i].conj().T @ cs[j]) 
            for i in range(Lsub) for j in range(Lsub))
        rho = expm(-Hmscr.toarray())
        rho /= np.trace(rho)
        return rho

        



