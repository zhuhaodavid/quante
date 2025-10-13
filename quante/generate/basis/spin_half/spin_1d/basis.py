# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-28 13:00:19
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-13 19:58:13

from ...basis_class import SpinHalfBasis
import numpy as np

class SpinHalf1DBasis(SpinHalfBasis):
    def __init__(self, L: int) -> None:
        super().__init__(L)


############################################
# noblock
############################################
class SpinHalfBasisNoBlock(SpinHalf1DBasis):
    def __init__(self, L: int) -> None:
        """
        参数:
        - L (int): 系统的大小。
        """
        super().__init__(L)
        self.s_list = range(1 << L)
        self.default_complex: bool = False
        self._maps_dict = {}
        self._pcon_args = {'N': L}
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        if self._isdiag(opnm, posn):
            from .matrix_core import diag_matrix_element_noblock
            return None, None, diag_matrix_element_noblock(opnm, posn, coef, self.L, self.Ns, ME_init.dtype)
        from .matrix_core import single_sparse_matrix_element_noblock
        return single_sparse_matrix_element_noblock(opnm, posn, coef, self.L, self.Ns, row_init, col_init, ME_init)
        
    def _heimat(self, jxy, jz, cyclic=False):
        from .matrix_core import heisenberg_matrix_element_noblock
        return heisenberg_matrix_element_noblock(self.L, jxy=jxy, jz=jz, cyclic=cyclic)
    
    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        vec = np.zeros(1 << self.L, dtype=np.float64)
        vec[index] = 1.
        return vec
    
    def projection_matrix(self):
        return np.eye(1 << self.L)
    
    def project(self, state):
        return state
    
    def recover(self, state):
        return state
    
    @classmethod
    def print_dims(cls, L:int):
        print(f"dim = {1 << L}")

############################################
# Nup
############################################
class SpinHalfBasisNup(SpinHalf1DBasis):
    def __init__(self, L: int, Nup: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L 的整数。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        from .basis_core import construct_Nup_basis
        self.Ns, self.s_list = construct_Nup_basis(self.L, self.Nup)
        self.default_complex = False
        self._maps_dict = {}
        self._pcon_args = {'N': L, 'Nup': Nup}

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 0 <= self.Nup <= self.L and isinstance(self.Nup, int), "Nup should be an integer between 0 and N"
        
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        if self._isdiag(opnm, posn):
            from .matrix_core import diag_matrix_element_Nup
            return None, None, diag_matrix_element_Nup(opnm, posn, coef, self.L, self.Ns, self.s_list, ME_init.dtype)
        from .matrix_core import single_sparse_matrix_element_Nup
        return single_sparse_matrix_element_Nup(opnm, posn, coef, self.L, self.Ns, self.s_list, row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        from .matrix_core import heisenberg_matrix_element_Nup
        return heisenberg_matrix_element_Nup(self.L, self.Ns, jxy=jxy, jz=jz, s_list=self.s_list, cyclic=cyclic)

    def __getitem__(self, index):
        return self.to_full_space(index)
    
    def to_full_space(self, index:int, from_Nup_space:bool = False):
        assert not from_Nup_space, "does not support from_Nup_space"
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        vec = np.zeros(1 << self.L, dtype=np.float64)
        vec[self.s_list[index]] = 1.
        return vec
    
    def projection_matrix(self):
        proj = np.eye(self.Ns, dtype=np.float64)
        from .basis_core import convert_project_Nup_to_full_space
        return convert_project_Nup_to_full_space(proj, self.L, self.s_list)
    
    def project(self, state):
        assert state.shape[0] == 1 << self.L, "state should be a vector of length 2**L"
        from .matrix_core import project_Nup
        return project_Nup(state, self.Ns, self.s_list)
    
    def recover(self, state: np.ndarray) -> np.ndarray:
        dim1, dim2 = state.shape
        assert dim1 == self.Ns, f"state should be a matrix of shape ({self.Ns}, N)"
        vec = np.zeros((1 << self.L, dim2), dtype=state.dtype)
        vec[self.s_list, :] = state
        return vec
    
    @classmethod
    def print_dims(cls, L:int):
        import math
        for Nup in range(L+1):
            print(f"Nup = {Nup}: {math.comb(L, Nup)}")


############################################
# pblock
##############################################
class SpinHalfBasisPBlock(SpinHalf1DBasis):
    def __init__(self, L: int, pblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - pblock (Optional[int]): 反演对称性块。-1 或 1。
        """
        super().__init__(L)
        self.pblock = pblock
        self._validate_pblock()
        from .basis_core import construct_pblock_basis
        self.Ns, self.s_list = construct_pblock_basis(self.L, self.pblock)
        self.default_complex = False
        self._maps_dict = {'p': (np.arange(L-1, -1, -1), pblock)}
        self._pcon_args = {'N': L}

    def _validate_pblock(self) -> None:
        assert self.pblock in [-1, 1], "pblock should be -1 or 1"
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_pblock
        return single_sparse_matrix_element_pblock(opnm, posn, coef, self.L, self.pblock, self.Ns, self.s_list, row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        from .matrix_core import heisenberg_matrix_element_pblock
        return heisenberg_matrix_element_pblock(self.L, self.Ns, self.pblock, jxy=jxy, jz=jz, s_list=self.s_list, cyclic=cyclic)

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int, from_Nup_space:bool = False):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        assert not from_Nup_space, "pblock basis does not support from_Nup_space"
        from .basis_core import recover_pblock
        state = np.zeros((self.Ns,1), dtype=np.float64)
        state[index, 0] = 1
        return recover_pblock(state, self.L, self.pblock, self.s_list, state.dtype)

    def projection_matrix(self, from_Nup_space:bool = False):
        assert not from_Nup_space, "pblock basis does not support from_Nup_space"
        from .basis_core import recover_pblock
        state = np.eye(self.Ns, dtype=np.float64)
        return recover_pblock(state, self.L, self.pblock, self.s_list, state.dtype)

    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_pblock
        return recover_pblock(state, self.L, self.pblock, self.s_list, state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_pblock_basis
        for p in [-1, 1]:
            Ns, _ = construct_pblock_basis(L, p)
            print(f"p = {p:>3}: {Ns:>3}")

################################################
# zblock
################################################
class SpinHalfBasisZBlock(SpinHalf1DBasis):
    def __init__(self, L: int, zblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L 的整数。
        - zblock (Optional[int]): 旋转对称性块。-1 或 1。
        """
        super().__init__(L)
        self.zblock = zblock
        self._validate_zblock()
        from .basis_core import construct_zblock_basis
        self.Ns, self.s_list = construct_zblock_basis(self.L, self.zblock)
        self.default_complex = False
        self._maps_dict = {'z': (-(np.arange(L) + 1), zblock)}
        self._pcon_args = {'N': L}

    def _validate_zblock(self) -> None:
        assert self.zblock in [-1, 1], "zblock should be -1 or 1"
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_zblock
        return single_sparse_matrix_element_zblock(opnm, posn, coef, self.L, self.zblock, self.Ns, self.s_list, row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        from .matrix_core import heisenberg_matrix_element_zblock
        return heisenberg_matrix_element_zblock(self.L, self.Ns, self.zblock, jxy=jxy, jz=jz, s_list=self.s_list, cyclic=cyclic)

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        from .basis_core import recover_zblock
        state = np.zeros((self.Ns,1), dtype=np.float64)
        state[index, 0] = 1
        return recover_zblock(state, self.L, self.zblock, self.s_list, state.dtype)

    def projection_matrix(self):
        from .basis_core import recover_zblock
        state = np.eye(self.Ns, dtype=np.float64)
        return recover_zblock(state, self.L, self.zblock, self.s_list, state.dtype)
        # from .zblock_basis import projective
        # return projective(self.s_list, self.Ns, self.L, self.zblock)

    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_zblock
        return recover_zblock(state, self.L, self.zblock, self.s_list, state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_zblock_basis
        for z in [-1, 1]:
            Ns, _ = construct_zblock_basis(L, z)
            print(f"z = {z:>3}: {Ns:>3}")

########################################
# pzblock
########################################

class SpinHalfBasisPZBlock(SpinHalf1DBasis):
    def __init__(self, L: int, pzblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L 的整数。
        - pzblock (Optional[int]): 旋转对称性块。-1 或 1。
        """
        super().__init__(L)
        self.pzblock = pzblock
        self._validate_pzblock()
        from .basis_core import construct_pzblock_basis
        self.Ns, self.s_list = construct_pzblock_basis(self.L, self.pzblock)
        self.default_complex = False
        self._maps_dict = {'pz': (-(np.arange(L-1,-1,-1) + 1), pzblock)}
        self._pcon_args = {'N': L}

    def _validate_pzblock(self) -> None:
        assert self.pzblock in [-1, 1], "pzblock should be -1 or 1"
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_pzblock
        return single_sparse_matrix_element_pzblock(opnm, posn, coef, self.L, self.pzblock, self.Ns, self.s_list, row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        from .matrix_core import heisenberg_matrix_element_pzblock
        return heisenberg_matrix_element_pzblock(self.L, self.Ns, self.pzblock, jxy=jxy, jz=jz, s_list=self.s_list, cyclic=cyclic)

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        from .basis_core import recover_pzblock
        state = np.zeros((self.Ns,1), dtype=np.float64)
        state[index, 0] = 1
        return recover_pzblock(state, self.L, self.pzblock, self.s_list, state.dtype)

    def projection_matrix(self):
        from .basis_core import recover_pzblock
        state = np.eye(self.Ns, dtype=np.float64)
        return recover_pzblock(state, self.L, self.pzblock, self.s_list, state.dtype)

    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_pzblock
        return recover_pzblock(state, self.L, self.pzblock, self.s_list, state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_pzblock_basis
        for pz in [-1, 1]:
            Ns, _ = construct_pzblock_basis(L, pz)
            print(f"pz = {pz:>3}: {Ns:>3}")



############################################
# kblock
############################################
class SpinHalfBasisKBlock(SpinHalf1DBasis):
    def __init__(self, L: int, kblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L 的整数。
        - kblock (Optional[int]): 动量块。从 0 到 L-1 的整数。
        """
        super().__init__(L)
        self.kblock = kblock
        self._validate_kblock()
        from .basis_core import construct_kblock_basis
        self.Ns, self.s_list, R_list = construct_kblock_basis(self.L, self.kblock)
        self.other_params["R_list"] = R_list
        self._maps_dict = {'k': ((np.arange(L) + 1) % L, kblock)}
        self._pcon_args = {'N': L}

    def _validate_kblock(self) -> None:
        assert self.kblock is not None and 0 <= self.kblock <= self.L - 1 and isinstance(self.kblock, int), f"kblock should be an integer between 0 and {self.L - 1}"
        
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_kblock
        return single_sparse_matrix_element_kblock(opnm, posn, coef, self.L, self.kblock, self.Ns, self.s_list, self.other_params["R_list"], row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        assert cyclic, "Only cyclic boundary condition is supported for kblock basis."
        from .matrix_core import heisenberg_matrix_element_kblock
        return heisenberg_matrix_element_kblock(self.L, self.Ns, self.kblock, jxy=jxy, jz=jz, s_list=self.s_list, R_list=self.other_params["R_list"])

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        vec = np.zeros((self.Ns,1), dtype=np.complex128)
        vec[index, 0] = 1
        from .basis_core import recover_kblock
        return recover_kblock(vec, self.L, self.kblock, self.s_list, self.other_params["R_list"])
    
    def projection_matrix(self):
        from .basis_core import recover_kblock
        return recover_kblock(np.eye(self.Ns, dtype=np.complex128), self.L, self.kblock, self.s_list, self.other_params["R_list"])
    
    def project(self, state):
        assert state.shape[0] == 1 << self.L, "state should be a vector of length 2**L"
        from .basis_core import project
        return project(state, self.L, self.kblock, self.Ns, self.s_list, self.other_params["R_list"])
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_kblock
        return recover_kblock(np.complex128(state), self.L, self.kblock, self.s_list, self.other_params["R_list"])
    
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_kblock_basis
        for k in range(L):
            Ns, _, _ = construct_kblock_basis(L, k)
            print(f"k = {k}: {Ns}")



############################################
# kpblock, pblock
############################################
class SpinHalfBasisKPBlock(SpinHalf1DBasis):
    def __init__(self, L: int, kblock: int, pblock: int) -> None:
        """
        参数：
        - N (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 N 的整数。
        - kblock (Optional[int]): 动量块。从 0 到 N//2 的整数。
        - pblock (Optional[int]): 反演对称性块。-1 或 1。
        """
        super().__init__(L)
        self.kblock = kblock
        self._validate_kblock()
        self.pblock = pblock
        self._validate_pblock()
        from .basis_core import construct_kblock_pblock_basis
        self.Ns, self.s_list, R_list, m_list = construct_kblock_pblock_basis(self.L, self.kblock, self.pblock)
        self.other_params["R_list"] = R_list
        self.other_params["m_list"] = m_list
        self._double_Ns = 4
        self.default_complex = False
        self._maps_dict = {'k': ((np.arange(L) + 1) % L, kblock), 'p': (np.arange(L-1, -1, -1), pblock)}
        self._pcon_args = {'N': L}

    def _validate_kblock(self) -> None:
        assert self.kblock is not None and 0 <= self.kblock <= self.L // 2 and isinstance(self.kblock, int), f"kblock should be an integer between 0 and {self.L // 2} when using kblock and pblock simutaniuously"

    def _validate_pblock(self) -> None:
        assert self.pblock in [-1, 1], "pblock should be -1 or 1"

    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_kblock_pblock
        return single_sparse_matrix_element_kblock_pblock(opnm, posn, coef, self.L, self.kblock, self.pblock, self.Ns, self.s_list, self.other_params["R_list"], self.other_params["m_list"], row_init, col_init, ME_init)

############################################
# Nup pblock
############################################
class SpinHalfBasisNupPBlock(SpinHalf1DBasis):
    def __init__(self, L: int, Nup: int, pblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L 的整数。
        - pblock (Optional[int]): 反演对称性块。-1 或 1。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        self.pblock = pblock
        self._validate_pblock()
        from .basis_core import construct_Nup_pblock_basis
        self.Ns, self.s_list = construct_Nup_pblock_basis(self.L, self.Nup, self.pblock)
        self.default_complex = False
        self._maps_dict = {'p': (np.arange(L-1, -1, -1), pblock)}
        self._pcon_args = {'N': L, 'Nup': Nup}

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 0 <= self.Nup <= self.L and isinstance(self.Nup, int), "Nup should be an integer between 0 and N"

    def _validate_pblock(self) -> None:
        assert self.pblock in [-1, 1], "pblock should be -1 or 1"
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_pblock
        return single_sparse_matrix_element_pblock(opnm, posn, coef, self.L, self.pblock, self.Ns, self.s_list, row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        from .matrix_core import heisenberg_matrix_element_pblock
        return heisenberg_matrix_element_pblock(self.L, self.Ns, self.pblock, jxy=jxy, jz=jz, s_list=self.s_list, cyclic=cyclic)

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        from .basis_core import recover_Nup_pblock
        state = np.zeros((self.Ns,1), dtype=np.float64)
        state[index, 0] = 1
        return recover_Nup_pblock(state, self.L, self.pblock, self.s_list, state.dtype)

    def projection_matrix(self):
        from .basis_core import recover_Nup_pblock
        state = np.eye(self.Ns, dtype=np.float64)
        return recover_Nup_pblock(state, self.L, self.pblock, self.s_list, state.dtype)

    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_Nup_pblock
        return recover_Nup_pblock(state, self.L, self.pblock, self.s_list, state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_Nup_pblock_basis
        for Nup in range(L+1):
            for p in [-1, 1]:
                Ns, _ = construct_Nup_pblock_basis(L, Nup, p)
                print(f"Nup = {Nup:>3}, p = {p:>3}: {Ns:>3}")

############################################
# Nup pzblock
############################################
class SpinHalfBasisNupPZBlock(SpinHalf1DBasis):
    def __init__(self, L: int, Nup: int, pzblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L//2 的整数。
        - pzblock (Optional[int]): 旋转对称性块。-1 或 1。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        self.pzblock = pzblock
        self._validate_pzblock()
        from .basis_core import construct_Nup_pzblock_basis
        self.Ns, self.s_list = construct_Nup_pzblock_basis(self.L, self.Nup, self.pzblock)
        self._maps_dict = {'pz': (-(np.arange(L-1, -1, -1)+1), pzblock)}
        self._pcon_args = {'N': L, 'Nup': Nup}

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 0 <= self.Nup <= self.L and isinstance(self.Nup, int), "Nup should be an integer between 0 and N"

    def _validate_pzblock(self) -> None:
        assert self.pzblock in [-1, 1], "pzblock should be -1 or 1"
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_pzblock
        return single_sparse_matrix_element_pzblock(opnm, posn, coef, self.L, self.pzblock, self.Ns, self.s_list, row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        from .matrix_core import heisenberg_matrix_element_pzblock
        return heisenberg_matrix_element_pzblock(self.L, self.Ns, self.pzblock, jxy=jxy, jz=jz, s_list=self.s_list, cyclic=cyclic)

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        from .basis_core import recover_Nup_pzblock
        state = np.zeros((self.Ns,1), dtype=np.float64)
        state[index, 0] = 1
        return recover_Nup_pzblock(state, self.L, self.pzblock, self.s_list, state.dtype)

    def projection_matrix(self):
        from .basis_core import recover_Nup_pzblock
        state = np.eye(self.Ns, dtype=np.float64)
        return recover_Nup_pzblock(state, self.L, self.pzblock, self.s_list, state.dtype)

    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_Nup_pzblock
        return recover_Nup_pzblock(state, self.L, self.pzblock, self.s_list, state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_Nup_pzblock_basis
        for Nup in range(L//2+1):
            for pz in [-1, 1]:
                Ns, _ = construct_Nup_pzblock_basis(L, Nup, pz)
                print(f"Nup = {Nup:>3}, pz = {pz:>3}: {Ns:>3}")

############################################
# Nup zblock
############################################
class SpinHalfBasisNupZBlock(SpinHalf1DBasis):
    def __init__(self, L: int, Nup: int, zblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L//2 的整数。
        - zblock (Optional[int]): 旋转对称性块。-1 或 1。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        self.zblock = zblock
        self._validate_zblock()
        from .basis_core import construct_Nup_zblock_basis
        self.Ns, self.s_list = construct_Nup_zblock_basis(self.L, self.Nup, self.zblock)
        self.default_complex: bool = False
        self._maps_dict = {'z': (-(np.arange(L) + 1), zblock)}
        self._pcon_args = {'N': L, 'Nup': Nup}

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and self.Nup <= self.L and isinstance(self.Nup, int), "Nup should be an integer between 0 and N"

    def _validate_zblock(self) -> None:
        assert self.zblock in [-1, 1], "zblock should be -1 or 1"
    
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_zblock
        return single_sparse_matrix_element_zblock(opnm, posn, coef, self.L, self.zblock, self.Ns, self.s_list, row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        from .matrix_core import heisenberg_matrix_element_zblock
        return heisenberg_matrix_element_zblock(self.L, self.Ns, self.zblock, jxy=jxy, jz=jz, s_list=self.s_list, cyclic=cyclic)

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        from .basis_core import recover_Nup_zblock
        state = np.zeros((self.Ns,1), dtype=np.float64)
        state[index, 0] = 1
        return recover_Nup_zblock(state, self.L, self.zblock, self.s_list, state.dtype)

    def projection_matrix(self):
        from .basis_core import recover_Nup_zblock
        state = np.eye(self.Ns, dtype=np.float64)
        return recover_Nup_zblock(state, self.L, self.zblock, self.s_list, state.dtype)

    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_Nup_zblock
        return recover_Nup_zblock(state, self.L, self.zblock, self.s_list, state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_Nup_zblock_basis
        for Nup in range(L//2+1):
            for z in [-1, 1]:
                Ns, _ = construct_Nup_zblock_basis(L, Nup, z)
                print(f"Nup = {Nup:>3}, z = {z:>3}: {Ns:>3}")


##############################################
# Nup kblock
##############################################
class SpinHalfBasisNupKBlock(SpinHalf1DBasis):
    def __init__(self, L: int, Nup: int, kblock: int) -> None:
        """
        参数：
        - L (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 L 的整数。
        - kblock (Optional[int]): 动量块。从 0 到 L-1 的整数。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        self.kblock = kblock
        self._validate_kblock()
        from .basis_core import construct_Nup_kblock_basis
        self.Ns, self.s_list, R_list = construct_Nup_kblock_basis(self.L, self.Nup, self.kblock)
        self.other_params["R_list"] = R_list
        self._maps_dict = {'k': ((np.arange(L) + 1) % L, kblock)}
        self._pcon_args = {'N': L, 'Nup': Nup}

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 0 <= self.Nup <= self.L and isinstance(self.Nup, int), "Nup should be an integer between 0 and N"
    
    def _validate_kblock(self) -> None:
        assert self.kblock is not None and 0 <= self.kblock <= self.L - 1 and isinstance(self.kblock, int), f"kblock should be an integer between 0 and {self.L - 1}"
        
    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_kblock
        return single_sparse_matrix_element_kblock(opnm, posn, coef, self.L, self.kblock, self.Ns, self.s_list, self.other_params["R_list"], row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        assert cyclic, "Only cyclic boundary condition is supported for kblock basis."
        from .matrix_core import heisenberg_matrix_element_kblock
        return heisenberg_matrix_element_kblock(self.L, self.Ns, self.kblock, jxy=jxy, jz=jz, s_list=self.s_list, R_list=self.other_params["R_list"])

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        assert isinstance(index, int) and 0 <= index < self.Ns, f"index should be an integer between 0 and {self.Ns - 1}"
        from .basis_core import recover_Nup_kblock
        state = np.zeros((self.Ns, 1), dtype=np.complex128)
        state[index, 0] = 1
        return recover_Nup_kblock(state, self.L, self.kblock, self.s_list, self.other_params["R_list"])

    def projection_matrix(self):
        from .basis_core import recover_Nup_kblock
        state = np.eye(self.Ns, dtype=np.complex128)
        return recover_Nup_kblock(state, self.L, self.kblock, self.s_list, self.other_params["R_list"])
    
    def project(self, state):
        assert state.shape[0] == 1 << self.L, "state should be a vector of length 2**L"
        from .matrix_core import project_kblock
        return project_kblock(state, self.L, self.kblock, self.Ns, self.s_list, self.other_params["R_list"])
    
    def recover(self, state):
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_Nup_kblock
        return recover_Nup_kblock(np.complex128(state), self.L, self.kblock, self.s_list, self.other_params["R_list"])
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_Nup_kblock_basis
        for Nup in range(L+1):
            for k in range(L):
                Ns, _, _ = construct_Nup_kblock_basis(L, Nup, k)
                print(f"Nup = {Nup}, k = {k}: {Ns}")


############################################
# Nup_kblock_pblock
############################################
class SpinHalfBasisNupKPBlock(SpinHalf1DBasis):
    def __init__(self, L: int, Nup: int, kblock: int, pblock: int) -> None:
        """
        参数：
        - N (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。从 0 到 N 的整数。
        - kblock (Optional[int]): 动量块。从 0 到 N//2 的整数。
        - pblock (Optional[int]): 反演对称性块。-1 或 1。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        self.kblock = kblock
        self._validate_kblock()
        self.pblock = pblock
        self._validate_pblock()
        from .basis_core import construct_Nup_kblock_pblock_basis
        self.Ns, self.s_list, R_list, m_list = construct_Nup_kblock_pblock_basis(self.L, self.Nup, self.kblock, self.pblock)
        self.other_params["R_list"] = R_list
        self.other_params["m_list"] = m_list
        self._double_Ns = 4
        self.default_complex = False
        self._maps_dict = {'k': ((np.arange(L) + 1) % L, kblock), 'p': (np.arange(L-1, -1, -1), pblock)}
        self._pcon_args = {'N': L, 'Nup': Nup}

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 0 <= self.Nup <= self.L and isinstance(self.Nup, int), "Nup should be an integer between 0 and N"

    def _validate_kblock(self) -> None:
        assert self.kblock is not None and 0 <= self.kblock <= self.L // 2 and isinstance(self.kblock, int), f"kblock should be an integer between 0 and {self.L // 2} when using kblock and pblock simutaniuously"

    def _validate_pblock(self) -> None:
        assert self.pblock in [-1, 1], "pblock should be -1 or 1"

    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_Nup_kblock_pblock
        return single_sparse_matrix_element_Nup_kblock_pblock(opnm, posn, coef, self.L, self.kblock, self.pblock, self.Ns, self.s_list, self.other_params["R_list"], self.other_params["m_list"], row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        assert cyclic, "Only cyclic boundary condition is supported for kblock basis."
        from .matrix_core import heisenberg_matrix_element_Nup_kblock_pblock
        return heisenberg_matrix_element_Nup_kblock_pblock(self.L, self.Ns, self.kblock, self.pblock, jxy=jxy, jz=jz, s_list=self.s_list, R_list=self.other_params["R_list"], m_list=self.other_params["m_list"])

    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        from .basis_core import recover_Nup_kblock_pblock
        state = np.zeros((self.Ns,1), dtype=np.float64)
        state[index, 0] = 1
        return recover_Nup_kblock_pblock(state, self.L, self.kblock, self.pblock, self.s_list, self.other_params["R_list"], self.other_params["m_list"], state.dtype)

    def projection_matrix(self):
        from .basis_core import recover_Nup_kblock_pblock
        state = np.eye(self.Ns, dtype=np.float64)
        return recover_Nup_kblock_pblock(state, self.L, self.kblock, self.pblock, self.s_list, self.other_params["R_list"], self.other_params["m_list"], state.dtype)

    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        # return self.projection_matrix() @ state
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_Nup_kblock_pblock
        return recover_Nup_kblock_pblock(state, self.L, self.kblock, self.pblock, self.s_list, self.other_params["R_list"], self.other_params["m_list"], state.dtype)
    
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_Nup_kblock_pblock_basis
        for Nup in range(L+1):
            for k in range(L//2+1):
                for p in [-1,1]:
                    Ns, _, _, _ = construct_Nup_kblock_pblock_basis(L, Nup, k, p)
                    print(f"Nup = {Nup:<4}, k = {k:<4}, p = {p:<4}:  {Ns:<4}")

############################################
# Nup_kblock_pblock_zblock
############################################
class SpinHalfBasisNupKPZBlock(SpinHalf1DBasis):
    def __init__(self, L: int, Nup: int , kblock: int, pblock: int, zblock: int) -> None:
        """
        参数：
        - N (int): 系统的大小。
        - Nup (Optional[int]): 自旋向上的粒子数。必须取 N//2。
        - kblock (Optional[int]): 动量块。从 0 到 N//2 的整数。
        - pblock (Optional[int]): 反演对称性块。-1 或 1。
        - zblock (Optional[int]): 旋转对称性块。-1 或 1。
        """
        super().__init__(L)
        self.Nup = Nup
        self._validate_Nup()
        self.kblock = kblock
        self._validate_kblock()
        self.pblock = pblock
        self._validate_pblock()
        self.zblock = zblock
        self._validate_zblock()
        from .basis_core import construct_Nup_kblock_pblock_zblock_basis
        self.Ns, self.s_list, R_list, m_list, c_list = construct_Nup_kblock_pblock_zblock_basis(self.L, self.kblock, self.pblock, self.zblock)
        self.other_params["R_list"] = R_list
        self.other_params["m_list"] = m_list
        self.other_params["c_list"] = c_list
        self.default_complex = False
        self._double_Ns = 4
        self._maps_dict = {
            'k': ((np.arange(L) + 1) % L, kblock), 
            'p': (np.arange(L-1, -1, -1), pblock), 
            'z': (-(np.arange(L) + 1), zblock)
        }
        self._pcon_args = {'N': L, 'Nup': Nup}

    def _validate_Nup(self) -> None:
        assert self.Nup is not None and 2*self.Nup == self.L and isinstance(self.Nup, int), "Nup must be N//2 when using Nup and kblock pblock zblock simutaniuously"

    def _validate_kblock(self) -> None:
        assert self.kblock is not None and 0 <= self.kblock <= self.L // 2 and isinstance(self.kblock, int), f"kblock should be an integer between 0 and {self.L // 2} when using kblock and pblock simutaniuously"

    def _validate_pblock(self) -> None:
        assert self.pblock in [-1, 1], "pblock should be -1 or 1"
    
    def _validate_zblock(self) -> None:
        assert self.zblock in [-1, 1], "zblock should be -1 or 1"

    def _Op(self, opnm, posn, coef, row_init, col_init, ME_init):
        from .matrix_core import single_sparse_matrix_element_Nup_kblock_pblock_zblock
        return single_sparse_matrix_element_Nup_kblock_pblock_zblock(opnm, posn, coef, self.L, self.kblock, self.pblock, self.zblock, self.Ns, self.s_list, self.other_params["R_list"], self.other_params["m_list"], self.other_params["c_list"], row_init, col_init, ME_init)
    
    def _heimat(self, jxy, jz, cyclic=True):
        assert cyclic, "Only cyclic boundary condition is supported for kblock basis."
        from .matrix_core import heisenberg_matrix_element_Nup_kblock_pblock_zblock
        return heisenberg_matrix_element_Nup_kblock_pblock_zblock(self.L, self.Ns, self.kblock, self.pblock, self.zblock, jxy=jxy, jz=jz, s_list=self.s_list, R_list=self.other_params["R_list"], m_list=self.other_params["m_list"], c_list=self.other_params["c_list"])
 
    def __getitem__(self, index):
        return self.to_full_space(index)
        
    def to_full_space(self, index:int):
        from .basis_core import recover_Nup_kblock_pblock_zblock
        state = np.zeros((self.Ns, 1), dtype=np.float64)
        state[index, 0] = 1.
        return recover_Nup_kblock_pblock_zblock(state, self.L, self.kblock, self.pblock, self.zblock, self.s_list, self.other_params["R_list"], self.other_params["m_list"], self.other_params["c_list"], state.dtype)

    def projection_matrix(self):
        from .basis_core import recover_Nup_kblock_pblock_zblock
        state = np.eye(self.Ns, dtype=np.float64)
        return recover_Nup_kblock_pblock_zblock(state, self.L, self.kblock, self.pblock, self.zblock, self.s_list, self.other_params["R_list"], self.other_params["m_list"], self.other_params["c_list"], state.dtype)
    
    def project(self, state):
        return self.projection_matrix().conj().T @ state
    
    def recover(self, state):
        # return self.projection_matrix() @ state
        assert state.shape[0] == self.Ns, f"state should be a vector of length {self.Ns}"
        from .basis_core import recover_Nup_kblock_pblock_zblock
        return recover_Nup_kblock_pblock_zblock(state, self.L, self.kblock, self.pblock, self.zblock, self.s_list, self.other_params["R_list"], self.other_params["m_list"], self.other_params["c_list"], state.dtype)
    
    @classmethod
    def print_dims(cls, L:int):
        from .basis_core import construct_Nup_kblock_pblock_zblock_basis
        for k in range(L//2+1):
            for p in [-1,1]:
                for z in [-1,1]:
                    Ns, _, _, _, _ = construct_Nup_kblock_pblock_zblock_basis(L, k, p, z)
                    print(f"Nup = {L//2:<4}, k = {k:<4}, p = {p:<4}, z = {z:<4}:  {Ns:<4}")
