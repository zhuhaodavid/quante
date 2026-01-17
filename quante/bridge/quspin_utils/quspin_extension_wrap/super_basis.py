# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-10 17:12:13
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-13 23:44:22

import scipy.sparse as sp    
from quspin.basis.user import user_basis  # Hilbert space user basis

try:
    import dowhen

    # there is a bug in user_basis.__init__ when Np is given as a list
    # we need to skip the check as define n_sectors manually
    dowhen.when(
        user_basis.__init__, "for np in Np:"
    ).goto(
        "Ns = sum(get_Ns_pcon(N, np) for np in Np)"
    ).do(
        "n_sectors = len(Np)"
    )
except ImportError:
    raise ImportError("Please install dowhen package to use this feature: pip install dowhen")

from quspin.basis.user import (
    next_state_sig_32,
    op_sig_32,
    map_sig_32,
    count_particles_sig_32,
)  # user basis data types signatures
from numba import carray, cfunc  # numba helper functions
from numba import uint32, int32  # numba data types
import numpy as np
from scipy.special import comb
from warnings import warn


#
############   create spin-1/2 user basis object   #############
#
######  function to call when applying operators
@cfunc(
    op_sig_32,
    locals=dict(s=int32, n=int32, b=uint32),
)
def op_pauli(op_struct_ptr, op_str, site_ind, N, args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr, 1)[0]
    err = 0
    #
    site_ind = N - site_ind - 1  # convention for QuSpin for mapping from bits to sites.
    n = (op_struct.state >> site_ind) & 1  # either 0 or 1
    s = (((op_struct.state >> site_ind) & 1) << 1) - 1  # either -1 or 1
    b = 1 << site_ind
    #
    if op_str == 120:  # "x" is integer value 120 = ord("x")
        op_struct.state ^= b

    elif op_str == 121:  # "y" is integer value 120 = ord("y")
        op_struct.state ^= b
        op_struct.matrix_ele *= 1.0j * s

    elif op_str == 43:  # "+" is integer value 43 = ord("+")
        if n:
            op_struct.matrix_ele = 0
        else:
            op_struct.state ^= b  # create spin

    elif op_str == 45:  # "-" is integer value 45 = ord("-")
        if n:
            op_struct.state ^= b  # destroy spin
        else:
            op_struct.matrix_ele = 0

    elif op_str == 122:  # "z" is integer value 120 = ord("z")
        op_struct.matrix_ele *= s

    elif op_str == 110:  # "n" is integer value 110 = ord("n")
        op_struct.matrix_ele *= n

    elif op_str == 73:  # "I" is integer value 73 = ord("I")
        pass

    else:
        op_struct.matrix_ele = 0
        err = -1
    #
    return err


@cfunc(
    op_sig_32,
    locals=dict(s=int32, n=int32, b=uint32),
)
def op_spin(op_struct_ptr, op_str, site_ind, N, args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr, 1)[0]
    err = 0
    #
    site_ind = N - site_ind - 1  # convention for QuSpin for mapping from bits to sites.
    n = (op_struct.state >> site_ind) & 1  # either 0 or 1
    s = (((op_struct.state >> site_ind) & 1) << 1) - 1  # either -1 or 1
    b = 1 << site_ind
    #
    if op_str == 120:  # "x" is integer value 120 = ord("x")
        op_struct.state ^= b
        op_struct.matrix_ele *= 0.5

    elif op_str == 121:  # "y" is integer value 120 = ord("y")
        op_struct.state ^= b
        op_struct.matrix_ele *= 0.5j * s

    elif op_str == 43:  # "+" is integer value 43 = ord("+")
        if n:
            op_struct.matrix_ele = 0
        else:
            op_struct.state ^= b  # create spin

    elif op_str == 45:  # "-" is integer value 45 = ord("-")
        if n:
            op_struct.state ^= b  # destroy spin
        else:
            op_struct.matrix_ele = 0

    elif op_str == 122:  # "z" is integer value 120 = ord("z")
        op_struct.matrix_ele *= 0.5*s

    elif op_str == 110:  # "n" is integer value 110 = ord("n")
        op_struct.matrix_ele *= n

    elif op_str == 73:  # "I" is integer value 73 = ord("I")
        pass

    else:
        op_struct.matrix_ele = 0
        err = -1
    #
    return err

#
######  define symmetry maps
#
@cfunc(
    map_sig_32,
    locals=dict(
        shift=uint32,
        xmax=uint32,
        x1=uint32,
        x2=uint32,
        period=int32,
        l=int32,
    ),
)
def translation_x(x, N, sign_ptr, args):
    """works for all system sizes N."""
    shift = args[0]  # translate state by shift sites
    period = N//2  # periodicity/cyclicity of translation
    xmax = args[1]
    l = (shift + period) % period

    # 
    s1 = x >> (N//2)
    sx1 = s1 >> (period - l)
    sx2 = (s1 << l) & xmax
    s1 = sx1 | sx2
    # 
    s2 = x & ((1 << (N//2)) - 1)
    sx1 = s2 >> (period - l)
    sx2 = (s2 << l) & xmax
    s2 = sx1 | sx2
    #
    return (s1 << period) | s2

#
@cfunc(
    map_sig_32,
    locals=dict(
        out=uint32,
        s=int32,
    ),
)
def parity_x(x, N, sign_ptr, args):
    """works for all system sizes N."""
    s1 = x >> (N//2)
    #
    s = N//2-1
    out1 = 0
    out1 ^= s1 & 1
    s1 >>= 1
    while s1:
        out1 <<= 1
        out1 ^= s1 & 1
        s1 >>= 1
        s -= 1
    #
    out1 <<= s

    s2 = x & ((1 << (N//2)) - 1)
    #
    s = N//2-1
    out2 = 0
    out2 ^= s2 & 1
    s2 >>= 1
    while s2:
        out2 <<= 1
        out2 ^= s2 & 1
        s2 >>= 1
        s -= 1
    #
    out2 <<= s

    return (out1 << (N//2)) | out2

#
@cfunc(
    map_sig_32,
    locals=dict(
        out=uint32,
        s=int32,
    ),
)
def parity_y(x, N, sign_ptr, args):
    """works for all system sizes N."""
    s1 = x >> (N//2)
    s2 = x & ((1 << (N//2)) - 1)
    return (s2 << (N//2)) | s1

#
@cfunc(
    map_sig_32,
    locals=dict(
        xmax=uint32,
    ),
)
def spin_inversion(x, N, sign_ptr, args):
    """works for all system sizes N."""
    xmax = args[0]  # maximum integer
    return x ^ xmax

#
######  define function to count particles in bit representation
#
@cfunc(count_particles_sig_32, locals=dict(s_count=uint32))
def count_particles(x, p_number_ptr, args):
    """Counts number of particles/spin-ups in a state stored in integer representation for up to N=32 sites"""
    #
    s_count = x & ((0x7FFFFFFF) >> (31 - args[0]))
    s_count = s_count - ((s_count >> 1) & 0x55555555)
    s_count = (s_count & 0x33333333) + ((s_count >> 2) & 0x33333333)
    s_count = (((s_count + (s_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
    #
    p_number_ptr[0] = s_count


#
######  function to implement magnetization/particle conservation
#
@cfunc(
    next_state_sig_32,
    locals=dict(t=uint32),
)
def next_state_Np(s, counter, N, args):
    """implements magnetization conservation."""
    if s == 0:
        return s
    #
    t = (s | (s - 1)) + 1
    res = t | ((((t & (0 - t)) // (s & (0 - s))) >> 1) - 1)
    return res 


# python function to calculate the starting state to generate the particle conserving basis
def get_s0_pcon_Np(N, Np):
    return sum(1 << i for i in range(Np))


# python function to calculate the size of the particle-conserved basis,
# i.e. BEFORE applying pre_check_state and symmetry maps
def get_Ns_pcon_Np(N, Np):
    return comb(N, Np, exact=True)

@cfunc(
    next_state_sig_32,
    locals=dict(t=uint32),
)
def next_state_Nd(s, counter, N, args):
    """implements magnetization conservation."""
    s = s ^ ((1<<(N//2)) -1)
    if s == 0:
        return s
    #
    t = (s | (s - 1)) + 1
    res = t | ((((t & (0 - t)) // (s & (0 - s))) >> 1) - 1)
    return res ^ ((1<<(N//2)) -1)


# python function to calculate the starting state to generate the particle conserving basis
def get_s0_pcon_Nd(N, Np):
    return sum(1 << i for i in range(Np)) ^ ((1<<(N//2)) -1)


# python function to calculate the size of the particle-conserved basis,
# i.e. BEFORE applying pre_check_state and symmetry maps
def get_Ns_pcon_Nd(N, Np):
    return comb(N, Np, exact=True)


#
######  function to implement magnetization/particle conservation
#
@cfunc(
    next_state_sig_32,
    locals=dict(t=uint32),
)
def next_state_Np_Nd(s_full, counter, N, args):
    """implements magnetization conservation."""
    if s_full == 0:
        return s_full
    
    L = N//2
    s1, s2 = s_full >> L, s_full & ((1<<L)-1)

    if s2 == 0:
        t = (s1 | (s1 - 1)) + 1
        res1 = t | ((((t & (0 - t)) // (s1 & (0 - s1))) >> 1) - 1)
        return res1 << L
    
    if s1 == 0:
        t = (s2 | (s2 - 1)) + 1
        res2 = t | ((((t & (0 - t)) // (s2 & (0 - s2))) >> 1) - 1)
        return res2
    
    t = (s2 | (s2 - 1)) + 1
    res2 = t | ((((t & (0 - t)) // (s2 & (0 - s2))) >> 1) - 1)
    if res2 < (1<<L):
        return (s1 << L) | res2

    t = (s1 | (s1 - 1)) + 1
    res1 = t | ((((t & (0 - t)) // (s1 & (0 - s1))) >> 1) - 1)
    res2 = 0
    for i in range(N//2):
        if s2 & (1<<i):
            res2 |= 1
            res2 <<= 1
    res2 >>= 1
    return (res1 << L) | res2

# python function to calculate the starting state to generate the particle conserving basis
def get_s0_pcon_Np_Nd(N, Np):
    return (sum(1 << i for i in range(Np[0])) << (N//2)) | sum(1 << i for i in range(Np[1]))


# python function to calculate the size of the particle-conserved basis,
# i.e. BEFORE applying pre_check_state and symmetry maps
def get_Ns_pcon_Np_Nd(N, Np):
    return comb(N//2, Np[0], exact=True) * comb(N//2, Np[1], exact=True)


def real_if_close(mat):
    if isinstance(mat, np.ndarray):
        return np.real_if_close(mat)
    else:
        mat.data = np.real_if_close(mat.data)
        return mat


class spin_super_basis(user_basis):
    def __init__(self, N, pauli:bool, Np=None, Nd=None, pblock=None, kblock=None, zblock=None, pyblock=None, **blocks):
        assert N < 16, "N too large"
        
        if Nd is None and Np is None:
            pcon_dict = None
        else:
            if Nd is None:
                next_state = next_state_Np
                get_Ns_pcon = get_Ns_pcon_Np
                get_s0_pcon = get_s0_pcon_Np
                if isinstance(Np, int):
                    next_state_args = np.array([Np], dtype=np.uint32) 
                else:
                    next_state_args = np.array(Np, dtype=np.uint32) 
            else:
                if isinstance(Nd, int):
                    Nd = [Nd]
                Nd = list(set(Nd))
                if len(Nd) != len(set(Nd + [-nd for nd in Nd])):
                    warn(f"Ndiff does not contain both +nd and -nd, cannot realify :(")
                    
                # if isinstance(Nd, int):
                #     Nd = list(set([Nd, -Nd]))
                # else:
                #     assert all([0 <= nd <= N for nd in Nd]), f"All Ndiff must be between 0 and N={N}, but got Ndiff={Nd}."
                #     Nd = list(set(Nd + [-nd for nd in Nd if nd != 0]))
                
                if Np is None:
                    next_state = next_state_Nd
                    get_Ns_pcon = get_Ns_pcon_Nd
                    get_s0_pcon = get_s0_pcon_Nd
                    Np = [N + nd for nd in Nd]
                    next_state_args = np.array(Np, dtype=np.uint32)
                else:
                    next_state = next_state_Np_Nd
                    get_Ns_pcon = get_Ns_pcon_Np_Nd
                    get_s0_pcon = get_s0_pcon_Np_Nd
                    if isinstance(Np, int):
                        Np = [Np]
                    Nup2 = []
                    for i in Np:
                        for j in Nd:
                            assert (i+j)%2 == 0, f"Nup + Ndiff must be even, but got Nup={i}, Ndiff={j}."
                            Nup2.append(((i+j)//2, (i-j)//2))
                    next_state_args = np.array(Nup2, dtype=np.uint32)
                    Np = next_state_args

            n_sectors = None  # number of particle sectors
            count_particles_args = np.array([2*N], dtype=np.int32)
            pcon_dict = dict(
                Np=Np,
                next_state=next_state,
                next_state_args=next_state_args,
                get_Ns_pcon=get_Ns_pcon,
                get_s0_pcon=get_s0_pcon,
                count_particles=count_particles,
                count_particles_args=count_particles_args,
                n_sectors=n_sectors,
            )

        op_args = np.array([], dtype=np.uint32)
        if pauli is True:
            op_dict = dict(op=op_pauli, op_args=op_args)
            self._pauli = -1
        elif pauli is False:
            op_dict = dict(op=op_spin, op_args=op_args)
            self._pauli = 0
        else:
            raise ValueError("pauli must be True or False")

        maps = blocks
        if pblock is not None:
            P_args = np.array([], dtype=np.uint32)
            maps["P_block"] = (parity_x, 2, pblock, P_args)
        if pyblock is not None:
            Py_args = np.array([], dtype=np.uint32)
            maps["Py_block"] = (parity_y, 2, pyblock, Py_args)
        if kblock is not None:
            T_args = np.array([1, (1 << N) - 1], dtype=np.uint32)
            maps["T_block"] = (translation_x, N, kblock, T_args)
        if zblock is not None:
            Z_args = np.array([(1 << (2*N)) - 1], dtype=np.uint32)
            maps["Z_block"] = (spin_inversion, 2, zblock, Z_args)
        super().__init__(
            np.uint32,
            2*N,
            op_dict=op_dict,
            allowed_ops=set("+-xyznI"),
            sps=2,
            pcon_dict=pcon_dict,
            **maps,
        )
        self._user_N = N
        self._user_maps = maps
        self._user_pcon_dict = pcon_dict
        self._user_op_dict = op_dict
        self._P = None
        self._sym_basis = None
        self._asym_basis = None

    def get_basis_pcon(self):
        if self._basis_pcon is None:
            self._basis_pcon = user_basis(
                np.uint32,
                2*self._user_N,
                op_dict=self._user_op_dict,
                allowed_ops=set("+-xyznI"),
                sps=2,
                pcon_dict=self._user_pcon_dict
            )
        return self._basis_pcon
    
    @property
    def sym_basis(self):
        if self._sym_basis is None:
            maps = self._user_maps.copy()
            Py_args = np.array([], dtype=np.uint32)
            maps["Py_block"] = (parity_y, 2, 0, Py_args)
            self._sym_basis = user_basis(
                np.uint32,
                2*self._user_N,
                op_dict=self._user_op_dict,
                allowed_ops=set("+-xyznI"),
                sps=2,
                pcon_dict=self._user_pcon_dict,
                **maps,
            )
        return self._sym_basis
    
    @property
    def asym_basis(self):
        if self._asym_basis is None:
            maps = self._user_maps.copy()
            Py_args = np.array([], dtype=np.uint32)
            maps["Py_block"] = (parity_y, 2, 1, Py_args)
            self._asym_basis = user_basis(
                np.uint32,
                2*self._user_N,
                op_dict=self._user_op_dict,
                allowed_ops=set("+-xyznI"),
                sps=2,
                pcon_dict=self._user_pcon_dict,
                **maps,
            )
        return self._asym_basis
     
    def project_matrix(self, pcon=False):
        if pcon:
            self._sym_basis._basis_pcon = self.get_basis_pcon()
            self._asym_basis._basis_pcon = self.get_basis_pcon()
        if self._P is None:
            if self.Ns == 2**(2*self._user_N):
                P_sym = self.sym_basis.get_proj(np.complex128, pcon=pcon)
                P_antisym = 1j*self.asym_basis.get_proj(np.complex128, pcon=pcon)
            else:
                P0 = self.get_proj(np.complex128, pcon=pcon)
                P_sym = P0.conj().T @ self.sym_basis.get_proj(np.complex128, pcon=pcon)
                P_antisym = 1j*(P0.conj().T @ self.asym_basis.get_proj(np.complex128, pcon=pcon))
            
            self._P = sp.hstack([P_sym, P_antisym], format='csr')
        return self._P
    
    def realify(self, liou_mat, pcon=False):
        P = self.project_matrix(pcon=pcon)
        res = P.conj().T @ liou_mat @ P
        return real_if_close(res)

    def real_proj_to(self, state, pcon=False, sparse=False):
        v1 = self.sym_basis.project_to(state, pcon=pcon, sparse=sparse)
        v1 = real_if_close(v1)
        v2 = -1j*self.asym_basis.project_to(state, pcon=pcon, sparse=sparse)
        v2 = real_if_close(v2)
        if sparse:
            v = sp.vstack([v1, v2], format='csr')
        else:
            v = np.hstack([v1, v2]).reshape(-1)
        return v
 
    def real_proj_from(self, state, pcon=False, sparse=False):
        if pcon is True:
            Ns = self.get_basis_pcon().Ns
        else:
            Ns = 2**(2*self._user_N)
        if sparse:
            res = sp.csr_array([], shape=(Ns, state.shape[0]), dtype=np.complex128)
        else:
            res = np.zeros(Ns, dtype=np.complex128)
        res += self.sym_basis.project_from(state[:self.sym_basis.Ns], pcon=pcon, sparse=sparse)
        res += 1j*self.asym_basis.project_from(state[self.sym_basis.Ns:], pcon=pcon, sparse=sparse)
        return res


