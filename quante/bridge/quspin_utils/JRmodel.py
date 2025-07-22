# -*- coding: utf-8 -*-# @Author: hzhu
# @Date:   2025-07-19 20:32:04
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-22 21:10:10

# This file is an example on 

import os
import pickle
import threading
import queue
import time

import numpy as np
import scipy.sparse.linalg as spla

from quspin.operators._make_hamiltonian import _consolidate_static
from quspin.basis.basis_general.base_general import _check_symm_map
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
from itertools import permutations
from scipy.sparse import save_npz, load_npz
from scipy.sparse.linalg import LinearOperator
from typing import Literal

class TriangularLattice:
    r"""generate a triangular lattice with periodic boundary conditions.
    
    .. code-block:: text
        |                    03    00    01    02    03
        |                      ⋱  ⋰  ⋱  ⋰  ⋱  ⋰  ⋱  ⋰  
        |                 23 ⋯ 20 -- 21 -- 22 -- 23 ⋯ 20
        |                   ⋱   /  \  /  \  /  \  /  ⋱ 
        |               19 ⋯ 16 -- 17 -- 18 -- 19 ⋯ 16
        |                 ⋱  /  \  /  \  /  \  /  ⋱  
        |            15 ⋯ 12 -- 13 -- 14 -- 15 ⋯ 12
        |              ⋱  /  \  /  \  /  \  /  ⋱ 
        |         11 ⋯ 08 -- 09 -- 10 -- 11 ⋯ 08
        |           ⋱  /  \  /  \  /  \  /  ⋱  
        |      07 ⋯ 04 -- 05 -- 06 -- 07 ⋯ 04
        |        ⋱  /  \  /  \  /  \  /  ⋱                    
        |   03 ⋯ 00 -- 01 -- 02 -- 03 ⋯ 00        / Ly
        |        ⋰  ⋱  ⋰  ⋱  ⋰  ⋱  ⋰  ⋱           /
        |      20    21    22    23    20         ----  Lx
    """    
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx * Ly  # total number of sites
    
    def c2i(self, x, y):
        """Convert coordinates (x, y) to index in the triangular lattice."""
        effx, effy = x % self.Lx, y % self.Ly
        return effx + self.Lx * effy
    
    def i2c(self, s):
        """Convert index s to coordinates (x, y) in the triangular lattice."""
        return s % self.Lx, s // self.Lx
    
    def nearest_neighbors(self, s):
        r"""Return the nearest neighbors of each site in a triangular lattice.
        
        Example:
        .. code-block:: text
            |        (1,2)     (2,2)         
            |            \\   //           
            |    (1,1) == (2,1) == (3,1)
            |            //   \\                          
            |        (2,0)     (3,0)
        """
        x, y = self.i2c(s)
        nns = [
            [x + 1, y], # Right
            [x, y + 1], # Up-Right
            [x - 1, y + 1], # Up-Left
            [x - 1, y], # Left
            [x, y - 1], # Down-Left
            [x + 1, y - 1]  # Down-Right
        ]
        return [self.c2i(x, y) for x, y in nns]
    
    def next_nearest_neighbors(self, s):
        r"""Return the next nearest neighbors of each site in a triangular lattice.

        Example:
        .. code-block:: text
            |                [2,4]
            |                /     \
            |    [1,3] -- (2,3)     (3,3) -- [4,3]
            |        \       \     /       /   
            |        (2,2) -- (3,2) -- (4,2)
            |        /       /     \       \                 
            |    [2,1] -- (3,1)     (4,1) -- [5,1]
            |                \     /
            |                [4,0]
        """
        x, y = self.i2c(s)
        nnns = [
            [x + 1, y + 1], # Up-Right
            [x - 1, y + 2], # Up
            [x - 2, y + 1], # Up-Left
            [x - 1, y - 1], # Down-Left
            [x + 1, y - 2], # Down
            [x + 2, y - 1], # Down-Right
        ]
        return [self.c2i(x, y) for x, y in nnns]
    
    def j1j2(self, j1=1., j2=1.):
        r"""Generate `quspin` list for the J1-J2 model on a triangular lattice.

        .. math::
            j1 \sum_{<i,j>} S_i \cdot S_j + j2 \sum_{<<i,j>>} S_i \cdot S_j
        
        """
        coef_posn = []
        for i in range(self.N):
            for j in self.nearest_neighbors(i):
                if j > i and abs(j1) > 0:
                    coef_posn.append([j1, i, j])
            for j in self.next_nearest_neighbors(i):
                if j > i and abs(j2) > 0:
                    coef_posn.append([j2, i, j])
        return [[oper, coef_posn] for oper in ['xx', 'yy', 'zz']]
    
    def diamond_cell(self, s):
        r"""Return the indices of the sites in the diamond cell at site s.
        
        There are three types of posible diamond cell:

        .. code-block:: text
            |       n:(3,3) -- j:(4,3)
            |          /    \\    /
            |  i/s:(3,2) -- m:(4,2)
        
        .. code-block:: text
            |          j:(3,3)
            |          /    \ 
            |  m/s:(3,2) == n:(4,2)
            |          \    /
            |          i:(4,1) 
        
        .. code-block:: text
            |  i:(2,3) -- n:(3,3)
            |      \    //    \ 
            |     m/s:(3,2) -- j:(4,2)

        return: (i,j,m,n) with i = s
        """
        x, y = self.i2c(s)
        diamond1 = [
            [x, y], [x + 1, y + 1], [x, y + 1], [x + 1, y],
        ]
        diamond2 = [
            [x + 1, y - 1], [x, y + 1], [x, y], [x + 1, y],
        ]
        diamond3 = [
            [x - 1, y + 1], [x + 1, y], [x, y], [x, y + 1],
        ]
        return [tuple(self.c2i(x, y) for x, y in diamond) 
                for diamond in [diamond1, diamond2, diamond3]]


    def cc(self, r=10.):
        r"""Generate `quspin` list for the chirality-chirality interaction.
        
        .. math::
            - r \sum_{<ijmn>} [S_i \cdot (S_m \times S_n)][S_j \cdot (S_m \times S_n)] + h.c.
        
        """
        posn = []
        for s in range(self.N):
            diamonds = self.diamond_cell(s)
            for i, j, m, n in diamonds:
                posn.append([i, j, m, n])
                # h.c.
                posn.append([j, i, m, n])

        terms, signs = generate_cross_dot_terms()
        res = []
        for term, sign in zip(terms, signs):
            coef_posn_positive = [
                [(-r)*sign, *pos] for pos in posn
            ]
            res.append([term, coef_posn_positive])
        return res
                

def generate_cross_dot_terms():
    """Generate cross dot product terms for the chirality-chirality interaction.
    
    total 36 terms.
    """
    axes = 'xyz'
    epsilon_sign = {}

    # Build the Levi-Civita sign table
    for p in permutations([0, 1, 2]):
        i, j, k = p
        key = axes[i] + axes[j] + axes[k]
        # Compute parity of permutation: even -> +1, odd -> -1
        parity = (
            1 if (i - j)*(j - k)*(k - i) > 0 else -1
        )
        epsilon_sign[key] = parity

    terms = []
    signs = []

    for abc in epsilon_sign:
        for a_b_c in epsilon_sign:
            sign = epsilon_sign[abc] * epsilon_sign[a_b_c]
            term, coef = reduce_term(abc, a_b_c)
            terms.append(term)
            signs.append(sign*coef)

    return terms, signs

reduce_dic = {
    ('x', 'x'): ('I', 1/4),
    ('y', 'y'): ('I', 1/4),
    ('z', 'z'): ('I', 1/4),
    ('x', 'y'): ('z', 1j/2),
    ('x', 'z'): ('y', -1j/2),
    ('y', 'x'): ('z', -1j/2),
    ('y', 'z'): ('x', 1j/2),
    ('z', 'x'): ('y', 1j/2),
    ('z', 'y'): ('x', -1j/2),
}           

def reduce_term(abc, a_b_c):
    a, b, c = abc
    a_, b_, c_ = a_b_c

    b__, c1 = reduce_dic[(b,b_)]
    c__, c2 = reduce_dic[(c,c_)]
    return f"{a}{a_}{b__}{c__}", c1 * c2

def clean_static(static):
    """Clean the static list by removing duplicate terms and combining coefficients."""
    # first expand with pmz
    from quspin.basis import spin_basis_general
    basis = spin_basis_general(N=1, S='1/2', pauli=0)  # dummy call to load the module
    tmp, _ = basis.expanded_form(static)
    res = []
    for oper, posn_coef in tmp:
        if 'I' in oper:
            # find all positions of 'i' and remove them
            inc_indx = [i for i, c in enumerate(oper) if c == 'I']
            new_oper = ''.join(c for i, c in enumerate(oper) if i not in inc_indx)
            new_posn_coef = [
                [t 
                 for i,t in enumerate(eachterm) 
                 if (i-1) not in inc_indx
                ] 
                for eachterm in posn_coef
            ]
            res.append([new_oper, new_posn_coef])
        else:
            res.append([oper, posn_coef])
           
    res, _ = basis.expanded_form(res)
    return res


def get_basis(Lx, Ly, Nup=None, kblock=(0,0), pblock=None, zblock=0):
    """Generate the basis for a 2D triangular lattice."""
    try:
        with open("data/basis_info.pkl", "rb") as f:
            datadic = pickle.load(f)
        if (datadic["Lx"] == Lx and datadic["Ly"] == Ly 
            and datadic["Nup"] == Nup and datadic["kblock"] == kblock 
            and datadic["pblock"] == pblock and datadic["zblock"] == zblock):
            return datadic["basis"]
    except FileNotFoundError:
        pass
    
    N_2d = Lx * Ly  # total number of sites
    s = np.arange(N_2d)  # sites [0,1,2,..]
    x = s % Lx  # x positions for sites
    y = s // Lx  # y positions for sites
    T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
    T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
    Z = -(s + 1)  # spin inversion
    P_x = x + Lx * (Ly - y - 1)  # reflection about x-axis
    P_y = (Lx - x - 1) + Lx * y  # reflection about y-axis
    if kblock is not None:
        kxblock = (T_x, kblock[0])
        kyblock = (T_y, kblock[1])
    else:
        kxblock = kyblock = None
    
    if pblock is not None:
        pxblock = (P_x, pblock[0])
        pyblock = (P_y, pblock[1])
    else:
        pxblock = pyblock = None

    if zblock is not None:
        _zblock = (Z, zblock) 

    basis = spin_basis_general(
        N_2d,
        S='1/2',
        pauli=0,
        Nup=Nup,
        kxblock=kxblock,
        kyblock=kyblock,
        pxblock=pxblock,
        pyblock=pyblock,
        zblock=_zblock,
    )
   
    
    with open("data/basis_info.pkl", "wb") as f:
        pickle.dump({
            "Lx": Lx,
            "Ly": Ly,
            "Nup": Nup,
            "kblock": kblock,
            "pblock": pblock,
            "zblock": zblock,
            "basis": basis
        }, f)
    return basis

def generate_sym_oper(basis, opstr, indx, J):
    generated_list = [(opstr, indx, J), ]
    while True:
        tmp = []
        bk = True
        for block, map in basis._maps_dict.items():
            _, missing_ops = _check_symm_map(
                map, basis._sort_opstr, generated_list
            )

            for opstr, indx, J in missing_ops:
                already_exists = False
                for i, (opstr1, indx1, J1) in enumerate(tmp):
                    if opstr == opstr1 and all(a == b for a, b in zip(indx, indx1)) and J1 == J:
                        already_exists = True
                        break
                if not already_exists:
                    tmp.append((opstr, indx, J))

            if missing_ops:
                bk = False
            
        if bk:
            break
        generated_list += tmp
    return generated_list
 
def split_static(static, basis):
    """accerlerate the generation of the Hamiltonian matrix using the hermitian property."""
    res = []

    remained_oper = _consolidate_static(static)

    while len(remained_oper) > 0:
        opstr0, indx0, J0 = remained_oper.pop(0)
        # generate the symmetry operations
        generated_list = generate_sym_oper(basis, opstr0, indx0, J0)

        # remove the generated list from the remained operations
        for opstr, indx, J in generated_list:
            if opstr== opstr0 and all(a==b for a,b in zip(indx, indx0)) and J == J0:
                continue

            notin = True
            for i, (opstr1, indx1, J1) in enumerate(remained_oper):
                if opstr == opstr1 and all(a==b for a,b in zip(indx, indx1)):
                    notin = False
                    remained_oper.pop(i)
                    if J != J1:
                        remained_oper.append((opstr, indx, J1 - J))
                    break
            if notin:
                print(opstr, indx, J, "not in remained operations")

        # consolidate the generated list
        static_dict = {}
        for opstr, indx, J  in generated_list:
            indx = list(indx)
            indx.insert(0, J)
            if opstr in static_dict:
                static_dict[opstr].append(indx)
            else:
                static_dict[opstr] = [indx]
        generated_static = [[str(key), list(value)] for key, value in static_dict.items()]

        res.append(generated_static)
    return res

def generate_hamiltonian(static, basis, mode='normal'):
    if mode == 'normal':
        H = hamiltonian(static, [], basis=basis, dtype=np.complex128, check_herm=True)
        print("Number of Non zeros:", H.static.nnz)
        return H
    elif mode == 'sequential':
        splited_static = split_static(static, basis)
        dtype = np.float64
        addany = False
        for idx, each_static in enumerate(splited_static):
            if os.path.exists(f"data/hamiltonian/{idx}.npz"):
                print(f"Matrix {idx} already exists, skipping.")
                continue
            addany = True
            H = hamiltonian(each_static, [], basis=basis, dtype=np.complex128, check_herm=False, check_pcon=False, check_symm=False).static
            # if the spare matrix is almost real, convert it to real
            if np.allclose(H.data.imag, 0, atol=1e-10):
                H = H.real
            else:
                dtype = np.complex128
            save_npz(f"data/hamiltonian/{idx}.npz", H, compressed=False)
            print(f"Saved matrix {idx} with shape {H.shape} and dtype {dtype}")
        if addany:
            with open(f"data/hamiltonian/mat_info.pkl", "wb") as f:
                pickle.dump({"dim": H.shape, "dtype": dtype, "number": len(splited_static)}, f)
        return None

def preload_matrices(filenames, q):
    for fname in filenames:
        Ai = load_npz(fname)
        q.put(Ai)
    q.put(None)  # 结束信号

ct = 0
monitor = 10
def matvec_factory(filenames):
    def matvec(v):
        result = np.zeros_like(v)
        q = queue.Queue(maxsize=2)  # 控制并行深度：最多缓存 2 个矩阵

        # 启动加载线程
        loader = threading.Thread(target=preload_matrices, args=(filenames, q))
        loader.start()

        while True:
            Ai = q.get()
            if Ai is None:
                break
            result += Ai @ v
            del Ai

        loader.join()
        global ct
        ct += 1
        if ct % monitor == 0:
            nm1 = np.linalg.norm(v)
            nm2 = np.linalg.norm(result)
            residual = np.linalg.norm(result/nm2 - v/nm1)
            print(f"Matvec #{ct}, norm(result): {nm2:.2e}, residual: {residual:.2e}")
            np.save("data/psi_ground.npy", v)
        else:
            print(f"[matvec #{ct}] done")
        return result
    return matvec

def lanczos(H, mode='normal'):
    if mode == 'normal':
        return H.eigsh(k=1, which="SA", maxiter=1e4, return_eigenvectors=True)
    elif mode == 'sequential':
        with open("data/hamiltonian/mat_info.pkl", "rb") as f:
            mat_info = pickle.load(f)
        N = mat_info["dim"][0]
        dtype = mat_info["dtype"]
        number = mat_info["number"]
        matrix_dir = "data/hamiltonian"
        matrix_files = [os.path.join(matrix_dir, f"{i}.npz") for i in range(number)]
        matvec = matvec_factory(matrix_files)
        A_linop = LinearOperator((N, N), matvec=matvec, dtype=dtype)
        try:
            v0 = np.load("data/psi_ground.npy")
        except FileNotFoundError:
            v0 = None
        t = time.time()
        res = spla.eigsh(A_linop, k=1, which="SA", maxiter=1e4, return_eigenvectors=True, v0=v0)
        print(f"Lanczos solver: {time.time() - t:.2f} seconds.")
        return res

def solve_ground_state(static, basis, mode:Literal['normal', 'sequential']='normal'):
    # Generate the Hamiltonian matrix
    print("Basis size:", basis.Ns)
    H = generate_hamiltonian(static, basis, mode=mode)

    # Solve for the ground state energy and wave function
    Emin, psi = lanczos(H, mode=mode)

    # save ground state energy
    print("Ground state energy:", Emin)
    np.savetxt("data/E_ground.txt", [Emin])  

    # save ground state wave function
    np.save("data/psi_ground.npy", psi) 

    return psi


if __name__ == "__main__":
    os.makedirs('data/hamiltonian', exist_ok=True)
    
    Lx = 4
    Ly = 6
    Jnn = 0.
    Jnnn = 0.
    r = 10.

    tl = TriangularLattice(Lx=Lx, Ly=Ly)
    static = tl.j1j2(j1=Jnn, j2=Jnnn) + tl.cc(r=r)
    static = clean_static(static)
    basis = get_basis(Lx, Ly, Nup=(Lx*Ly)//2, kblock=(0,0), zblock=0)
    print(basis.Ns)

    
    import quante as qt
    from quante.bridge.quspin_utils import optimize_basis
    basis = optimize_basis(basis)
    # import dowhen
    # dowhen.do("op_list = tqdm(op_list, ascii=True)").when(basis._make_matrix, "+2")

    with qt.basicfun.Timer("total time"):
        solve_ground_state(static, basis, mode='sequential')
    
    