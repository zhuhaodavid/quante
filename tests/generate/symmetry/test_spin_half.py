# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-09-08 17:12:39
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-31 21:10:49

# !! 通过与 quspin 的对比，测试 quante 中的生成基矢的代码

import unittest
import numpy as np
import sys
sys.dont_write_bytecode = True

import os
# 增加当前路径的上级路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
great_grandparent_dir = os.path.dirname(grandparent_dir)
sys.path.append(great_grandparent_dir)


# import sys
# sys.dont_write_bytecode = True
# sys.path.append(r'E:\hzhu\OneDrive\Share\python_library')
# import quante as qt
# import numpy as np
# import scipy.sparse.linalg as sla
# import time

# from quspin.basis import spin_basis_1d
# from quspin.operators import hamiltonian
# L = 20
# ham = qt.generate.operas.heisenberg_operator(L, j=(1, 1, 1))
# ham = ham.expandxy(pauli=False)
# quspin_basis = spin_basis_1d(L=L, pauli=0)
# lis = ham.quspin_form()

# t = time.time()
# mat3 = hamiltonian(lis, [], basis=quspin_basis, check_symm=False, check_herm=False, dtype=np.float64).static
# print(time.time()-t)

# basis = qt.generate.basis.spin_basis(L=L)
# t = time.time()
# mat1 = ham.to_matrix(basis, sparse=True)
# print(time.time()-t)


# print(sla.norm(mat1 - mat3))


class TestSpinHalf(unittest.TestCase):
    
    def _generate_random_hamiltonian(self, L):
        import quante.generate.operas as op
        return op.sum(np.random.randn() * op.xx(i,(i+1)%L) + np.random.randn() * op.yy(i,(i+1)%L) + np.random.randn() * op.zz(i,(i+1)%L) for i in range(L)) + op.sum(np.random.randn() * op.x(i) + np.random.randn() * op.y(i) + np.random.randn() * op.z(i) for i in range(L))
    
    def _generate_Nup_hamiltonian(self, L):
        import quante.generate.operas as op
        return op.sum(np.random.randn() * (op.xx(i,(i+1)%L) + op.yy(i,(i+1)%L)) + np.random.randn() * op.zz(i,(i+1)%L) for i in range(L)) + op.sum(np.random.randn() * op.z(i) for i in range(L))
    
    def _generate_kblock_hamiltonian(self, L):
        import quante.generate.operas as op
        jx = np.random.randn()
        jy = np.random.randn()
        jz = np.random.randn()
        hx = np.random.randn()
        hy = np.random.randn()
        hz = np.random.randn()
        return op.sum(jx * op.xx(i,(i+1)%L) + jy * op.yy(i,(i+1)%L) + jz * op.zz(i,(i+1)%L) for i in range(L)) + op.sum(hx * op.x(i) + hy * op.y(i) + hz * op.z(i) for i in range(L))
    
    def _generate_pblock_hamiltonian(self, L):
        import quante.generate.operas as op
        jx = np.random.randn(L - 1)
        jx = jx + jx[::-1]
        jy = np.random.randn(L - 1)
        jy = jy + jy[::-1]
        jz = np.random.randn(L - 1)
        jz = jz + jz[::-1]
        hx = np.random.randn(L)
        hx = hx + hx[::-1]
        hy = np.random.randn(L)
        hy = hy + hy[::-1]
        hz = np.random.randn(L)
        hz = hz + hz[::-1]
        return op.sum(jx[i] * op.xx(i,(i+1)%L) + jy[i] * op.yy(i,(i+1)%L) + jz[i] * op.zz(i,(i+1)%L) for i in range(L-1)) + op.sum(hx[i] * op.x(i) + hy[i] * op.y(i) + hz[i] * op.z(i) for i in range(L))
    
    def _generate_zblock_hamiltonian(self, L):
        import quante.generate.operas as op
        jx = np.random.randn(L)
        jx = jx + jx[::-1]
        jz = np.random.randn(L)
        jz = jz + jz[::-1]
        return op.sum(jx[i] * (op.xx(i,(i+1)%L) + op.yy(i,(i+1)%L)) + jz[i] * op.zz(i,(i+1)%L) for i in range(L))
        
    def _generate_Nup_kblock_hamiltonian(self, L):
        import quante.generate.operas as op
        jx = np.random.randn()
        jz = np.random.randn()
        hz = np.random.randn()
        return op.sum(jx * (op.xx(i,(i+1)%L) + op.yy(i,(i+1)%L)) + jz * op.zz(i,(i+1)%L) for i in range(L)) + op.sum(hz * op.z(i) for i in range(L))
    
    def _generate_Nup_kblock_zblock_hamiltonian(self, L):
        import quante.generate.operas as op
        jx = np.random.randn()
        jz = np.random.randn()
        hz = np.random.randn()
        return op.sum(jx * (op.xx(i,(i+1)%L) + op.yy(i,(i+1)%L)) + jz * op.zz(i,(i+1)%L) for i in range(L))
    
    def _generate_Nup_pblock_hamiltonian(self, L):
        import quante.generate.operas as op
        jx = np.random.randn(L - 1)
        jx = jx + jx[::-1]
        jz = np.random.randn(L - 1)
        jz = jz + jz[::-1]
        hz = np.random.randn(L)
        hz = hz + hz[::-1]
        return op.sum(jx[i] * (op.xx(i,(i+1)%L) + op.yy(i,(i+1)%L)) + jz[i] * op.zz(i,(i+1)%L) for i in range(L-1)) + op.sum(hz[i] * op.z(i) for i in range(L))
    
    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_noblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.noblock.matrixele import heisenberg_matrix_element
        L = 10
        for cyclic in [True, False]:
            jxy = np.random.randn()
            jz = np.random.randn()
            mat1 = heisenberg_matrix_element(L, jxy, jz, cyclic=cyclic)
            
            from quante.generate.operas import heisenberg_operator
            from quante.generate.basis import spin_basis
            ham = heisenberg_operator(L, (jxy, jxy, jz), 0., cyclic=cyclic)
            basis = spin_basis(L=L)
            mat2 = ham.to_matrix(basis, sparse=True).toarray()
        
            self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d    # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_random_hamiltonian(L)
        quspin_basis = spin_basis_1d(L=L, pauli=0)
        mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False).toarray()
        mat4 = ham.to_matrix(basis, sparse=True).toarray()
        
        self.assertTrue(np.allclose(mat3, mat4))

    @unittest.skipIf(sys.version_info <= (3,11), "requires python3.10 or lower")
    def test_Nup(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.Nup.matrixele import heisenberg_matrix_element
        from quante.generate.symmetry.spin_half.Nup.defbasis import construct_Nup_basis
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        jxy = np.random.randn()
        jz = np.random.randn()
        for cyclic in [True, False]:
            for Nup in range(L+1):
                M, s_list = construct_Nup_basis(L, Nup)
                mat1 = heisenberg_matrix_element(L, M, s_list, jxy, jz, cyclic=cyclic)
                
                ham = heisenberg_operator(L, (jxy, jxy, jz), 0., cyclic=cyclic)
                basis = spin_basis(L=L, Nup=Nup)
                mat2 = ham.to_matrix(basis, sparse=True).toarray()
                
                self.assertTrue(np.allclose(mat1, mat2))
            
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_Nup_hamiltonian(L)
        for Nup in range(L+1):
            quspin_basis = spin_basis_1d(L=L, Nup=Nup, pauli=0)
            mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
            basis = spin_basis(L=L, Nup=Nup)
            mat4 = ham.to_matrix(basis, sparse=True).toarray()
            
            self.assertTrue(np.allclose(mat3, mat4))
        
    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_kblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.kblock.defbasis import construct_kblock_basis
        from quante.generate.symmetry.spin_half.kblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        for k in range(L):
            M, s_list, R_list = construct_kblock_basis(L, k)
            mat1 = heisenberg_matrix_element(L, M, k, jxy=j, jz=h, s_list=s_list, R_list=R_list)
            basis = spin_basis(L=L, kblock=k)
            ham = heisenberg_operator(L, (j, j, h), 0., cyclic=True)
            mat2 = ham.to_matrix(basis, sparse=True).toarray()
            self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_kblock_hamiltonian(L)
    
        for k in range(L):
            quspin_basis = spin_basis_1d(L=L, pauli=0, kblock=k)
            mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
            basis = spin_basis(L=L, kblock=k)
            mat4 = ham.to_matrix(basis, sparse=True).toarray()
    
            self.assertTrue(np.allclose(mat3, mat4))
        
    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_pblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.pblock.defbasis import construct_pblock_basis
        from quante.generate.symmetry.spin_half.pblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        cyclic = False
        for p in [-1, 1]:
            M, s_list = construct_pblock_basis(L, p)
            mat1 = heisenberg_matrix_element(L, M, p, jxy=j, jz=h, s_list=s_list, cyclic=cyclic)
            basis = spin_basis(L=L, pblock=p)
            ham = heisenberg_operator(L, (j, j, h), 0., cyclic=cyclic)
            mat2 = ham.to_matrix(basis, sparse=True).toarray()
            self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_pblock_hamiltonian(L)
        
        for p in [-1, 1]:
            quspin_basis = spin_basis_1d(L=L, pauli=0, pblock=p)
            mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
            basis = spin_basis(L=L, pblock=p)
            mat4 = ham.to_matrix(basis, sparse=True).toarray()
        
            self.assertTrue(np.allclose(mat3, mat4))
        
    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_zblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.zblock.defbasis import construct_zblock_basis
        from quante.generate.symmetry.spin_half.zblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        for cyclic in [True, False]:
            for z in [-1, 1]:
                M, s_list = construct_zblock_basis(L, z)
                mat1 = heisenberg_matrix_element(L, M, z, jxy=j, jz=h, s_list=s_list, cyclic = cyclic)
                basis = spin_basis(L=L, zblock=z)
                ham = heisenberg_operator(L, (j, j, h), 0., cyclic=cyclic)
                mat2 = ham.to_matrix(basis, sparse=True).toarray()
                self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_zblock_hamiltonian(L)
        
        for z in [-1, 1]:
            quspin_basis = spin_basis_1d(L=L, pauli=0, zblock=z)
            mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
            basis = spin_basis(L=L, zblock=z)
            mat4 = ham.to_matrix(basis, sparse=True).toarray()
        
            self.assertTrue(np.allclose(mat3, mat4))

    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_pzblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.pzblock.defbasis import construct_pzblock_basis
        from quante.generate.symmetry.spin_half.pzblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        for cyclic in [True, False]:
            for pz in [-1, 1]:
                M, s_list = construct_pzblock_basis(L, pz)
                mat1 = heisenberg_matrix_element(L, M, pz, jxy=j, jz=h, s_list=s_list, cyclic = cyclic)
                basis = spin_basis(L=L, pzblock=pz)
                ham = heisenberg_operator(L, (j, j, h), 0., cyclic=cyclic)
                mat2 = ham.to_matrix(basis, sparse=True).toarray()
                self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_zblock_hamiltonian(L)
        
        for pz in [-1, 1]:
            quspin_basis = spin_basis_1d(L=L, pauli=0, pzblock=pz)
            mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
            basis = spin_basis(L=L, pzblock=pz)
            mat4 = ham.to_matrix(basis, sparse=True).toarray()
        
            self.assertTrue(np.allclose(mat3, mat4))

    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_Nup_kblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.Nup_kblock.defbasis import construct_Nup_kblock_basis
        from quante.generate.symmetry.spin_half.kblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        cyclic = True
        for Nup in range(L+1):
            for k in range(L):
                M, s_list, R_list = construct_Nup_kblock_basis(L=L,Nup=Nup,k=k)
                mat1 = heisenberg_matrix_element(L, M, k, jxy=j, jz=h, s_list=s_list, R_list=R_list)
                basis = spin_basis(L=L, Nup=Nup, kblock=k)
                ham = heisenberg_operator(L, (j, j, h), 0., cyclic=cyclic)
                mat2 = ham.to_matrix(basis, sparse=True).toarray()
                self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_Nup_kblock_hamiltonian(L)
        
        for Nup in range(L+1):
            for k in range(L):
                quspin_basis = spin_basis_1d(L=L, pauli=0, Nup=Nup, kblock=k)
                mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
                basis = spin_basis(L=L, Nup=Nup, kblock=k)
                mat4 = ham.to_matrix(basis, sparse=True).toarray()
            
                self.assertTrue(np.allclose(mat3, mat4))
            
    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_Nup_pblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.Nup_pblock.defbasis import construct_Nup_pblock_basis
        from quante.generate.symmetry.spin_half.pblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        for cyclic in [True, False]:
            for Nup in range(L+1):
                for p in [-1,1]:
                    M, s_list = construct_Nup_pblock_basis(L, Nup, p)
                    mat1 = heisenberg_matrix_element(L, M, p, jxy=j, jz=h, s_list=s_list, cyclic=cyclic)
                    basis = spin_basis(L=L, Nup=Nup, pblock=p)
                    ham = heisenberg_operator(L, (j, j, h), 0., cyclic=cyclic)
                    mat2 = ham.to_matrix(basis, sparse=True).toarray()
                    self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_Nup_pblock_hamiltonian(L)
        
        for Nup in range(L+1):
            for p in [-1,1]:
                quspin_basis = spin_basis_1d(L=L, pauli=0, Nup=Nup, pblock=p)
                mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
                basis = spin_basis(L=L, Nup=Nup, pblock=p)
                mat4 = ham.to_matrix(basis, sparse=True).toarray()
            
                self.assertTrue(np.allclose(mat3, mat4))
            
    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_Nup_zblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.Nup_zblock.defbasis import construct_Nup_zblock_basis
        from quante.generate.symmetry.spin_half.zblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        for cyclic in [True, False]:
            for Nup in range(L//2+1):
                for z in [-1,1]:
                    M, s_list = construct_Nup_zblock_basis(L, Nup, z)
                    mat1 = heisenberg_matrix_element(L, M, z, jxy=j, jz=h, s_list=s_list, cyclic=cyclic)
                    basis = spin_basis(L=L, Nup=Nup, zblock=z)
                    ham = heisenberg_operator(L, (j, j, h), 0., cyclic=cyclic)
                    mat2 = ham.to_matrix(basis, sparse=True).toarray()
                    self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_zblock_hamiltonian(L)
        
        Nup = L//2
        for z in [-1,1]:
            quspin_basis = spin_basis_1d(L=L, pauli=0, Nup=Nup, zblock=z)
            mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
            basis = spin_basis(L=L, Nup=Nup, zblock=z)
            mat4 = ham.to_matrix(basis, sparse=True).toarray()
        
            self.assertTrue(np.allclose(mat3, mat4))

    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_Nup_pzblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.Nup_pzblock.defbasis import construct_Nup_pzblock_basis
        from quante.generate.symmetry.spin_half.pzblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        for cyclic in [True, False]:
            for Nup in range(L//2+1):
                for pz in [-1,1]:
                    M, s_list = construct_Nup_pzblock_basis(L, Nup, pz)
                    mat1 = heisenberg_matrix_element(L, M, pz, jxy=j, jz=h, s_list=s_list, cyclic=cyclic)
                    basis = spin_basis(L=L, Nup=Nup, pzblock=pz)
                    ham = heisenberg_operator(L, (j, j, h), 0., cyclic=cyclic)
                    mat2 = ham.to_matrix(basis, sparse=True).toarray()
                    self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_zblock_hamiltonian(L)
        
        Nup = L//2
        for pz in [-1,1]:
            quspin_basis = spin_basis_1d(L=L, pauli=0, Nup=Nup, pzblock=pz)
            mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
            basis = spin_basis(L=L, Nup=Nup, pzblock=pz)
            mat4 = ham.to_matrix(basis, sparse=True).toarray()
        
            self.assertTrue(np.allclose(mat3, mat4))

    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_Nup_kblock_pblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.Nup_kblock_pblock.defbasis import construct_Nup_kblock_pblock_basis
        from quante.generate.symmetry.spin_half.Nup_kblock_pblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        cyclic = True
        for Nup in range(L+1):
            for k in range(L//2+1):
                for p in [-1, 1]:
                    M, s_list, R_list, m_list = construct_Nup_kblock_pblock_basis(L=L, Nup=Nup, k=k, p=p)
                    mat1 = heisenberg_matrix_element(L, M, k, p, jxy=j, jz=h, s_list=s_list, R_list=R_list, m_list=m_list)
                    basis = spin_basis(L=L, Nup=Nup, kblock=k, pblock=p)
                    ham = heisenberg_operator(L, (j, j, h), 0., cyclic=cyclic)
                    mat2 = ham.to_matrix(basis, sparse=True).toarray()
                    self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_Nup_kblock_hamiltonian(L)
        for Nup in range(L+1):
            for k in range(L//2+1):
                for p in [-1, 1]:
                    quspin_basis = spin_basis_1d(L=L, pauli=0, Nup=Nup, kblock=k, pblock=p)
                    mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
                    engs3 = np.linalg.eigvalsh(mat3)
                    basis = spin_basis(L=L, Nup=Nup, kblock=k, pblock=p)
                    mat4 = ham.to_matrix(basis, sparse=True).toarray()
                    engs4 = np.linalg.eigvalsh(mat4)
                    
                    self.assertTrue(np.allclose(engs3, engs4))
        
    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_Nup_kblock_pblock_zblock(self):
        # 海森堡测试：
        from quante.generate.symmetry.spin_half.Nup_kblock_pblock_zblock.defbasis import construct_Nup_kblock_pblock_zblock_basis
        from quante.generate.symmetry.spin_half.Nup_kblock_pblock_zblock.matrixele import heisenberg_matrix_element
        from quante.generate.operas import heisenberg_operator
        from quante.generate.basis import spin_basis
        
        L = 10
        j = np.random.randn()
        h = np.random.randn()
        
        cyclic = True
        for k in range(L//2+1):
            for p in [-1, 1]:
                for z in [-1, 1]:
                    M, s_list, R_list, m_list, c_list = construct_Nup_kblock_pblock_zblock_basis(L=L, k=k, p=p, z=z)
                    mat1 = heisenberg_matrix_element(L, M, k, p, z, jxy=j, jz=h, s_list=s_list, R_list=R_list, m_list=m_list, c_list=c_list)
                    basis = spin_basis(L=L, Nup=L//2, kblock=k, pblock=p, zblock=z)
                    ham = heisenberg_operator(L, (j, j, h), 0., cyclic=cyclic)
                    mat2 = ham.to_matrix(basis, sparse=True).toarray()
                    self.assertTrue(np.allclose(mat1, mat2))
        
        # general hamiltonian
        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        ham = self._generate_Nup_kblock_zblock_hamiltonian(L)
        for k in range(L//2+1):
            for p in [-1, 1]:
                for z in [-1, 1]:
                    quspin_basis = spin_basis_1d(L=L, pauli=0, Nup=L//2, kblock=k, pblock=p, zblock=z)
                    mat3 = hamiltonian(ham.quspin_form(), [], basis=quspin_basis, check_symm=False, check_herm=False, check_pcon=False).toarray()
                    engs3 = np.linalg.eigvalsh(mat3)
                    basis = spin_basis(L=L, Nup=L//2, kblock=k, pblock=p, zblock=z)
                    mat4 = ham.to_matrix(basis, sparse=True).toarray()
                    engs4 = np.linalg.eigvalsh(mat4)
                    
                    self.assertTrue(np.allclose(engs3, engs4))

    @unittest.skipIf(sys.version_info <= (3,10), "requires python3.10 or lower")
    def test_time(self):
        import sys
        sys.dont_write_bytecode = True
        import quante as qt
        import numpy as np
        import scipy.sparse.linalg as sla  # type: ignore
        import time
        op = qt.generate.operas

        from quspin.basis import spin_basis_1d  # type: ignore
        from quspin.operators import hamiltonian  # type: ignore
        L = 20
        Nup = None
        ham = op.heisenberg_operator(L, j=(1, 1, 1))
        ham = ham.expandxy(pauli=False)
        quspin_basis = spin_basis_1d(L=L, Nup=Nup, pauli=0)
        lis = ham.quspin_form()

        t = time.time()
        mat3 = hamiltonian(lis, [], basis=quspin_basis, check_symm=False, check_herm=False, dtype=np.float64).static
        print(time.time()-t)

        basis = qt.generate.basis.spin_basis(L=L, Nup=Nup)
        t = time.time()
        mat1 = ham.to_matrix(basis, sparse=True)
        print(time.time()-t)
        
        self.assertAlmostEqual((sla.norm(mat1 - mat3)),0)
        
if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestSpinHalf("test_noblock"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    
    # import sys
    # sys.dont_write_bytecode = True
    # import quante as qt
    # import numpy as np
    # import scipy.sparse.linalg as sla
    # import time
    # op = qt.generate.operas

    # from quspin.basis import spin_basis_1d
    # from quspin.operators import hamiltonian
    # L = 10
    # Nup = None
    # ham = op.heisenberg_operator(L, j=(1, 1, 1))
    # ham = ham.expandxy(pauli=False)
    # quspin_basis = spin_basis_1d(L=L, Nup=Nup, pauli=0)
    # lis = ham.quspin_form()

    # t = time.time()
    # for i in range(100):
    #     mat3 = hamiltonian(lis, [], basis=quspin_basis, check_symm=False, check_herm=False, dtype=np.float64).static.toarray()
    # print(time.time()-t)

    # basis = qt.generate.basis.spin_basis(L=L, Nup=Nup)
    # t = time.time()
    # for i in range(500):
    #     mat1 = basis._heimat(1.,1.)
    # print(time.time()-t)
    