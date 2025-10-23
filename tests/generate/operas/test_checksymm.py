# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-10-22 12:29:23
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-22 23:00:42

import unittest
import quante as qt
import numpy as np
import quante.bridge.quspin_utils as qs
op = qt.generate.operas.spin

class TestSlaterState(unittest.TestCase):
    def _generate_random_hamiltonian(self, L):
        return (
            op.sum(
                    np.random.randn() * op.xx(i,j) 
                    + np.random.randn() * op.yy(i,j) 
                    + np.random.randn() * op.zz(i,j) 
                    + np.random.randn() * op.pm(i,j) 
                    + np.random.randn() * op.mp(i,j) 
                    + np.random.randn() * op.m(i) * op.m(j) 
                    + np.random.randn() * op.p(i) * op.p(j) for i in range(L) for j in range(L)
                ) +
            op.sum(
                    np.random.randn() * op.x(i) 
                    + np.random.randn() * op.y(i) 
                    + np.random.randn() * op.z(i) 
                    + np.random.randn() * op.n(i) 
                    for i in range(L)
                )
        )
    
    def _generate_random_hamiltonian_Nup(self, L):
        return (
            op.sum(
                    np.random.randn() * (op.xx(i,j) + op.yy(i,j))
                    + np.random.randn() * op.zz(i,j) 
                    + np.random.randn() * (op.pm(i,j) + op.mp(i,j))
                    for i in range(L) for j in range(L)
                ) +
            op.sum(
                    + np.random.randn() * op.z(i) 
                    + np.random.randn() * op.n(i) 
                    for i in range(L)
                )
        )
    
    def test_blocks(self):
        L = 10
        H = self._generate_random_hamiltonian(L)

        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, blocks='k')
        H_kblock = H.copy()
        for i in range(1,L):
            H = H.transform(L=L, perm='k')
            H_kblock += H
        H_kblock.check_symm(L=L, pauli=True, blocks='k')

        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, blocks='p')
        H_pblock = H + H.transform(L=L, perm='p')
        H_pblock.check_symm(L=L, pauli=True, blocks='p')

        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, blocks='z')
        H_pblock = H + H.expandn(to='pm').transform(L=L, perm='z')
        H_pblock.check_symm(L=L, pauli=True, blocks='z')

        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, blocks='pz')
        H_pblock = H + H.expandn(to='pm').transform(L=L, perm='pz')
        H_pblock.check_symm(L=L, pauli=True, blocks='pz')

        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, blocks=['k', 'p', 'z'])
        H_kpblock = H.copy()
        for i in range(1,L):
            H = H.transform(L=L, perm='k')
            H_kpblock += H
        H_kpblock = H_kpblock + H_kpblock.transform(L=L, perm='p')
        H_kpblock = H_kpblock + H_kpblock.expandn(to='pm').transform(L=L, perm='z')
        H_kpblock.check_symm(L=L, pauli=True, blocks=['k', 'p', 'z'])

    
        H = self._generate_random_hamiltonian_Nup(L)

        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, blocks=['Nup', 'k'])
        H_kblock = H.copy()
        for i in range(1,L):
            H = H.transform(L=L, perm='k')
            H_kblock += H
        H_kblock.check_symm(L=L, pauli=True, blocks=['Nup', 'k'])
    

    def test_spinbasis(self):
        L = 10
        H = self._generate_random_hamiltonian(L)
        basisfunc = qt.generate.basis.spin_basis

        basis = basisfunc(L=L, kblock=0)
        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, basis=basis)
        H_kblock = H.copy()
        for i in range(1,L):
            H = H.transform(L=L, perm='k')
            H_kblock += H
        H_kblock.check_symm(L=L, pauli=True, basis=basis) 

        basis = basisfunc(L=L, pblock=1)
        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, basis=basis)
        H_kblock = H + H.transform(L=L, perm='p')
        H_kblock.check_symm(L=L, pauli=True, basis=basis)

        basis = basisfunc(L=L, zblock=1)
        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, basis=basis)
        H_kblock = H + H.expandn(to='pm').transform(L=L, perm='z')
        H_kblock.check_symm(L=L, pauli=True, basis=basis)

        basis = basisfunc(L=L, kblock=0, pblock=1)
        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, basis=basis)
        H_kblock = H.copy()
        for i in range(1,L):
            H = H.transform(L=L, perm='k')
            H_kblock += H
        H_kblock = H_kblock + H_kblock.transform(L=L, perm='p')
        H_kblock.check_symm(L=L, pauli=True, basis=basis)

            
        H = self._generate_random_hamiltonian_Nup(L)
        basis = basisfunc(L=L, Nup=1, kblock=0)
        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, basis=basis)
        H_kblock = H.copy()
        for i in range(1,L):
            H = H.transform(L=L, perm='k')
            H_kblock += H
        H_kblock.check_symm(L=L, pauli=True, basis=basis)
    
    
    def test_spinbasis2d(self):
        Lx, Ly = 3,4
        L = Lx * Ly
        H = self._generate_random_hamiltonian(L)
        basisfunc = qt.generate.basis.spin_basis_2d

        basis = basisfunc(Lx=Lx, Ly=Ly, kxblock=0)
        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, basis=basis)
        s = np.arange(L)
        x = s % Lx
        y = s // Lx
        permx = (x + 1) % Lx + Lx * y
        H_kblock = H.copy()
        for i in range(1,Lx):
            H = H.transform(L=L, perm=permx)
            H_kblock += H
        H_kblock.check_symm(L=L, pauli=True, basis=basis) 

        basis = basisfunc(Lx=Lx, Ly=Ly, kyblock=0)
        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, basis=basis)
        s = np.arange(L)
        x = s % Lx
        y = s // Lx
        permy = x + Lx * ((y + 1) % Ly)
        H_kblock = H.copy()
        for i in range(1,Ly):
            H = H.transform(L=L, perm=permy)
            H_kblock += H
        H_kblock.check_symm(L=L, pauli=True, basis=basis) 


        basis = basisfunc(Lx=Lx, Ly=Ly, kxblock=0, kyblock=0)
        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, basis=basis)
        s = np.arange(L)
        x = s % Lx
        y = s // Lx
        permx = (x + 1) % Lx + Lx * y
        permy = x + Lx * ((y + 1) % Ly)
        H_kblock = H.copy()
        for i in range(1,Lx):
            H = H.transform(L=L, perm=permx)
            H_kblock += H
        H = H_kblock.copy()
        for i in range(1,Ly):
            H = H.transform(L=L, perm=permy)
            H_kblock += H
        H_kblock.check_symm(L=L, pauli=True, basis=basis) 
         
        H = self._generate_random_hamiltonian_Nup(L)
        basis = basisfunc(Lx=Lx, Ly=Ly, Nup=1, kxblock=0)
        with self.assertRaises(ValueError):
            H.check_symm(L=L, pauli=True, basis=basis)
        H_kblock = H.copy()
        for i in range(1,Lx):
            H = H.transform(L=L, perm=permx)
            H_kblock += H
        H_kblock.check_symm(L=L, pauli=True, basis=basis)
    
    def test_spinbasissuper(self):
        L = 5

        H = self._generate_random_hamiltonian(L)
        lind_ops = [np.random.randn() * op.x(i) + np.random.randn() * op.y(i) + np.random.randn() * op.z(i) for i in range(L)]
        Liou = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        
        basisfunc = qt.generate.basis.spin_super_basis
        basis = basisfunc(L=L, pblock=0)
        with self.assertRaises(ValueError):
            Liou.check_symm(L=L, pauli=True, basis=basis)
        H = H + H.expandn(to='pm').transform(L=L, perm='p')
        lind_ops = [L_op + L_op.expandn(to='pm').transform(L=L, perm='p') for L_op in lind_ops] 
        Liou = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        Liou.check_symm(L=L, pauli=True, basis=basis)

        basis = basisfunc(L=L, Nup=L)
        with self.assertRaises(ValueError):
            Liou.check_symm(L=L, pauli=True, basis=basis)
        H = self._generate_random_hamiltonian_Nup(L)
        lind_ops = [np.random.randn() * op.z(i) for i in range(L)]
        Liou_Nup = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        Liou_Nup.check_symm(L=L, pauli=True, basis=basis)

        basis = basisfunc(L=L, Nup=L, Ndiff=1)
        with self.assertRaises(ValueError):
            Liou.check_symm(L=L, pauli=True, basis=basis)
        Liou_Nup.check_symm(L=L, pauli=True, basis=basis)

        basis = basisfunc(L=L, Ndiff=1)
        with self.assertRaises(ValueError):
            Liou.check_symm(L=L, pauli=True, basis=basis)
        Liou_Nup.check_symm(L=L, pauli=True, basis=basis)

       

    def test_qs_spinbasis(self):
        L = 10
        H = self._generate_random_hamiltonian(L)
        basisfunc = qs.spin_basis

        basis = basisfunc(L=L, pauli=True, kblock=0)
        with self.assertRaises(ValueError):
            qs.check_symm(H, basis)
        H_kblock = H.copy()
        for i in range(1,L):
            H = H.transform(L=L, perm='k')
            H_kblock += H
        qs.check_symm(H_kblock, basis)

        basis = basisfunc(L=L, pauli=True, pblock=1)
        with self.assertRaises(ValueError):
            qs.check_symm(H, basis)
        H_kblock = H + H.transform(L=L, perm='p')
        qs.check_symm(H_kblock, basis)

        basis = basisfunc(L=L, pauli=True, zblock=1)
        with self.assertRaises(ValueError):
            qs.check_symm(H, basis)
        H_kblock = H + H.expandn(to='pm').transform(L=L, perm='z')
        qs.check_symm(H_kblock, basis)

        basis = basisfunc(L=L, pauli=True, kblock=0, pblock=1)
        with self.assertRaises(ValueError):
            qs.check_symm(H, basis)
        H_kblock = H.copy()
        for i in range(1,L):
            H = H.transform(L=L, perm='k')
            H_kblock += H
        H_kblock = H_kblock + H_kblock.transform(L=L, perm='p')
        qs.check_symm(H_kblock, basis)

            
        H = self._generate_random_hamiltonian_Nup(L)
        basis = basisfunc(L=L, pauli=True, Nup=1, kblock=0)
        with self.assertRaises(ValueError):
            qs.check_symm(H, basis)
        H_kblock = H.copy()
        for i in range(1,L):
            H = H.transform(L=L, perm='k')
            H_kblock += H
        qs.check_symm(H_kblock, basis)
    
    def test_qs_spinbasis2d(self):
        Lx, Ly = 3,4
        L = Lx * Ly
        H = self._generate_random_hamiltonian(L)
        basisfunc = qs.spin_basis_2d

        basis = basisfunc(Lx=Lx, Ly=Ly, pauli=True, kxblock=0)
        with self.assertRaises(ValueError):
            qs.check_symm(H, basis)
        s = np.arange(L)
        x = s % Lx
        y = s // Lx
        permx = (x + 1) % Lx + Lx * y
        H_kblock = H.copy()
        for i in range(1,Lx):
            H = H.transform(L=L, perm=permx)
            H_kblock += H
        qs.check_symm(H_kblock, basis)

        basis = basisfunc(Lx=Lx, Ly=Ly, pauli=True, kyblock=0)
        with self.assertRaises(ValueError):
            # H.check_symm(L=L, pauli=True, basis=basis)
            qs.check_symm(H, basis)
        s = np.arange(L)
        x = s % Lx
        y = s // Lx
        permy = x + Lx * ((y + 1) % Ly)
        H_kblock = H.copy()
        for i in range(1,Ly):
            H = H.transform(L=L, perm=permy)
            H_kblock += H
        qs.check_symm(H_kblock, basis)


        basis = basisfunc(Lx=Lx, Ly=Ly, kxblock=0, kyblock=0, pauli=True)
        with self.assertRaises(ValueError):
            qs.check_symm(H, basis)
        s = np.arange(L)
        x = s % Lx
        y = s // Lx
        permx = (x + 1) % Lx + Lx * y
        permy = x + Lx * ((y + 1) % Ly)
        H_kblock = H.copy()
        for i in range(1,Lx):
            H = H.transform(L=L, perm=permx)
            H_kblock += H
        H = H_kblock.copy()
        for i in range(1,Ly):
            H = H.transform(L=L, perm=permy)
            H_kblock += H
        qs.check_symm(H_kblock, basis)
         
        H = self._generate_random_hamiltonian_Nup(L)
        basis = basisfunc(Lx=Lx, Ly=Ly, Nup=1, kxblock=0, pauli=True)
        with self.assertRaises(ValueError):
            qs.check_symm(H, basis)
        H_kblock = H.copy()
        for i in range(1,Lx):
            H = H.transform(L=L, perm=permx)
            H_kblock += H
        qs.check_symm(H_kblock, basis)
  

    def test_qs_spinbasissuper(self):
        L = 5

        H = self._generate_random_hamiltonian(L)
        lind_ops = [np.random.randn() * op.x(i) + np.random.randn() * op.y(i) + np.random.randn() * op.z(i) for i in range(L)]
        Liou = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        
        basisfunc = qs.spin_super_basis
        basis = basisfunc(N=L, pblock=0, pauli=True)
        with self.assertRaises(ValueError):
            # Liou.check_symm(L=L, pauli=True, basis=basis)
            qs.check_symm(Liou, basis)
        H = H + H.expandn(to='pm').transform(L=L, perm='p')
        lind_ops = [L_op + L_op.expandn(to='pm').transform(L=L, perm='p') for L_op in lind_ops] 
        Liou = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        qs.check_symm(Liou, basis)

        basis = basisfunc(N=L, Np=L, pauli=True)
        with self.assertRaises(ValueError):
            qs.check_symm(Liou, basis)
        H = self._generate_random_hamiltonian_Nup(L)
        lind_ops = [np.random.randn() * op.z(i) for i in range(L)]
        Liou_Nup = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        qs.check_symm(Liou_Nup, basis)

        basis = basisfunc(N=L, Nd=L, pauli=True)
        with self.assertRaises(ValueError):
            qs.check_symm(Liou, basis)
        H = self._generate_random_hamiltonian_Nup(L)
        lind_ops = [np.random.randn() * op.z(i) for i in range(L)]
        Liou_Nup = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        qs.check_symm(Liou_Nup, basis)

        basis = basisfunc(N=L, Np=L, Nd=1, pauli=True)
        with self.assertRaises(ValueError):
            qs.check_symm(Liou, basis)
        H = self._generate_random_hamiltonian_Nup(L)
        lind_ops = [np.random.randn() * op.z(i) for i in range(L)]
        Liou_Nup = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        qs.check_symm(Liou_Nup, basis)
        
 
    def test_qs_spinbasissuper(self):
        L = 5
        H = self._generate_random_hamiltonian(L)
        res = H.symmetry(L=L, pauli=True)
        self.assertTrue(res == [])

        H_block = H.copy()
        for k in range(1, L):
            H = H.transform(L=L, perm='k')
            H_block += H
        H_block.check_symm(L=L, pauli=True, blocks='k')
        res = H_block.symmetry(L=L, pauli=True)
        self.assertTrue(['kblock'] == res)

        H_block = H_block + H_block.transform(L=L, perm='p')
        res = H_block.symmetry(L=L, pauli=True)
        self.assertTrue(set(['kblock', 'pblock']) == set(res))

        H = self._generate_random_hamiltonian_Nup(L)
        res = H.symmetry(L=L, pauli=True)
        self.assertTrue(res == ["Nup"])


        H = self._generate_random_hamiltonian(L)
        lind_ops = [np.random.randn() * op.x(i) + np.random.randn() * op.y(i) + np.random.randn() * op.z(i) for i in range(L)]
        Liou = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        res = Liou.symmetry(pauli=True)
        self.assertTrue(res == [])

        H = H + H.expandn(to='pm').transform(L=L, perm='p')
        lind_ops = [L_op + L_op.expandn(to='pm').transform(L=L, perm='p') for L_op in lind_ops] 
        Liou = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)
        res = Liou.symmetry(pauli=True)
        self.assertTrue(['pblock'] == res)

        H = self._generate_random_hamiltonian_Nup(L)
        lind_ops = [np.random.randn() * op.z(i) for i in range(L)]
        Liou_Nup = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)       
        res = Liou_Nup.symmetry(pauli=True)
        self.assertTrue(set(['Nup', "Ndiff"]) == set(res))

        H = self._generate_random_hamiltonian_Nup(L)
        lind_ops = [np.random.randn() * op.z(i) for i in range(L)]
        H = H + H.expandn(to='pm').transform(L=L, perm='p')
        lind_ops = [L_op + L_op.expandn(to='pm').transform(L=L, perm='p') for L_op in lind_ops]
        Liou_Nup = qt.generate.operas.super_oper.Lindbladian(L=L, ham=H, jump_ops=lind_ops)       
        res = Liou_Nup.symmetry(pauli=True)
        self.assertTrue(set(['Nup', "pblock", "Ndiff"]) == set(res))

if __name__ == '__main__':
    unittest.main()
