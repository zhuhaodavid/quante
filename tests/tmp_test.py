# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-01 11:19:29
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-21 15:15:02
#

# import quante as qt
# from quante.generate.matrix.models.JR import TriangularLattice

# Lx, Ly = 4, 4
# Jnn = Jnnn = 0.0
# r = 10.0

# tl = TriangularLattice(Lx=Lx, Ly=Ly)
# static = tl.j1j2(j1=Jnn, j2=Jnnn) + tl.cc(r=r)
# ham = qt.generate.operas.SpinOper.from_quspin(static)
# res = ham.to_mpo(backend='tenpy')

# from quante.bridge.tenpy_utils import get_scipy_sparse_Hamiltonian

# mat1 = get_scipy_sparse_Hamiltonian(res)
