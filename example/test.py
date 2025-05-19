# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2024-12-30 14:31:20
# @Last Modified by:   dzwang
# @Last Modified time: 2025-05-19 14:49:33
from re import U
import numpy as np
import quante as qt
from quante.basicfun import println
from quante.generate.basis import spin_basis
import quante.generate.operas.spin as op
dtype = np.float64


L = 10
Jz = 1.
hx = hz = -0.0
Ham = op.sum(Jz*op.zz(i,i+1) for i in range(L-1)) + op.sum(hx*op.x(i) + hz*op.z(i) for i in range(L))
basis = spin_basis(L)

dt = 0.01
hammat = Ham.to_matrix(basis=basis)
Uexact = qt.linalg.expm(hammat, c=-1j*dt)
Utay = np.eye(hammat.shape[0]) - dt*1j/2*hammat - dt**2*1/4*hammat@hammat

println(qt.linalg.norm(Uexact - Utay))


