# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2024-12-30 14:31:20
# @Last Modified by:   dzwang
# @Last Modified time: 2025-01-12 16:43:20
import quante.generate.operas as op
from quante.basicfun import println
from quante.tensor.networks.mps import MPS
from quante.tensor.algorithms.tebd import TEBDEngine


L = 4
T = 0.03
tau = 0.01
N_steps = int(T/tau)

H = op.heisenberg_operator(L, j=1.)
pos, gates = H.trotter_gates(L, tau=0.01, order="2", evolve_type="time")
println(pos)

state = ["spin_up"] * L
psi = MPS.from_product_state(L, state)

eng = TEBDEngine(psi, H)
for n in range(N_steps):
    eng.evolve(pos, gates)
    
    Ds = eng.psi.get_bond_dimension()
    eng.psi.check_mixed_canonical_form()
    println(eng.psi.llim, eng.psi.rlim)
    # eng.psi.set_right_canonical_form()
    # eng.psi.check_right_canonical_form()
    println(eng.psi.llim, eng.psi.rlim)
    println(Ds)
    
println(eng.psi.llim)
shape = eng.psi.Ws[L//2].shape
println(shape)
