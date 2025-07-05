# # -*- coding: utf-8 -*-
# # @Author: hzhu
# # @Date:   2025-06-26 17:28:00
# # @Last Modified by:   hzhu
# # @Last Modified time: 2025-07-05 22:11:16

# import numpy as np

# from tenpy.networks.mps import MPS
# from tenpy.models.tf_ising import TFIChain
# from tenpy.models import SpinModel
# from tenpy.algorithms import dmrg

# import quante as qt
# op = qt.generate.operas.spin

# from quante.generate.tenpy_bridge.dmrg import tenpy_mpo_model, tenpy_dmrg_params, tenpy_dmrg
# from quante.generate.tenpy_bridge.convert import set_tenpy_logging

# import dowhen as dw

# def example_DMRG_tf_ising_finite(L, g):
#     print("finite DMRG, transverse field Ising model")
#     print("L={L:d}, g={g:.2f}".format(L=L, g=g))

#     # model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
#     # M = TFIChain(model_params)

#     ham = - op.sum(op.xx(i, i+1) for i in range(L-1)) - g * op.sum(op.z(i) for i in range(L)) + op.xx(1,3)
#     M = tenpy_mpo_model(
#         L=L, oper=ham, pauli=True, conserve='None', bc_MPS='finite',
#     )

#     product_state = ["up"] * M.lat.N_sites
#     psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

#     # dmrg_params = {
#     #     'mixer': None,  # setting this to True helps to escape local minima
#     #     'trunc_params': {
#     #         'chi_max': 30,
#     #         'svd_min': 1.e-10
#     #     },
#     #     'combine': True
#     # }
#     dmrg_params = tenpy_dmrg_params(
#         active_sites=2,
#         chi_max=30,
#         svd_min=1.e-10,
#     )

#     # info = dmrg.run(psi, M, dmrg_params)  # the main work...
#     E, _, _ = tenpy_dmrg(psi, M, dmrg_params)  # the main work...

#     print("E = {E:.13f}".format(E=E))
#     print("final bond dimensions: ", psi.chi)
#     mag_x = np.sum(psi.expectation_value("Sigmax"))
#     mag_z = np.sum(psi.expectation_value("Sigmaz"))
#     print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
#     print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
#     if L < 20:  # compare to exact result
#         import quante as qt
#         E_exact = ham.gdenergy(pauli=True)
#         print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
#         print("relative error: ", abs((E - E_exact) / E_exact))
#     return E, psi, M



# def example_1site_DMRG_tf_ising_finite(L, g):
#     print("single-site finite DMRG, transverse field Ising model")
#     print("L={L:d}, g={g:.2f}".format(L=L, g=g))
#     # model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
#     # M = TFIChain(model_params)

#     ham = - op.sum(op.xx(i, i+1) for i in range(L-1)) - g * op.sum(op.z(i) for i in range(L))
#     M = tenpy_mpo_model(
#         L=L, oper=ham, pauli=True, conserve='None', bc_MPS='finite',
#     )

#     product_state = ["up"] * M.lat.N_sites
#     psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

#     dmrg_params = tenpy_dmrg_params(
#         active_sites=1,
#         max_E_err=1.e-10,
#         chi_max=30,
#         svd_min=1.e-10,
#         mixer=True,
#         combine=False,
#     )
    
#     dw.do('print(engine.options["mixer_params"])').when(dmrg.run, 'return {') 
#     # info = dmrg.run(psi, M, dmrg_params)
#     # E = info['E']
#     E, psi, _ = tenpy_dmrg(psi, M, dmrg_params)  # the main work...
#     print("E = {E:.13f}".format(E=E))
#     print("final bond dimensions: ", psi.chi)
#     mag_x = np.sum(psi.expectation_value("Sigmax"))
#     mag_z = np.sum(psi.expectation_value("Sigmaz"))
#     print("magnetization in X = {mag_x:.5f}".format(mag_x=mag_x))
#     print("magnetization in Z = {mag_z:.5f}".format(mag_z=mag_z))
#     if L < 20:  # compare to exact result
#         # from tfi_exact import finite_gs_energy
#         # E_exact = finite_gs_energy(L, 1., g)
#         E_exact = ham.gdenergy(pauli=True)
#         print("Exact diagonalization: E = {E:.13f}".format(E=E_exact))
#         print("relative error: ", abs((E - E_exact) / E_exact))
#     return E, psi, M


# def example_DMRG_tf_ising_infinite(g):
#     print("infinite DMRG, transverse field Ising model")
#     print("g={g:.2f}".format(g=g))
#     # model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None)
#     # M = TFIChain(model_params)

#     op = qt.generate.operas.spin
#     L = 2
#     ham = - op.sum(op.xx(i, i+1) for i in range(L)) - g * op.sum(op.z(i) for i in range(L))
#     M = tenpy_mpo_model(
#         L=L, oper=ham, pauli=True, conserve='None', bc_MPS='infinite',
#     )
   

#     product_state = ["up"] * M.lat.N_sites
#     psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

#     dmrg_params = tenpy_dmrg_params(
#         mixer=True,
#         chi_max=30,
#         svd_min=1.e-10,
#         max_E_err=1.e-10,
#     )
#     # dmrg_params = {
#     #     'mixer': True,  # setting this to True helps to escape local minima
#     #     'trunc_params': {
#     #         'chi_max': 30,
#     #         'svd_min': 1.e-10
#     #     },
#     #     'max_E_err': 1.e-10,
#     # }
#     # Sometimes, we want to call a 'DMRG engine' explicitly
#     # eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
#     # E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.

#     E, psi, _ = tenpy_dmrg(psi, M, dmrg_params)  # the main work...

#     print("E = {E:.13f}".format(E=E))
#     print("final bond dimensions: ", psi.chi)
#     mag_x = np.mean(psi.expectation_value("Sigmax"))
#     mag_z = np.mean(psi.expectation_value("Sigmaz"))
#     print("<sigma_x> = {mag_x:.5f}".format(mag_x=mag_x))
#     print("<sigma_z> = {mag_z:.5f}".format(mag_z=mag_z))
#     print("correlation length:", psi.correlation_length())
#     # compare to exact result
#     # from tfi_exact import infinite_gs_energy
#     # E_exact = infinite_gs_energy(1., g)
#     E_exact = qt.generate.operas.spin.heisenberg_operator(L=np.inf, j=(-1.,0,0), h=(0,0,-g)).gdenergy(pauli=True)
#     print("Analytic result: E (per site) = {E:.13f}".format(E=E_exact))
#     print("relative error: ", abs((E - E_exact) / E_exact))
#     return E, psi, M


# def example_1site_DMRG_tf_ising_infinite(g):
#     print("single-site infinite DMRG, transverse field Ising model")
#     print("g={g:.2f}".format(g=g))
#     # model_params = dict(L=2, J=1., g=g, bc_MPS='infinite', conserve=None)
#     # M = TFIChain(model_params)
#     L = 2
#     ham = - op.sum(op.xx(i, i+1) for i in range(L)) - g * op.sum(op.z(i) for i in range(L))
#     M = tenpy_mpo_model(
#         L=L, oper=ham, pauli=True, conserve='None', bc_MPS='infinite',
#     )

#     product_state = ["up"] * M.lat.N_sites
#     psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
#     # dmrg_params = {
#     #     'mixer': True,  # setting this to True is essential for the 1-site algorithm to work.
#     #     'trunc_params': {
#     #         'chi_max': 30,
#     #         'svd_min': 1.e-10
#     #     },
#     #     'max_E_err': 1.e-10,
#     #     'combine': True
#     # }
#     dmrg_params = tenpy_dmrg_params(
#         active_sites=1,
#         mixer=True,
#         chi_max=30,
#         svd_min=1.e-10,
#         max_E_err=1.e-10,
#         combine=True, 
#     )
    
#     # eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params)
#     # E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
#     E, psi, _ = tenpy_dmrg(psi, M, dmrg_params)  # the main work...

#     print("E = {E:.13f}".format(E=E))
#     print("final bond dimensions: ", psi.chi)
#     mag_x = np.mean(psi.expectation_value("Sigmax"))
#     mag_z = np.mean(psi.expectation_value("Sigmaz"))
#     print("<sigma_x> = {mag_x:.5f}".format(mag_x=mag_x))
#     print("<sigma_z> = {mag_z:.5f}".format(mag_z=mag_z))
#     print("correlation length:", psi.correlation_length())
#     # compare to exact result
#     # from tfi_exact import infinite_gs_energy
#     # E_exact = infinite_gs_energy(1., g)
#     E_exact = qt.generate.operas.spin.heisenberg_operator(L=np.inf, j=(-1.,0,0), h=(0,0,-g)).gdenergy(pauli=True)
#     print("Analytic result: E (per site) = {E:.13f}".format(E=E_exact))
#     print("relative error: ", abs((E - E_exact) / E_exact))


# def example_DMRG_heisenberg_xxz_infinite(Jz, conserve='best'):
#     print("infinite DMRG, Heisenberg XXZ chain")
#     print("Jz={Jz:.2f}, conserve={conserve!r}".format(Jz=Jz, conserve=conserve))
#     model_params = dict(
#         L=2,
#         S=0.5,  # spin 1/2
#         Jx=1.,
#         Jy=1.,
#         Jz=Jz,  # couplings
#         bc_MPS='infinite',
#         conserve=conserve)
#     M = SpinModel(model_params)
#     print(M.coupling_terms['Sp_i Sm_j'].coupling_terms)

#     L = 2
#     ham = op.sum(0.5*(op.pm(i, i+1) + op.pm(i+1, i)) + Jz * op.zz(i, i+1) for i in range(L))
#     M = tenpy_mpo_model(
#         L=L, oper=ham, pauli=False, conserve='Sz', bc_MPS='infinite',
#     )
#     print(M.coupling_terms['pm'].coupling_terms)
    

#     product_state = ["up", "down"]  # initial Neel state
#     psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

#     dmrg_params = tenpy_dmrg_params(
#         mixer=True,  # setting this to True helps to escape local minima
#         chi_max=100,
#         svd_min=1.e-10,
#         max_E_err=1.e-10,
#     )
    
#     # dmrg_params = {
#     #     'mixer': True,  # setting this to True helps to escape local minima
#     #     'trunc_params': {
#     #         'chi_max': 100,
#     #         'svd_min': 1.e-10,
#     #     },
#     #     'max_E_err': 1.e-10,
#     # }

#     # info = dmrg.run(psi, M, dmrg_params)
#     # E = info['E']
#     E, _, _ = tenpy_dmrg(psi, M, dmrg_params)  # the main work...

#     print("E = {E:.13f}".format(E=E))
#     print("final bond dimensions: ", psi.chi)
#     Sz = psi.expectation_value("Sz")  # Sz instead of Sigma z: spin-1/2 operators!
#     mag_z = np.mean(Sz)
#     print("<S_z> = [{Sz0:.5f}, {Sz1:.5f}]; mean ={mag_z:.5f}".format(Sz0=Sz[0],
#                                                                      Sz1=Sz[1],
#                                                                      mag_z=mag_z))
#     # note: it's clear that mean(<Sz>) is 0: the model has Sz conservation!
#     print("correlation length:", psi.correlation_length())
#     corrs = psi.correlation_function("Sz", "Sz", sites1=range(10))
#     print("correlations <Sz_i Sz_j> =")
#     print(corrs)
#     return E, psi, M


# if __name__ == "__main__":
#     set_tenpy_logging(level=1, savelog=True)
#     from quante.basicfun import logger
#     logger.warning("hello, this is a test for tenpy bridge")
#     # tenpy_logger = logging.getLogger("tenpy")
#     # tenpy_logger.handlers = qt.basicfun.logger.handlers
#     # tenpy_logger.setLevel(qt.basicfun.logger.level)
#     example_DMRG_tf_ising_finite(L=10, g=1.)
#     # print("-" * 100)
#     # example_1site_DMRG_tf_ising_finite(L=10, g=1.)
#     # print("-" * 100)
#     # example_DMRG_tf_ising_infinite(g=1.5)
#     # print("-" * 100)
#     # example_1site_DMRG_tf_ising_infinite(g=1.5)
#     # print("-" * 100)
#     # example_DMRG_heisenberg_xxz_infinite(Jz=1.5)
