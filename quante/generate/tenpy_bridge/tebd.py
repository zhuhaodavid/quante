# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-04 10:42:20
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-05 20:52:24

import tenpy
from tenpy.models import CouplingMPOModel, NearestNeighborModel, Chain
from tenpy.networks import SpinHalfSite, OnsiteTerms, CouplingTerms
from ..operas.spin import SpinOper
from typing import Literal
from tqdm import tqdm
from warnings import warn

__all__ = [
    'tenpy_tebd_model', 'tenpy_tebd_params_imag_time',
    'tenpy_tebd_params_real_time', 'tenpy_tebd_GS_', 'tenpy_tebd_',
]

class TenpyTEBDModel(CouplingMPOModel, NearestNeighborModel):
    """A TenPy model for the TEBD algorithm.

    This class is designed to work with the TEBD algorithm in TenPy.

    It initializes the model parameters, including the lattice, operator,
    boundary conditions, and conservation laws. It also sets up the onsite
    and coupling terms based on the provided operator.

    Parameters
    ----------
    model_params : dict
        A dictionary containing the model parameters. The following keys are expected:
        - 'L' : int
            The length of the chain.
        - 'oper' : OperSpin
            The operator defining the Hamiltonian.
        - 'bc_MPS' : str
            The boundary condition for the MPS, either 'finite' or 'periodic'.
        - 'conserve' : str or None
            The conservation law, can be 'parity', 'Sz', or None.
        - 'pauli' : bool
            Whether to use Pauli matrices for the operators.
        - 'explicit_plus_hc' : bool
            Whether to explicitly include the Hermitian conjugate terms in the Hamiltonian.
        - 'lattice' : class
            The lattice class to use, default is Chain.
        - 'random_seed' : int
            Random seed for reproducibility.
        - 'order' : str
            The order of the TEBD algorithm, default is 'default'.
        - 'sort_charge' : bool
            Whether to sort the charge of the sites.
        - 'bc_x' : str
            The boundary condition for the x-direction, either 'open' or 'periodic'.
        - 'helical' : None or str
            Whether to use helical boundary conditions, default is None.
        - 'irregular_remove' : None or str
            Whether to remove irregular sites, default is None.
        - 'sort_mpo_legs' : bool
            Whether to sort the legs of the MPO, default is False.
    """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'None', str)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):

        oper = model_params.get('oper', None, None)
        pauli = model_params.get('pauli', False, bool)
        oper._check_pauli(pauli)

        if any(o not in 'IpmZxyz' for opnm in oper.data for o in opnm):
            oper = oper.expandxy(pauli=pauli)
            warn(
                "Operator contains unsupported characters, "
                "expanding to Pauli operators."
            )
        
        Sx = 'Sigmax' if pauli else 'Sx'
        Sy = 'Sigmay' if pauli else 'Sy'
        Sz = 'Sigmaz' if pauli else 'Sz'
        name_map = {'I': 'Id', 'p': 'Sp', 'm': 'Sm', 'Z': 'Sigmaz',
                    'x': Sx, 'y': Sy, 'z': Sz}

        for opnm, pos_strength in oper.data.items():
            if len(opnm) == 1:
                # single-site operator
                tenpy_opname = name_map.get(opnm, None)
                assert tenpy_opname is not None, "Unknown operator name: {}".format(opnm)
                category = tenpy_opname
                ot = self.onsite_terms.setdefault(category, OnsiteTerms(self.lat.N_sites))
                for pos, strength in zip(*pos_strength):
                    ot.add_onsite_term(strength, pos[0], tenpy_opname)
            elif len(opnm) == 2:
                # two-site operator
                op1 = name_map.get(opnm[0], None)
                op2 = name_map.get(opnm[1], None)
                assert op1 is not None and op2 is not None, "Unknown operator name: {}".format(opnm)
                category = "{op1}_i {op2}_j".format(op1=op1, op2=op2)
                ct = self.coupling_terms.setdefault(category, CouplingTerms(self.lat.N_sites))
                for pos, strength in zip(*pos_strength):
                    i, j = pos
                    assert abs(i-j) == 1, "Only nearest-neighbor couplings are supported, got: {}".format(pos)
                    if i < j:
                        o1, o2 = op1, op2
                    else:
                        i, j = j, i
                        o1, o2 = op2, op1
                    ct.add_coupling_term(strength, i, j, o1, o2, 'Id')
            else:
                raise ValueError("Only single-site and two-site operators are supported, got: {}".format(opnm))


def tenpy_tebd_model(
    L: int,
    oper: SpinOper,  
    pauli: bool = False,
    conserve: Literal['Sz', 'parity', 'None'] = 'None',
    bc_MPS: Literal['finite', 'periodic'] = 'finite',
    **kwargs
):
    """Create a TenPy TEBD model.

    Parameters
    ----------
    L : int
        The length of the chain.
    oper : OperSpin
        The operator defining the Hamiltonian.
    pauli : bool, optional
        Whether to use Pauli matrices for the operators. Default is False.
    conserve : str, optional
        The conservation law, can be 'parity', 'Sz', or None. Default is None.
    bc_MPS : str, optional
        The boundary condition for the MPS, either 'finite' or 'periodic'. Default is 'finite'.
    **kwargs : dict, optional
        Additional parameters for the model, such as: 
        - explicit_plus_hc : bool, optional
            Whether to explicitly include the Hermitian conjugate terms in the Hamiltonian. Default is True.
        - lattice : type, optional
            The lattice class to use, default is Chain.
        - random_seed : int, optional
            Random seed for reproducibility. Default is None.
        - order : str, optional
            The order of the TEBD algorithm, default is 'default'.
        - sort_charge : bool, optional
            Whether to sort the charge of the sites. Default is True.
        - bc_x : str, optional
            The boundary condition for the x-direction, either 'open' or 'periodic'. Default is 'open'.
        - helical : str, optional
            Whether to use helical boundary conditions, default is None.
        - irregular_remove : str, optional
            Whether to remove irregular sites, default is None.
        - sort_mpo_legs : bool, optional
            Whether to sort the legs of the MPO, default is False.
    
    Returns
    -------
    TenpyTEBDModel
        An instance of the TenpyTEBDModel class, which is a CouplingMPOModel with SpinHalfSite as the site type.
    """
    model_params = {
        'L': L,
        'oper': oper,
        'pauli': pauli,
        'conserve': conserve,
        'bc_MPS': bc_MPS,
    }
    model_params.update(kwargs)
    return TenpyTEBDModel(model_params)


def tenpy_tebd_params_imag_time(
    delta_tau_list: list,
    order: int = 2,   
    N_steps: int = 10,
    max_error_E: float = 1.e-13,
    chi_max: int = 100,
    svd_min: float = 1.e-14,
    chi_min: int = None,
    degeneracy_tol: float = None,
    trunc_cut: float = 1.e-14,
    **kwargs
):
    """Create a dictionary of parameters for the TenPy TEBD algorithm.

    ``U_bond = exp(- dt H_bond)`` for ``type_evo='imag'``.
    
    Parameters
    ----------
    delta_tau_list : list
        A list of floats: the timesteps to be used.
        Choosing a large timestep `delta_tau` introduces large (Trotter) errors,
        but a too small time step requires a lot of steps to reach
        ``exp(-tau H) --> |psi0><psi0|``.
        Therefore, we start with fairly large time steps for a quick time evolution until
        convergence, and then gradually decrease the time step.
    order : int, optional
        Order of the Suzuki-Trotter decomposition. Default is 2.
        The total error for evolution up to a fixed time `t`,
        scales as ``O(t*dt^order)``.
    N_steps : int, optional
        Number of steps before measurement can be performed. Default is 10.
    max_error_E : float, optional
        Threshold for the maximum error in energy expectation value.
        Default is 1.e-13.
    chi_max : int, optional
        Keep at most `chi_max` Schmidt values. Default is 100.
    svd_min : float, optional
        Discard all small Schmidt values ``S[i] < svd_min``. Default is 1.e-14.
    chi_min : int, optional
        Keep at least `chi_min` Schmidt values. Default is None.
    degeneracy_tol : float, optional
        Don't cut between neighboring Schmidt values with
            ``|log(S[i]/S[j])| < degeneracy_tol``, or equivalently
            ``|S[i] - S[j]|/S[j] < exp(degeneracy_tol) - 1 ~= degeneracy_tol``
            for small `degeneracy_tol`.
            In other words, keep either both `i` and `j` or none, if the
            Schmidt values are degenerate with a relative error smaller
            than `degeneracy_tol`, which we expect to happen in the case
            of symmetries. Default is None.
    trunc_cut : float, optional
        Discard all small Schmidt values as long as
            ``sum_{i discarded} S[i]**2 <= trunc_cut**2``. Default is 1.e-14.
    **kwargs : dict, optional
        Additional parameters for the TEBD algorithm, such as:
        - start_time : float, optional
            The initial time for the evolution. Default is 0.
        - start_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`, optional
            Initial truncation error for :attr:`trunc_err`. Default is None.
        - max_delta_t : float, optional
            Threshold for raising errors on too large time steps. Default is 1.0.
            The trotterization in the time evolution operator assumes that the time step is small.
            We raise an error if it is not. Can be downgraded to a warning by setting this option to None.
        - max_N_sites_per_ring : int, optional
            Threshold for raising errors on too many sites per ring. Default is 18.
            See :meth:`~tenpy.tools.misc.consistency_check`.
            In a higher-dimensional geometry, the area law implies that the entropy of a bipartition
            is linear in ``N_sites_per_ring`` and thus the required bond dimension is exponential
            in it. This makes MPS simulations with large ``N_sites_per_ring`` unfeasible.
            If it is too large, you will not be able to choose a reasonably large bond dimension
            *and* have enough RAM to do the simulation. We raise an error in that case.
            Can be downgraded to a warning by setting this option to None.

    Returns
    -------
    dict
        A dictionary containing the TEBD parameters. The following keys are expected:
    """
    trunc_params = {
        'chi_max': chi_max,
        'svd_min': svd_min,
        'chi_min': chi_min,
        'degeneracy_tol': degeneracy_tol,
        'trunc_cut': trunc_cut,
    }
    tebd_params = {
        'delta_tau_list': delta_tau_list,
        'order': order,
        'N_steps': N_steps,
        'max_error_E': max_error_E,
        'trunc_params': trunc_params,
    }
    tebd_params.update(kwargs)
    return tebd_params

def tenpy_tebd_params_real_time(
    dt: float,
    N_steps: int = 5,
    order: int = 2,   
    E_offset: float | list[float] = None,
    preserve_norm: bool = None,
    chi_max: int = 100,
    svd_min: float = 1.e-14,
    chi_min: int = None,
    degeneracy_tol: float = None,
    trunc_cut: float = 1.e-14,
    **kwargs
):
    """Create a dictionary of parameters for the TenPy TEBD algorithm.
    
    ``U_bond = exp(-i dt (H_bond-E_offset_bond))``

    Parameters
    ----------
    dt : float
        Minimal time step by which to evolve.
    N_steps : int = 5
        Number of time steps `dt` to evolve by in :meth:`run`.
        Adjusting `dt` and `N_steps` at the same time allows to keep the evolution time
        done in :meth:`run` fixed.
        Further, e.g., the Trotter decompositions of order > 1 are slightly more efficient
        if more than one step is performed at once.
    order : int = 2
        Order of the algorithm. The total error for evolution up to a fixed time `t`
            scales as ``O(t*dt^order)``.
    E_offset : None | list of float = None
        Possible offset added to `H_bond` for real-time evolution.
    preserve_norm : bool = None
        Whether the state will be normalized to its initial norm after each time step.
        Per default, this is ``False`` for real time evolution and ``True`` for imaginary time.
    chi_max : int, optional
        Keep at most `chi_max` Schmidt values. Default is 100.
    svd_min : float, optional
        Discard all small Schmidt values ``S[i] < svd_min``. Default is 1.e-14.
    chi_min : int, optional
        Keep at least `chi_min` Schmidt values. Default is None.
    degeneracy_tol : float, optional
        Don't cut between neighboring Schmidt values with
            ``|log(S[i]/S[j])| < degeneracy_tol``, or equivalently
            ``|S[i] - S[j]|/S[j] < exp(degeneracy_tol) - 1 ~= degeneracy_tol``
            for small `degeneracy_tol`.
            In other words, keep either both `i` and `j` or none, if the
            Schmidt values are degenerate with a relative error smaller
            than `degeneracy_tol`, which we expect to happen in the case
            of symmetries. Default is None.
    trunc_cut : float, optional
        Discard all small Schmidt values as long as
            ``sum_{i discarded} S[i]**2 <= trunc_cut**2``. Default is 1.e-14.
    **kwargs : dict, optional
        - start_time : float = 0.0
            Initial value for :attr:`evolved_time`.
        - start_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            Initial truncation error for :attr:`trunc_err`.
        - max_delta_t : float | None = 1.0
            Threshold for raising errors on too large time steps. Default ``1.0``.
            The trotterization in the time evolution operator assumes that the time step is small.
            We raise an error if it is not.
            Can be downgraded to a warning by setting this option to ``None``.
        - max_N_sites_per_ring: int = 18
            Threshold for raising errors on too many sites per ring. Default is 18.
                See :meth:`~tenpy.tools.misc.consistency_check`.
                In a higher-dimensional geometry, the area law implies that the entropy of a bipartition
                is linear in ``N_sites_per_ring`` and thus the required bond dimension is exponential
                in it. This makes MPS simulations with large ``N_sites_per_ring`` unfeasible.
                If it is too large, you will not be able to choose a reasonably large bond dimension
                *and* have enough RAM to do the simulation. We raise an error in that case.
                Can be downgraded to a warning by setting this option to None.
    """
    trunc_params = {
        'chi_max': chi_max,
        'svd_min': svd_min,
        'chi_min': chi_min,
        'degeneracy_tol': degeneracy_tol,
        'trunc_cut': trunc_cut,
    }
    tebd_params = {
        'dt': dt,
        'N_steps': N_steps,
        'order': order,
        'E_offset': E_offset,
        'preserve_norm': preserve_norm,
        'trunc_params': trunc_params,
    }
    tebd_params.update(kwargs)
    return tebd_params


def tenpy_tebd_GS_(psi, M, tebd_params):
    """Perform imaginary time evolution using the TenPy TEBD algorithm.

    This function returns a TenPy TEBD model and the final state after running the TEBD algorithm.
    
    The bond energies of the final state can be accessed via the `bond_energies` method of the model.
    >>> print(M.bond_energies(psi))

    Parameters
    ----------
    psi : tenpy.networks.MPS
        The initial state as a Matrix Product State (MPS).
    M : TenpyTEBDModel
        The TenPy TEBD model initialized with the given parameters.
    tebd_params : dict
        A dictionary containing the TEBD parameters for the imaginary time evolution.
    
    Returns
    -------
    M : TenpyTEBDModel
        The TenPy TEBD model initialized with the given parameters.
    psi : tenpy.networks.MPS
        The final state after running the TEBD algorithm.
    """
    eng = tenpy.algorithms.tebd.TEBDEngine(psi, M, tebd_params)
    eng.run_GS()
    return eng


def tenpy_tebd_(psi, M, final_time, tebd_params, progressbar=False):
    """Perform real time evolution using the TenPy TEBD algorithm.

    This function updates the state `psi` by running the TEBD algorithm

    Parameters
    ----------
    psi : tenpy.networks.MPS
        The initial state as a Matrix Product State (MPS).
    M : TenpyTEBDModel
        The TenPy TEBD model initialized with the given parameters.
    final_time : float
        The final time for the evolution.
    tebd_params : dict
        A dictionary containing the TEBD parameters for the imaginary time evolution.
    
    Yields
    ------
    t : float
        The current time after each evolution step.
    psi : tenpy.networks.MPS
        The updated state after each evolution step.
    
    Example
    -------
    >>> for t, psi in tenpy_tebd_(psi, M, final_time, tebd_params):
    >>>     print(f"Time: {t}, State: {psi}")
    """
    dt_measure = tebd_params['dt'] * tebd_params['N_steps']
    start_time = tebd_params.get('start_time', 0.0)
    eng = tenpy.algorithms.tebd.TEBDEngine(psi, M, tebd_params)
    Delta_t = final_time - start_time
    Ns = int(Delta_t / dt_measure + 0.5)
    t_iter = tqdm(range(Ns), ascii=True) if progressbar else range(Ns)
    for _ in t_iter:
        eng.run()
        t = eng.evolved_time.real
        yield t, psi
