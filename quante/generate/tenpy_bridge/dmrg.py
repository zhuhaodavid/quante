# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-26 17:28:00
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-05 20:58:20

import tenpy
from tenpy.models import CouplingMPOModel, NearestNeighborModel, Chain
from tenpy.networks import SpinHalfSite, OnsiteTerms, CouplingTerms
from ..operas.spin import SpinOper
from typing import Literal
from tqdm import tqdm
from warnings import warn

from tenpy.tools.params import asConfig
from tenpy.algorithms.dmrg import SingleSiteDMRGEngine, TwoSiteDMRGEngine


class TenpyMPOModel(CouplingMPOModel):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'None', str)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        oper = model_params.get('oper', None, None)
        pauli = model_params.get('pauli', False, bool)

        if any(o not in 'IpmZxyz' for opnm in oper.data for o in opnm):
            oper = oper.expandxy(pauli=pauli)
            warn(
            "Operator contains unsupported characters, "
            "expanding to Pauli operators."
            )
        
        oper._check_pauli(pauli)
        Sx = 'Sigmax' if pauli else 'Sx'
        Sy = 'Sigmay' if pauli else 'Sy'
        Sz = 'Sigmaz' if pauli else 'Sz'
        name_map = {'I': 'Id', 'p': 'Sp', 'm': 'Sm', 'Z': 'Sigmaz',
                    'x': Sx, 'y': Sy, 'z': Sz}
        for opnm, pos, strength in oper.each_term():
            term = [(name_map[o], [p, 0]) for o, p in zip(opnm, pos)] 
            self.add_local_term(strength, term, category=opnm)


def tenpy_mpo_model(
    L: int,
    oper: SpinOper,  
    pauli: bool = False,
    conserve: Literal['Sz', 'parity', 'None'] = 'None',
    bc_MPS: Literal['finite', 'periodic'] = 'finite',
    **kwargs
):
    """Create a TenPy MPO model.

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
    TenpyMPOModel
        An instance of the TenpyMPOModel class, which is a CouplingMPOModel with SpinHalfSite as the site type.
    """
    model_params = {
        'L': L,
        'oper': oper,
        'pauli': pauli,
        'conserve': conserve,
        'bc_MPS': bc_MPS,
    }
    model_params.update(kwargs)
    return TenpyMPOModel(model_params)


def tenpy_dmrg_params(
    active_sites: int = 2,
    chi_list: dict[int, int] | None = None,
    diag_method: Literal['default', 'lanczos', 'arpack', 'ED_block', 'ED_all'] = 'default',
    mixer: str | type | bool | None = None,
    max_E_err: float = 1.e-8,
    chi_max: int = 100,
    svd_min: float = 1.e-14,
    **kwargs
):
    """
    Parameters
    ----------
    active_sites : int, optional
        Default is 2.
        The number of active sites to be used by DMRG.
        If set to 1, :class:`SingleSiteDMRGEngine` is used.
        If set to 2, DMRG is handled by :class:`TwoSiteDMRGEngine`.
    chi_list : dict[int, int] | None, optional
        By default (``None``) this feature is disabled.
        A dict allows to gradually increase the `chi_max`.
        An entry `at_sweep: chi` states that starting from sweep `at_sweep`,
        the value `chi` is to be used for ``trunc_params['chi_max']``.
        For example ``chi_list={0: 50, 20: 100}`` uses ``chi_max=50`` for the first
        20 sweeps and ``chi_max=100`` afterwards.
        A value of `None` is initialized to the current value of
        ``trunc_params['chi_max']`` at algorithm initialization.
    diag_method : {'default', 'lanczos', 'arpack', 'ED_block', 'ED_all'}, optional
        Default is 'default'.
        One of the following strings:
            - 'default': Same as ``'lanczos'`` for large bond dimensions, but if the
              total dimension of the effective Hamiltonian does not exceed
              the DMRG parameter ``'max_N_for_ED'`` it uses ``'ED_block'``.
            - 'lanczos': :func:`~tenpy.linalg.lanczos.lanczos`
              Default, the Lanczos implementation in TeNPy.
            - 'arpack': :func:`~tenpy.linalg.lanczos.lanczos_arpack`
              Based on :func:`scipy.linalg.sparse.eigsh`.
              Slower than 'lanczos', since it needs to convert the npc arrays
              to numpy arrays during *each* matvec, and possibly does many
              more iterations.
            - 'ED_block': :func:`full_diag_effH`
              Contract the effective Hamiltonian to a (large!) matrix and
              diagonalize the block in the charge sector of the initial state.
              Preserves the charge sector of the explicitly conserved charges.
              However, if you don't preserve a charge explicitly, it can break
              it. For example if you use a ``SpinChain({'conserve': 'parity'})``,
              it could change the total "Sz", but not the parity of 'Sz'.
            - 'ED_all': :func:`full_diag_effH`
              Contract the effective Hamiltonian to a (large!) matrix and
              diagonalize it completely.
              Allows to change the charge sector *even for explicitly
              conserved charges*. For example if you use a
              ``SpinChain({'conserve': 'Sz'})``, it **can** change the total "Sz".
    mixer : str | class | bool | None, optional
        The default is None.
        Specifies which :class:`Mixer` to use, if any.
        A string stands for one of the mixers defined in this module.
        A class is assumed to have the same interface as :class:`Mixer` and is used
        to instantiate the :attr:`mixer`.
        ``None`` uses no mixer.
        ``True`` uses the mixer specified by the :attr:`DefaultMixer` class attribute.
    max_E_err : float, optional
        default is 1.e-8.
        Convergence if the change of the energy in each step
        satisfies ``|Delta E / max(E, 1)| < max_E_err``. Note that
        this might be satisfied even if ``Delta E > 0``,
        i.e., if the energy increases (due to truncation).
    chi_max : int, optional
        Default is 100.
        Keep at most `chi_max` Schmidt values. 
    svd_min : float, optional
        Default is 1.e-14.
        Discard all small Schmidt values ``S[i] < svd_min``. 
    **kwargs : dict, optional
        Additional parameters for the DMRG algorithm. You can pass:
        - N_sweeps_check : int, optional.
            Default is 1 if the `psi` is finite, otherwise 10.
            Number of sweeps to perform between checking convergence
            criteria and giving a status update.
        - min_sweeps : int, optional.
            Default is the maximum of :attr:`chi_list` keys and 
            ``int(1.5 * self.N_sweeps_check)``.
            Minimum number of sweeps to perform.
        - max_sweeps : int, optional.
            Default is 1000.
            Maximum number of sweeps to perform.
        - max_hours : float, optional.
            Defaulit is 24 * 365.
            If the DMRG took longer (measured in wall-clock time),
            'shelve' the simulation, i.e. stop and return with the flag
            ``shelve=True``.
        - max_S_err : float, optional.
            Default is 1.e-5.
            Convergence if the relative change of the entropy in each step
            satisfies ``|Delta S|/S < max_S_err``
        - max_trunc_err : float, optional.
            Default 0.0001.
            Threshold for raising errors on too large truncation errors. 
            See :meth:`~tenpy.tools.misc.consistency_check`.
            If any truncation error :attr:`~tenpy.algorithms.truncation.TruncationError.eps`
            on the final sweep exceeds this value, we raise.
            Can be downgraded to a warning by setting this option to ``None``.
        - max_N_for_ED : int, optional.
            Default is 400.
            Maximum matrix dimension of the effective hamiltonian
            up to which the ``'default'`` `diag_method` uses ED instead of
            Lanczos. 
        - combine : bool. 
            Default is False.
            Whether to combine legs into pipes. This combines the virtual and
            physical leg for the left site (when moving right) or right side
            (when moving left) into pipes. This reduces the overhead of
            calculating charge combinations in the contractions, but one
            :meth:`matvec` is formally more expensive,
            :math:`O(2 d^3 \chi^3 D)`.           
        - max_N_sites_per_ring : int, optional.
            Default is 18.
            Threshold for raising errors on too many sites per ring. 
            See :meth:`~tenpy.tools.misc.consistency_check`.
            In a higher-dimensional geometry, the area law implies that the entropy of a bipartition
            is linear in ``N_sites_per_ring`` and thus the required bond dimension is exponential
            in it. This makes MPS simulations with large ``N_sites_per_ring`` unfeasible.
            If it is too large, you will not be able to choose a reasonably large bond dimension
            *and* have enough RAM to do the simulation. We raise an error in that case.
            Can be downgraded to a warning by setting this option to None. 
        ---   
        trunc_params : dict
        Dictionary with truncation parameters:
        - chi_max : int, optional
            Default is 100.
            Keep at most `chi_max` Schmidt values. 
        - svd_min : float, optional
            Default is 1.e-14.
            Discard all small Schmidt values ``S[i] < svd_min``.
        - chi_min : int, optional
            Default is None.
            Keep at least `chi_min` Schmidt values.
        - degeneracy_tol : float, optional
            Default is None.
            Don't cut between neighboring Schmidt values with
            ``|log(S[i]/S[j])| < degeneracy_tol``, or equivalently
            ``|S[i] - S[j]|/S[j] < exp(degeneracy_tol) - 1 ~= degeneracy_tol``
            for small `degeneracy_tol`.
            In other words, keep either both `i` and `j` or none, if the
            Schmidt values are degenerate with a relative error smaller
            than `degeneracy_tol`, which we expect to happen in the case
            of symmetries. 
        - trunc_cut : float, optional
            Default is 1.e-14.
            Discard all small Schmidt values as long as
            ``sum_{i discarded} S[i]**2 <= trunc_cut**2``.
        ---   
        mixer_params : dict
        Dictionary with truncation parameters:
        - amplitude : float | None, optional
            Default is 1.e-5.
            Current amplitude of the mixer. Meaning is specific to the concrete Mixer subclass.
            A value of ``None`` indicates that the given mixer has no tuneable amplitude.
        - decay : float | None, optional
            Default is 2.0 if `finite` else 2.0 ** (15 / 50).
            If both `amplitude` and `decay` are not None, the `amplitude` is divided by `decay` after
            each sweep.
        - disable_after : int | None, optional
            Default is 15 if `finite` else 50.
            We disable the mixer completely after this number of sweeps.
            ``None`` means to never disable the mixer.
        ---
        - P_tol_to_trunc : float, optional
            Default is 0.05.
            It's reasonable to choose the Lanczos convergence criteria
            ``'P_tol'`` not many magnitudes lower than the current
            truncation error. Therefore, if `P_tol_to_trunc` is not
            ``None``, we update `P_tol` of `lanczos_params` to
            ``max_trunc_err*P_tol_to_trunc``,
            restricted to the interval [`P_tol_min`, `P_tol_max`],
            where ``max_trunc_err`` is the maximal truncation error
            (discarded weight of the Schmidt values) due to truncation
            right after each Lanczos optimization during the sweeps.
        - P_tol_min : float, optional
            Default is max(1.e-30, svd_min**2 * P_tol_to_trunc, trunc_cut**2 * P_tol_to_trunc).
            Lower bound for `P_tol` in Lanczos.
        - P_tol_max : float, optional
            Default is 1.e-4.
            Upper bound for `P_tol` in Lanczos.
        - E_tol_to_trunc : float, optional
            Default to None.
            It's reasonable to choose the Lanczos convergence criteria
            ``'E_tol'`` not many magnitudes lower than the current
            truncation error. Therefore, if `E_tol_to_trunc` is not
            ``None``, we update `E_tol` of `lanczos_params` to
            ``max_E_trunc*E_tol_to_trunc``,
            restricted to the interval [`E_tol_min`, `E_tol_max`],
            where ``max_E_trunc`` is the maximal energy difference due to
            truncation right after each Lanczos optimization during the
            sweeps.
        - E_tol_min : float, optional
            Default is 1.e-16
            Lower bound for `E_tol` in Lanczos.
        - E_tol_max : float, optional
            Default is 1.e-4
            Upper bound for `E_tol` in Lanczos.
        --- 
        lanczos_params : dict
        Dictionary with Lanczos parameters:
        - E_tol : float, optional
            Default is inf.
            Stop if energy difference per step < `E_tol`
        - N_min : int, optional
            Default is 2.
            Minimum number of steps to perform.
        - N_max : int, optional
            Default is 20.
            Maximum number of steps to perform.
        - N_cache : int, optional
            Default is N_max.
            The maximum number of `psi` to keep in memory during the first iteration.
            By default, we keep all states (up to N_max).
            Set this to a number >= 2 if you are short on memory.
            The penalty is that one needs another Lanczos iteration to
            determine the ground state in the end, i.e., runtime is large.
        - P_tol : float, optional
            Default is 1.e-14.
            Tolerance for the error estimate from the Ritz Residual,
            stop if ``(RitzRes/gap)**2 < P_tol``
        - min_gap : float, optional
            Default is 1.e-12.
            Lower cutoff for the gap estimate used in the P_tol criterion.
        - reortho : bool, optional
            Default is False.
            For poorly conditioned matrices, one can quickly lose orthogonality of the
            generated Krylov basis.
            If `reortho` is True, we re-orthogonalize against all the
            vectors kept in cache to avoid that problem.
        - E_shift : float | None, optional
            Default is None.
            Shift the energy (=eigenvalues) by that amount *during* the Lanczos run by using the
            :class:`~tenpy.linalg.sparse.ShiftNpcLinearOperator`.
            The ground state energy `E0` returned by :meth:`run` is made independent of the shift.
            This option is useful if the :class:`~tenpy.linalg.sparse.OrthogonalNpcLinearOperator`
            is used: the orthogonal vectors are *exact* eigenvectors with eigenvalue 0 independent
            of the shift, so you can use it to ensure that the energy is smaller than zero
            to avoid getting those.
        - cutoff : float, optional
            Default is np.finfo(psi0.dtype if not isinstance(psi0, list) else psi0[0].dtype).eps * 100
            Numerical cutoff for Lanczos.
        ---
        The following parameters are only relevant for infinite boundary conditions:
        - update_env : int, optional
            Default is ``self.N_sweeps_check // 2``.
            For infinite boundary conditions, number of sweeps without bond 
            optimization to update the environment,
            performed every `N_sweeps_check` sweeps.
        - norm_tol_iter : int, optional
            Default is 5.
            For infinite boundary conditions, perform at most `norm_tol_iter`
            *`update_env` sweeps to converge the norm error below `norm_tol`.
        - norm_tol : float, optional
            Default is 1.e-5.
            For infinite boundary conditions, after the DMRG run, update 
            the environment with at most `norm_tol_iter` sweeps until
            ``np.linalg.norm(psi.norm_err()) < norm_tol``.
        - norm_tol_final : float, optional
            Default is 1.e-10.
            For infinite boundary conditions, after performing `norm_tol_iter`
            *`update_env` sweeps, if ``np.linalg.norm(psi.norm_err()) < norm_tol_final``, 
            call :meth:`~tenpy.networks.mps.canonical_form` to canonicalize
            instead. This tolerance should be stricter than `norm_tol`
            to ensure canonical form even if DMRG cannot fully converge.
    
    Returns
    -------
    dmrg_params : dict
        Dictionary of DMRG parameters suitable for TeNPy DMRG algorithms.
    """
    dmrg_params = {
        'active_sites': active_sites,
        'chi_list': chi_list,
        'diag_method': diag_method,
        'mixer': mixer,
        'max_E_err': max_E_err,
    }
    # Merge user-provided truncation, mixer, and lanczos params if present
    dmrg_params['trunc_params'] = {'chi_max': chi_max, 'svd_min': svd_min}
    dmrg_params['mixer_params'] = {}
    dmrg_params['lanczos_params'] = {}
    for key, value in kwargs.items():
        if key in ('trunc_params', 'mixer_params', 'lanczos_params'):
            dmrg_params[key].update(value)
        else:
            dmrg_params[key] = value
    return dmrg_params


def tenpy_dmrg(psi, model, options):
    r"""Run the DMRG algorithm to find the ground state of the given model.

    Parameters
    ----------
    psi : :class:`~tenpy.networks.mps.MPS`
        Initial guess for the ground state, which is to be optimized in-place.
    model : :class:`~tenpy.models.MPOModel`
        The model representing the Hamiltonian for which we want to find the ground state.
    options : dict
        Further optional parameters as described in :cfg:config:`DMRG`.
    
    Returns
    -------
    E : float
        The ground state energy of the model.
    gs : :class:`~tenpy.networks.mps.MPS`
        The ground state MPS after the DMRG run.
    engine : :class:`~tenpy.algorithms.dmrg.DMRGEngine`
        The DMRG engine used to run the algorithm, which contains additional information
        such as the number of sweeps, bond dimensions, and convergence criteria.
    """
    # initialize the engine
    options = asConfig(options, 'DMRG')
    active_sites = options.get('active_sites', 2, int)
    if active_sites == 1:
        engine = SingleSiteDMRGEngine(psi, model, options)
    elif active_sites == 2:
        engine = TwoSiteDMRGEngine(psi, model, options)
    else:
        raise ValueError("For DMRG, can only use 1 or 2 active sites, not {}".format(active_sites))
    E, gs = engine.run()
    return E, gs, engine









