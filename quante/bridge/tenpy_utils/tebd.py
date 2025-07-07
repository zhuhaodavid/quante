# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-04 10:42:20
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-06 12:53:03

import tenpy
from tqdm import tqdm


def tenpy_tebd_params_imag_time(
    delta_tau_list: list,
    chi_max: int = 100,
    svd_min: float = 1.e-14,
    order: int = 2,   
    max_error_E: float = 1.e-13,
    **kwargs
):
    """Create a dictionary of parameters for the TenPy TEBD algorithm for imaginary time evolution.

    ``U_bond = exp(- dt H_bond)`` for ``type_evo='imag'``.
    
    .. note ::
        It is almost always more efficient (and hence advisable) to use DMRG.
        This algorithms can nonetheless be used quite well as a benchmark and for comparison.
    
    Parameters
    ----------
    delta_tau_list : list
        A list of floats: the timesteps to be used.
        Choosing a large timestep `delta_tau` introduces large (Trotter) errors,
        but a too small time step requires a lot of steps to reach
        ``exp(-tau H) --> |psi0><psi0|``.
        Therefore, we start with fairly large time steps for a quick time evolution until
        convergence, and then gradually decrease the time step.
        For example, ``delta_tau_list = [0.1, 0.01, 0.001, 1.e-4, 1.e-5]``
    chi_max : int, optional
        Keep at most `chi_max` Schmidt values. Default is 100.
    svd_min : float, optional
        Discard all small Schmidt values ``S[i] < svd_min``. Default is 1.e-14.
    order : int, optional
        Order of the Suzuki-Trotter decomposition. Default is 2.
        The total error for evolution up to a fixed time `t`,
        scales as ``O(t*dt^order)``.
    max_error_E : float, optional
        Threshold for the maximum error in energy expectation value.
        Default is 1.e-13.
    **kwargs : dict, optional
        Additional parameters for the TEBD algorithm, such as:
        - N_steps : int, optional
            Default is 10.
            Number of steps before measurement can be performed. 
        - start_time : float, optional
            Default is 0.
            The initial time for the evolution. 
        - start_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`, optional
            Initial truncation error for :attr:`trunc_err`. Default is None.
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
        - max_delta_t : float, optional
            Default is 1.0.
            Threshold for raising errors on too large time steps. 
            The trotterization in the time evolution operator assumes that the time step is small.
            We raise an error if it is not. Can be downgraded to a warning by setting this option to None.
        - max_N_sites_per_ring : int, optional
            Default is 18.
            Threshold for raising errors on too many sites per ring. 
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
        A dictionary containing the TEBD parameters. 
    """
    allowed_trunc_params_keys = ['chi_min', 'degeneracy_tol', 'trunc_cut']
    trunc_params = {
        key: kwargs.pop(key) for key in allowed_trunc_params_keys if key in kwargs
    }
    tebd_params = {
        'delta_tau_list': delta_tau_list,
        'order': order,
        'max_error_E': max_error_E,
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': svd_min,
            **trunc_params
        },
        **kwargs
    }
    return tebd_params


def tenpy_tebd_GS_(psi, M, tebd_params):
    """Perform imaginary time evolution using the TenPy TEBD algorithm.

    This function returns a TenPy TEBD model and the final state after running the TEBD algorithm.
    
    The bond energies of the final state can be accessed via the `bond_energies` method of the model.
    >>> print(M.bond_energies(psi))

    .. note ::
        It is almost always more efficient (and hence advisable) to use DMRG.
        This algorithms can nonetheless be used quite well as a benchmark and for comparison.

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


def tenpy_tebd_params_real_time(
    dt: float,
    chi_max: int = 100,
    svd_min: float = 1.e-14,
    order: int = 2,   
    preserve_norm: bool = None,
    **kwargs
):
    """Create a dictionary of parameters for the TenPy TEBD algorithm.

    This function returns a dictionary of parameters for the TenPy TEBD algorithm
    for real-time evolution. 

    ``U_bond = exp(-i dt (H_bond-E_offset_bond))``

    Parameters
    ----------
    dt : float
        Minimal time step by which to evolve.
    chi_max : int, optional
        Keep at most `chi_max` Schmidt values. Default is 100.
    svd_min : float, optional
        Discard all small Schmidt values ``S[i] < svd_min``. Default is 1.e-14.
    order : int = 2
        Order of the algorithm. The total error for evolution up to a fixed time `t`
        scales as ``O(t*dt^order)``.
    preserve_norm : bool = None
        Whether the state will be normalized to its initial norm after each time step.
        Per default, this is ``False`` for real time evolution and ``True`` for imaginary time.
    **kwargs : dict, optional
        - N_steps : int.
            Default is 1.
            Number of time steps `dt` to evolve by in :meth:`run`.
            Adjusting `dt` and `N_steps` at the same time allows to keep the evolution time
            done in :meth:`run` fixed.
            Further, e.g., the Trotter decompositions of order > 1 are slightly more efficient
            if more than one step is performed at once.
        - start_time : float
            Default is 0.0.
            Initial value for :attr:`evolved_time`.
        - trunc_cut : float, optional
            Default is 1.e-14.
            Discard all small Schmidt values as long as
            ``sum_{i discarded} S[i]**2 <= trunc_cut**2``. 
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
        - E_offset : None | list of float.
            Default is None.
            Possible offset added to `H_bond` for real-time evolution.
        - start_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            Initial truncation error for :attr:`trunc_err`.
        - max_delta_t : float | None.
            Default is 1.0.
            Threshold for raising errors on too large time steps. Default ``1.0``.
            The trotterization in the time evolution operator assumes that the time step is small.
            We raise an error if it is not.
            Can be downgraded to a warning by setting this option to ``None``.
        - max_N_sites_per_ring: int.
            Default is 18.
            Threshold for raising errors on too many sites per ring.
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
        A dictionary containing the TEBD parameters
    """
    allowed_trunc_params_keys = ['chi_min', 'degeneracy_tol', 'trunc_cut']
    trunc_params = {
        key: kwargs.pop(key) for key in allowed_trunc_params_keys if key in kwargs
    }
    tebd_params = {
        'dt': dt,
        'order': order,
        'preserve_norm': preserve_norm,
        'trunc_params': {
            'chi_max': chi_max,
            'svd_min': svd_min,
            **trunc_params
        },
        **kwargs
    }
    return tebd_params



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
        A dictionary containing the TEBD parameters for the real time evolution.
    
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
    engine = tenpy.algorithms.tebd.TEBDEngine(psi, M, tebd_params)
    dt = tebd_params['dt']
    N_steps = tebd_params.get('N_steps', 1)
    start_time = tebd_params.get('start_time', 0.0)
    dt_measure = dt * N_steps
    Ns = int((final_time - start_time) / dt_measure + 0.5)
    t_iter = tqdm(range(Ns), ascii=True) if progressbar else range(Ns)
    for _ in t_iter:
        engine.run()
        t = engine.evolved_time.real
        yield t, psi
