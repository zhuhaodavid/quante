# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-06 10:06:06
# @Last Modified by:   hzhu
# @Last Modified time: 2025-08-04 20:33:45

from tqdm import tqdm
from tenpy.algorithms.tdvp import SingleSiteTDVPEngine, TwoSiteTDVPEngine

def tenpy_tdvp_params(
    dt: float,
    chi_list: dict[int, int] | None = None,
    svd_min: float = 1.e-14,
    **kwargs
):
    r"""Create a dictionary of parameters for the TenPy TDVP algorithm.
    
    Parameters
    ----------
    dt : float
        Minimal time step by which to evolve.
    chi_list : dict[int, int] | None, optional
        By default (``None``) this feature is disabled.
        A dict allows to gradually increase the `chi_max`.
        An entry `at_sweep: chi` states that starting from sweep `at_sweep`,
        the value `chi` is to be used for ``trunc_params['chi_max']``.
        For example ``chi_list={0: 50, 20: 100}`` uses ``chi_max=50`` for the first
        20 sweeps and ``chi_max=100`` afterwards.
        A value of `None` is initialized to the current value of
        ``trunc_params['chi_max']`` at algorithm initialization.
        When setting to `None`, the default value of `chi_max=100` is used.
    svd_min : float, optional
        Discard all small Schmidt values ``S[i] < svd_min``. Default is 1.e-14.
    **kwargs : dict, optional
        - N_steps : int.
            Default is 1.
            Number of time steps `dt` to evolve by in :meth:`run`.
            Adjusting `dt` and `N_steps` at the same time allows to keep the evolution time
            done in :meth:`run` fixed.
            Further, e.g., the Trotter decompositions of order > 1 are slightly more efficient
            if more than one step is performed at once.
        - start_time : float.
            Default is 0.0.
            Initial value for :attr:`evolved_time`.
        - combine : bool. 
            Default is False.
            Whether to combine legs into pipes. This combines the virtual and
            physical leg for the left site (when moving right) or right side
            (when moving left) into pipes. This reduces the overhead of
            calculating charge combinations in the contractions, but one
            :meth:`matvec` is formally more expensive,
            :math:`O(2 d^3 \chi^3 D)`.
        - max_dt : float
            Defaulit is 1.0.
            'dt > ``max_dt`` is unreasonably large for TDVP.',
        - start_trunc_err : :class:`~tenpy.algorithms.truncation.TruncationError`
            Initial truncation error for :attr:`trunc_err`.
        ---
        trunc_params:
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
        lanczos_params:
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
    
    Returns
    -------
    tdvp_params : dict
        Dictionary of TDVP parameters suitable for TeNPy TDVP algorithms.
    """
    allowed_keys = {
        'trunc_params': ['chi_max', 'chi_min', 'degeneracy_tol', 'trunc_cut'],
        'lanczos_params': [
            'E_tol', 'N_min', 'N_max', 'N_cache', 'P_tol', 'min_gap',
            'reortho', 'E_shift', 'cutoff'
        ],
    }

    extracted_params = {
        key: {k: kwargs.pop(k) for k in keys if k in kwargs}
        for key, keys in allowed_keys.items()
    }
    
    tdvp_params = {
        'dt': dt,
        'chi_list': chi_list,
        'trunc_params': {
            'svd_min': svd_min,
            **extracted_params['trunc_params'],
        },
        'lanczos_params': extracted_params['lanczos_params'],
        **kwargs,
    }
    return tdvp_params

def tenpy_tdvp_(psi, M, final_time, tdvp_params, progressbar=False, active_sites=2):
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
    tdvp_params : dict
        A dictionary containing the TDVP parameters.
    
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
    if active_sites == 1:
        engine = SingleSiteTDVPEngine(psi, M, tdvp_params)
    elif active_sites == 2:
        engine = TwoSiteTDVPEngine(psi, M, tdvp_params)
    else:
        raise ValueError("For DMRG, can only use 1 or 2 active sites, not {}".format(active_sites))

    dt = tdvp_params['dt']
    N_steps = tdvp_params.get('N_steps', 1)
    start_time = tdvp_params.get('start_time', 0.0)
    dt_measure = dt * N_steps
    Ns = int((final_time - start_time) / dt_measure + 0.5)

    t_iter = tqdm(range(Ns), ascii=True) if progressbar else range(Ns)
    for _ in t_iter:
        engine.run()
        t = engine.evolved_time
        yield t, psi

