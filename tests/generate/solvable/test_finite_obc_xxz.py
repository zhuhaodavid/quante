import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from quante.generate.solvable.bethe_ansatz import finite_obc_xxz
from quante.generate.solvable.bethe_ansatz.bethe_state import XXZBetheKernel
from quante.generate.solvable.heisenberg import heisenberg_matrix


@pytest.mark.parametrize("delta", [-0.5, 0.0, 0.7])
@pytest.mark.parametrize("L", [4, 5, 6, 7, 8])
def test_obc_ground_energy_matches_exact_diagonalization(L, delta):
    state = finite_obc_xxz.solve_xxz_state(L, delta, M=L // 2)
    matrix = heisenberg_matrix(
        L,
        j=(1.0, 1.0, delta),
        cyclic=False,
        Nup=L - L // 2,
        pauli=False,
    )
    exact = np.linalg.eigvalsh(matrix)[0]

    assert state.converged
    assert state.residual_norm < 1e-10
    np.testing.assert_allclose(state.qnums, np.arange(1, L // 2 + 1))
    assert np.isclose(state.xxz_energy(pauli=False), exact, atol=1e-11)


def test_obc_state_and_energy_conventions():
    state = finite_obc_xxz.solve_xxz_state(6, delta=0.7, M=3)

    assert state.metadata["boundary_condition"] == "open"
    assert state.metadata["boundary"] == "FreeOpenXXZBoundary"
    np.testing.assert_allclose(
        state.roots,
        -state.eta / 2.0 + 0.5j * state.alphas,
    )
    assert np.isclose(
        state.xxz_energy(pauli=True),
        4.0 * state.xxz_energy(pauli=False),
    )


@pytest.mark.parametrize("delta", [-0.6, 0.4, 1.0, 1.5])
def test_diagonal_boundary_state_matches_exact_diagonalization(delta):
    L = 6
    M = 3
    h_minus = 0.2
    h_plus = -0.1
    boundary = finite_obc_xxz.DiagonalOpenXXZBoundary(h_minus, h_plus)
    state = finite_obc_xxz.solve_xxz_state(
        L, delta, M=M, boundary=boundary
    )
    hz = np.zeros(L)
    hz[0] = h_minus
    hz[-1] = h_plus
    matrix = heisenberg_matrix(
        L,
        j=(1.0, 1.0, delta),
        hz=hz,
        cyclic=False,
        Nup=L - M,
        pauli=True,
    )

    assert state.metadata["h_minus"] == h_minus
    assert state.metadata["h_plus"] == h_plus
    assert np.isclose(
        state.xxz_energy(pauli=True),
        np.linalg.eigvalsh(matrix)[0],
        atol=1e-11,
    )


@pytest.mark.parametrize("delta", [-1.0, -1.5])
def test_negative_diagonal_boundary_state_matches_sector_top(delta):
    L = 6
    M = 3
    h_minus = 0.1
    h_plus = -0.15
    boundary = finite_obc_xxz.DiagonalOpenXXZBoundary(h_minus, h_plus)
    state = finite_obc_xxz.solve_xxz_state(
        L, delta, M=M, boundary=boundary
    )
    hz = np.zeros(L)
    hz[0] = h_minus
    hz[-1] = h_plus
    matrix = heisenberg_matrix(
        L,
        j=(1.0, 1.0, delta),
        hz=hz,
        cyclic=False,
        Nup=L - M,
        pauli=True,
    )
    assert np.isclose(
        state.xxz_energy(pauli=True),
        np.linalg.eigvalsh(matrix)[-1],
        atol=1e-11,
    )


def test_negative_easy_axis_same_sign_boundary_ground_is_polarized():
    L = 6
    delta = -1.5
    h_minus = 0.2
    h_plus = 0.1
    hz = np.zeros(L)
    hz[0] = h_minus
    hz[-1] = h_plus
    exact = np.linalg.eigvalsh(
        heisenberg_matrix(
            L,
            j=(1.0, 1.0, delta),
            hz=hz,
            cyclic=False,
            pauli=True,
        )
    )[0]
    assert np.isclose(
        finite_obc_xxz.ground_energy(
            L,
            delta=delta,
            h_minus=h_minus,
            h_plus=h_plus,
        ),
        exact,
        atol=1e-12,
    )


def test_negative_easy_axis_opposite_boundary_ground_is_rejected():
    with pytest.raises(NotImplementedError, match="domain-wall"):
        finite_obc_xxz.ground_energy(
            6,
            delta=-1.5,
            h_minus=0.2,
            h_plus=-0.1,
        )


@pytest.mark.parametrize("delta", [-0.6, 0.4, 1.0, 1.5])
def test_diagonal_boundary_ground_energy_matches_exact_diagonalization(delta):
    L = 6
    h_minus = 0.2
    h_plus = -0.1
    hz = np.zeros(L)
    hz[0] = h_minus
    hz[-1] = h_plus
    exact = np.linalg.eigvalsh(
        heisenberg_matrix(
            L,
            j=(1.0, 1.0, delta),
            hz=hz,
            cyclic=False,
            pauli=True,
        )
    )[0]
    assert np.isclose(
        finite_obc_xxz.ground_energy(
            L,
            delta=delta,
            h_minus=h_minus,
            h_plus=h_plus,
            pauli=True,
        ),
        exact,
        atol=1e-11,
    )


@pytest.mark.parametrize("j", [0.7, -1.2])
def test_diagonal_boundary_nonunit_j_ground_matches_exact_diagonalization(j):
    L = 6
    delta = 0.4
    h_minus = 0.2
    h_plus = -0.1
    hz = np.zeros(L)
    hz[0] = j * h_minus
    hz[-1] = j * h_plus
    exact = np.linalg.eigvalsh(
        heisenberg_matrix(
            L,
            j=(j, j, j * delta),
            hz=hz,
            cyclic=False,
            pauli=True,
        )
    )[0]
    assert np.isclose(
        finite_obc_xxz.ground_energy(
            L,
            j=j,
            delta=delta,
            h_minus=h_minus,
            h_plus=h_plus,
        ),
        exact,
        atol=1e-11,
    )


@pytest.mark.parametrize(
    ("delta", "field"),
    [(0.4, 1.4), (1.0, 2.0), (1.5, np.sqrt(1.5 ** 2 - 1.0))],
)
def test_boundary_bound_state_threshold_is_rejected(delta, field):
    boundary = finite_obc_xxz.DiagonalOpenXXZBoundary(field, 0.0)
    with pytest.raises(ValueError, match="complex boundary roots"):
        finite_obc_xxz.solve_xxz_state(
            6, delta, M=3, boundary=boundary
        )


def test_diagonal_boundary_zero_field_reduces_to_free_boundary():
    free = finite_obc_xxz.solve_xxz_state(6, 0.4, M=3)
    diagonal = finite_obc_xxz.solve_xxz_state(
        6,
        0.4,
        M=3,
        boundary=finite_obc_xxz.DiagonalOpenXXZBoundary(),
    )
    np.testing.assert_allclose(diagonal.alphas, free.alphas, atol=1e-12)
    assert np.isclose(
        diagonal.xxz_energy(pauli=True),
        free.xxz_energy(pauli=True),
        atol=1e-12,
    )


def test_diagonal_boundary_spin_operator_normalization():
    L = 6
    delta = 0.4
    h_minus = 0.2
    h_plus = -0.1
    boundary = finite_obc_xxz.DiagonalOpenXXZBoundary(h_minus, h_plus)
    state = finite_obc_xxz.solve_xxz_state(
        L, delta, M=L // 2, boundary=boundary
    )
    hz = np.zeros(L)
    hz[0] = h_minus / 2.0
    hz[-1] = h_plus / 2.0
    matrix = heisenberg_matrix(
        L,
        j=(1.0, 1.0, delta),
        hz=hz,
        cyclic=False,
        Nup=L - L // 2,
        pauli=False,
    )
    assert np.isclose(
        state.xxz_energy(pauli=False),
        np.linalg.eigvalsh(matrix)[0],
        atol=1e-11,
    )


@pytest.mark.parametrize("delta", [1.2, 1.5])
@pytest.mark.parametrize("L", [4, 5, 6, 7, 8])
def test_massive_obc_ground_energy_matches_exact_diagonalization(L, delta):
    state = finite_obc_xxz.solve_xxz_state(L, delta, M=L // 2)
    matrix = heisenberg_matrix(
        L,
        j=(1.0, 1.0, delta),
        cyclic=False,
        Nup=L - L // 2,
        pauli=False,
    )
    exact = np.linalg.eigvalsh(matrix)[0]

    assert state.metadata["regime"] == "massive"
    assert np.all(state.alphas > 0.0)
    assert np.all(state.alphas < np.pi)
    assert state.residual_norm < 1e-10
    assert np.isclose(state.xxz_energy(pauli=False), exact, atol=1e-11)


def test_massive_obc_singular_real_root_is_rejected():
    with pytest.raises(RuntimeError, match="singular alpha=pi root"):
        finite_obc_xxz.solve_xxz_state(8, delta=2.0, M=4)


def test_massive_obc_stagnation_is_identified_as_singular_root():
    with pytest.raises(RuntimeError, match="not a root-method failure"):
        finite_obc_xxz.ground_energy(50, delta=5.0)


def test_massive_obc_public_ground_energy():
    state = finite_obc_xxz.solve_xxz_state(6, delta=1.5, M=3)
    assert np.isclose(
        finite_obc_xxz.ground_energy(6, delta=1.5, pauli=False),
        state.xxz_energy(pauli=False),
    )


@pytest.mark.parametrize("L", [4, 5, 6, 7, 8])
def test_isotropic_obc_ground_energy_matches_exact_diagonalization(L):
    state = finite_obc_xxz.solve_xxz_state(L, delta=1.0, M=L // 2)
    matrix = heisenberg_matrix(
        L,
        j=(1.0, 1.0, 1.0),
        cyclic=False,
        Nup=L - L // 2,
        pauli=False,
    )
    exact = np.linalg.eigvalsh(matrix)[0]

    assert state.metadata["regime"] == "isotropic"
    assert state.residual_norm < 1e-10
    np.testing.assert_allclose(
        state.roots,
        -0.5 + 0.5j * state.alphas,
    )
    assert np.isclose(state.xxz_energy(pauli=False), exact, atol=1e-11)


def test_isotropic_obc_jacobian():
    L = 8
    qnums = np.arange(1, L // 2 + 1, dtype=float)
    kernel = XXZBetheKernel.from_delta(1.0)
    boundary = finite_obc_xxz.FreeOpenXXZBoundary()
    state = finite_obc_xxz.solve_xxz_state(L, delta=1.0, qnums=qnums)

    analytic = finite_obc_xxz._jacobian(
        state.alphas, L, kernel, boundary, qnums
    )
    numeric = approx_derivative(
        lambda alphas: finite_obc_xxz._residual(
            alphas, L, kernel, boundary, qnums
        ),
        state.alphas,
    )
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("delta", [-1.0, -1.2, -1.5])
@pytest.mark.parametrize("L", [4, 5, 6, 7])
def test_negative_obc_mapped_state_matches_sector_top(L, delta):
    M = L // 2
    state = finite_obc_xxz.solve_xxz_state(L, delta=delta, M=M)
    matrix = heisenberg_matrix(
        L,
        j=(1.0, 1.0, delta),
        cyclic=False,
        Nup=L - M,
        pauli=False,
    )
    sector_top = np.linalg.eigvalsh(matrix)[-1]

    assert state.metadata["mapped_delta"] == abs(delta)
    assert state.metadata["regime"] in {
        "isotropic_negative",
        "massive_negative",
    }
    assert np.isclose(state.xxz_energy(pauli=False), sector_top, atol=1e-11)


@pytest.mark.parametrize("delta", [-1.0, -1.2, -2.0])
@pytest.mark.parametrize("L", [4, 5, 6, 7, 8])
def test_negative_obc_ground_energy_is_polarized(L, delta):
    expected_spin = (L - 1) * delta / 4.0

    assert np.isclose(
        finite_obc_xxz.ground_energy(L, delta=delta, pauli=False),
        expected_spin,
    )
    assert np.isclose(
        finite_obc_xxz.ground_energy(L, delta=delta, pauli=True),
        4.0 * expected_spin,
    )


def test_obc_quantum_number_validation():
    with pytest.raises(ValueError, match="positive"):
        finite_obc_xxz.solve_xxz_state(6, delta=0.7, qnums=[0, 1, 2])
    with pytest.raises(ValueError, match="integers"):
        finite_obc_xxz.solve_xxz_state(6, delta=0.7, qnums=[0.5, 1.5, 2.5])


def test_negative_isotropic_obc_root_mapping():
    state = finite_obc_xxz.solve_xxz_state(6, delta=-1.0, M=3)
    np.testing.assert_allclose(
        state.roots,
        -0.5 + 0.5j * state.alphas,
    )


@pytest.mark.parametrize("delta", [-1.5, -1.0, -0.7, 0.7, 1.0, 1.5])
@pytest.mark.parametrize("L", [4, 5, 6, 7])
@pytest.mark.parametrize("j", [0.4, -1.7])
def test_obc_nonunit_j_ground_energy_matches_exact_diagonalization(
    L, delta, j
):
    matrix = heisenberg_matrix(
        L,
        j=(j, j, j * delta),
        cyclic=False,
        pauli=False,
    )
    exact = np.linalg.eigvalsh(matrix)[0]

    assert np.isclose(
        finite_obc_xxz.ground_energy(
            L,
            j=j,
            delta=delta,
            pauli=False,
        ),
        exact,
        atol=1e-11,
    )


def test_obc_zero_j_ground_energy():
    assert finite_obc_xxz.ground_energy(6, j=0.0, delta=3.0) == 0.0
