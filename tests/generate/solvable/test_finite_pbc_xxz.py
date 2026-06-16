import numpy as np
import pytest

from quante.generate.solvable.bethe_ansatz.finite_pbc_xxz import (
    energy_from_rapidities,
    ground_energy as finite_ground_energy,
    solve_xxz_state,
)
from quante.generate.solvable.bethe_ansatz.bethe_state import XXZBetheKernel
from quante.generate.solvable.bethe_ansatz.infinite_pbc_xxz import (
    _zero_field_density,
    _zero_field_root_density,
    compute_ground_state_density,
    ground_energy,
)


def test_isotropic_two_root_equation_and_energy():
    state = solve_xxz_state(4, delta=1.0, M=2)

    expected = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    np.testing.assert_allclose(state.alphas, expected, atol=1e-11, rtol=0.0)
    np.testing.assert_allclose(
        state.roots,
        -0.5 + 0.5j * expected,
        atol=1e-11,
        rtol=0.0,
    )
    assert state.metadata["regime"] == "isotropic"
    assert state.residual_norm < 1e-10
    assert np.isclose(state.xxz_energy(pauli=False), -2.0)


def test_isotropic_vacuum_energy():
    assert np.isclose(
        energy_from_rapidities([], L=6, delta=1.0, pauli=False),
        1.5,
    )


@pytest.mark.parametrize(
    ("pauli", "expected"),
    [
        (True, -2.5 * 6),
        (False, -0.5 * 6 - 0.25 * 6),
    ],
)
def test_negative_isotropic_ground_energy(pauli, expected):
    assert np.isclose(
        finite_ground_energy(
            L=6,
            j=2.0,
            delta=-1.0,
            h=-0.5,
            pauli=pauli,
        ),
        expected,
    )


def test_negative_isotropic_kernel():
    kernel = XXZBetheKernel.from_delta(-1.0)

    assert kernel.regime == "isotropic_negative"
    assert kernel.is_isotropic
    assert kernel.is_ferromagnetic_isotropic
    assert kernel.mapped_delta == 1.0
    assert kernel.energy_sign == -1.0
    np.testing.assert_allclose(
        kernel.map_alpha_to_u(np.array([-1.0, 1.0])),
        np.array([-0.5 - 0.5j, -0.5 + 0.5j]),
    )
    assert np.isclose(kernel.finite_energy([], 6, pauli=True), -6.0)


@pytest.mark.parametrize(
    ("pauli", "expected"),
    [
        (True, -2.5),
        (False, -0.75),
    ],
)
def test_negative_isotropic_infinite_ground_energy(pauli, expected):
    state = compute_ground_state_density(delta=-1.0, h=0.25)

    assert state.regime == "isotropic_negative"
    assert state.B == 0.0
    assert state.filling_density() == 0.0
    assert np.isclose(state.energy_density, -1.25)
    assert np.isclose(
        ground_energy(delta=-1.0, j=2.0, h=-0.5, pauli=pauli),
        expected,
    )


def test_isotropic_zero_field_root_density():
    alpha = np.array([-1.0, 0.0, 1.0])
    rho = _zero_field_density(XXZBetheKernel.from_delta(1.0), alpha)
    expected = 0.25 / np.cosh(np.pi * alpha / 2.0)

    np.testing.assert_allclose(rho, expected)
    with pytest.raises(ValueError, match="0 < eta < pi"):
        _zero_field_root_density(alpha, eta=0.0)


def test_isotropic_infinite_zero_field_state():
    state = compute_ground_state_density(delta=1.0, h=0.0, n_quad=401)

    assert state.regime == "isotropic"
    assert np.isinf(state.B)
    assert state.filling_density() == 0.5
    assert np.isclose(state.rho[len(state.rho) // 2], 0.25)
    assert np.isclose(state.energy_density, 1.0 - 4.0 * np.log(2.0))
    assert np.isclose(
        ground_energy(delta=1.0, h=0.0, pauli=False),
        0.25 - np.log(2.0),
    )
