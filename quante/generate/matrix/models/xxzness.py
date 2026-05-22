# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-22 18:45:01
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-22 23:36:36

r"""Exact MPA/MPS NESS of the boundary driven XXZ chain.

This script implements Eqs. (2.22), (2.31), and (2.36) of arXiv:1504.00783v3
for the maximally boundary-driven XXZ chain,

    L(rho) = -i[H, rho] + eps D_{sigma_1^+}(rho) + eps D_{sigma_n^-}(rho),
    D_L(rho) = 2 L rho L^\dagger - {L^\dagger L, rho}.
"""

from dataclasses import dataclass
import numpy as np

def local_operators() -> dict[str, np.ndarray]:
    """Spin-1/2 matrices in the sigma^z eigenbasis |up>, |down>."""
    return {
        "0": np.eye(2, dtype=complex),
        "+": np.array([[0, 1], [0, 0]], dtype=complex),
        "-": np.array([[0, 0], [1, 0]], dtype=complex),
        "z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

@dataclass(frozen=True)
class XXZMPA:
    n: int
    delta: float
    eps: float

    @property
    def aux_dim(self) -> int:
        # Length-n walks starting and ending at 0 never need k > n.
        return self.n + 1

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Return arrays a_k, b_k from Eqs. (2.32)-(2.34), truncated to aux_dim."""
        a = np.zeros(self.aux_dim, dtype=complex)
        a[0] = 1.0
        if self.aux_dim > 1:
            a[1] = self.delta + 0.5j * self.eps
        for idx in range(1, self.aux_dim - 1):
            a[idx + 1] = 2 * self.delta * a[idx] - a[idx - 1]

        b = np.zeros(self.aux_dim, dtype=complex)
        b[0] = 1j * self.eps
        for idx in range(self.aux_dim - 1):
            b[idx + 1] = b[idx] + 2 * a[idx + 1] * (self.delta * a[idx + 1] - a[idx])
        return a, b

    def auxiliary_matrices(self) -> dict[str, np.ndarray]:
        """Build A_0, A_+, A_- of Eq. (2.31)."""
        d = self.aux_dim
        a, b = self.coefficients()
        amat = {
            "0": np.diag(a),
            "+": np.zeros((d, d), dtype=complex),
            "-": np.zeros((d, d), dtype=complex),
        }
        for k in range(d - 1):
            amat["+"][k, k + 1] = b[k]
            amat["-"][k + 1, k] = 1.0
        return amat

    def coefficient_mps_tensor(self) -> np.ndarray:
        """Rank-3 coefficient tensor B[l,r,alpha] for Eq. (2.22)."""
        amat = self.auxiliary_matrices()
        return np.stack([amat["+"], amat["-"], amat["0"]], axis=2)

    def operator_basis_tensor(self) -> np.ndarray:
        """Rank-3 local basis tensor P[alpha,out,in] = sigma^alpha."""
        ops = local_operators()
        return np.stack([ops["+"], ops["-"], ops["0"]], axis=0)

    def omega_mpo_tensor(self) -> np.ndarray:
        """Rank-4 MPO tensor W[l,r,out,in] for Omega_n."""
        return np.einsum(
            "lra,aoi->loir",
            self.coefficient_mps_tensor(),
            self.operator_basis_tensor(),
            optimize=True,
        )

    def omega_mpo(self) -> list[np.ndarray]:
        """Return Omega_n as MPO tensors with axes (bond, bra, ket, bond).

        The open boundaries <0| and |0> are absorbed into the first and last
        tensors, so the returned boundary bond dimensions are 1.

        to mpo:
        >>> import quante.tensornetwork as tn
        >>> tn.MPO(XXZMPA(n=n, delta=delta, eps=eps).omega_mpo())
        MPO;  complex128;  norm: 7.606e+00;  maxbonddim: 6;
        physdim:    2|    2|    2|    2|    2| 
                 ----O-----O-----O-----O-----O----
        physdim:    2|    2|    2|    2|    2| 
        bonddim:  1     6     6     6     6     1
        site:        0     1     2     3     4  
        """
        tensor = self.omega_mpo_tensor()
        tensors = [tensor.copy() for _ in range(self.n)]
        tensors[0] = tensors[0][0:1, :, :, :]
        tensors[-1] = tensors[-1][:, :, :, 0:1]
        return tensors
    
    def omega_mat(self) -> np.ndarray:
        """Contract Omega_n MPO to a dense matrix.

        This is only suitable for small n. The local MPO tensor convention is
        (left_bond, bra, ket, right_bond), and the returned matrix uses the
        standard grouped ordering (bra_1...bra_n, ket_1...ket_n).
        """
        env = np.array([[[1.0 + 0.0j]]])
        for tensor in self.omega_mpo():
            env = np.einsum("aij,aklb->bikjl", env, tensor, optimize=True)
            env = env.reshape(
                tensor.shape[-1],
                env.shape[1] * env.shape[2],
                env.shape[3] * env.shape[4],
            )
        return env[0]
    
def verify_ed(
    n: int,
    delta: float,
    eps: float,
    tol: float = 1e-10,
    verbose: bool = True,
) -> bool:
    """Verify L(rho)=0 by exact diagonalization/dense matrices.

    This function intentionally contains the dense helper logic locally:
    Kronecker products, the XXZ Hamiltonian, the elementary dissipator, and the
    full Lindbladian. It is meant for small n only.
    """
    model = XXZMPA(n=n, delta=delta, eps=eps)
    omega = model.omega_mat()
    rho = omega @ omega.conj().T

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")

    dim = rho.shape[0]
    n_float = np.log2(dim)
    n = int(round(n_float))
    if 2**n != dim:
        raise ValueError("rho dimension must be 2**n")

    ops = local_operators()

    def kron_all(mats: list[np.ndarray]) -> np.ndarray:
        out = np.array([[1.0 + 0.0j]])
        for mat in mats:
            out = np.kron(out, mat)
        return out

    def xxz_hamiltonian() -> np.ndarray:
        h2 = (
            2 * np.kron(ops["+"], ops["-"])
            + 2 * np.kron(ops["-"], ops["+"])
            + delta * np.kron(ops["z"], ops["z"])
        )
        h = np.zeros((dim, dim), dtype=complex)
        for site in range(n - 1):
            left = (
                kron_all([ops["0"]] * site)
                if site
                else np.array([[1.0 + 0.0j]])
            )
            right_len = n - site - 2
            right = (
                kron_all([ops["0"]] * right_len)
                if right_len
                else np.array([[1.0 + 0.0j]])
            )
            h += np.kron(np.kron(left, h2), right)
        return h

    def dissipator_dense(jump: np.ndarray, state: np.ndarray) -> np.ndarray:
        jdagj = jump.conj().T @ jump
        return 2 * jump @ state @ jump.conj().T - jdagj @ state - state @ jdagj

    def lindblad_dense(state: np.ndarray) -> np.ndarray:
        ident = ops["0"]
        h = xxz_hamiltonian()
        jump_left = kron_all([ops["+"]] + [ident] * (n - 1))
        jump_right = kron_all([ident] * (n - 1) + [ops["-"]])
        return (
            -1j * (h @ state - state @ h)
            + eps * dissipator_dense(jump_left, state)
            + eps * dissipator_dense(jump_right, state)
        )

    lrho = lindblad_dense(rho)
    abs_residual = np.linalg.norm(lrho)
    rho_norm = np.linalg.norm(rho)
    rel_residual = abs_residual / max(rho_norm, np.finfo(float).tiny)
    ok = bool(rel_residual < tol)

    if verbose:
        print(f"ED verify: n={n}, Delta={delta}, eps={eps}")
        print(f"||L(rho)|| = {abs_residual:.3e}")
        print(f"||L(rho)|| / ||rho|| = {rel_residual:.3e}")
        print(f"passed = {ok}")

    return ok
