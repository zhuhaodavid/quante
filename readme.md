# Quante

*A Python toolkit for simulating quantum many-body spin systems (especially 1D chains) using exact diagonalization and tensor-network methods.*

## Overview

**Quante** is a lightweight and flexible research-oriented toolkit designed for numerical simulation of **quantum many-body spin systems**.  
It provides a unified framework for operator representation, Hamiltonian construction, and time evolution, supporting both **exact diagonalization (ED)** and **tensor-network** approaches.

The package aims to facilitate fast prototyping and reproducible research in the study of:
- Quantum chaos and thermalization  
- Disordered and non-Hermitian systems  
- Nonequilibrium and open-system dynamics  

## Key Features

- **Operator abstraction** for constructing many-body Hamiltonians with symbolic composition and automatic matrix generation.
- **Efficient matrix-element generation** accelerated by [`Numba`](https://numba.pydata.org/) JIT compilation.
- **Exact diagonalization** and **Krylov-based time evolution** with support for sparse matrices and GPU acceleration.
- **Tensor-network algorithms** including TEBD and DMRG, implemented with [`PyTorch`](https://pytorch.org/) backend.
- **Unified basis and operator interface**, compatible with external libraries such as [`QuSpin`](https://github.com/QuSpin/QuSpin), [`TenPy`](https://github.com/tenpy/tenpy), [`ITensor`](https://github.com/ITensor/ITensors.jl)
- Simple I/O utilities for saving and loading large-scale state and operator data.


## Installation

```bash
git clone https://github.com/zhuhaodavid/quante.git
cd quante
pip install -e .
```

Requirements:

* Python ≥ 3.10
* `Numba`, `NumPy`, `SciPy`, `PyTorch`
* (optional) `Matplotlib` for visualization


## Example Usage

### 1. Constructing a spin Hamiltonian

```python
import quante as qt
op = qt.generate.operas

# Define a spin-1/2 Heisenberg chain with open boundary
L = 4
ham = op.sum(op.xx(i,i+1) + op.yy(i,i+1) + op.zz(i,i+1) for i in range(L-1))
basis = qt.generate.basis.spin_basis(L=L, Nup=L//2)
hammat = ham.to_matrix(basis=basis, pauli=False, sparse=True)
```

### 2. Diagonalization and dynamics

```python
import quante as qt
import numpy as np
op = qt.generate.operas

L = 10
hammat = qt.generate.matrix.ising_matrix(L, sparse=True)
print(qt.linalg.krylov.eigsolve(hammat, which='SR')[0])
print(qt.generate.solvable.heisenberg.ising_ground_energy(L))

tlist = np.linspace(0, 10, 200)
init_state = qt.generate.state.neel(L=L, down_first=True)
basis = qt.generate.basis.spin_basis(L=L)
obsoper = [op.z(i).to_matrix(basis=basis, pauli=False, sparse=True) for i in range(L)]

qt.linalg.evolve_and_measure(
    hammat, init_state, tlist,
    measure=obsoper
)
```

### 3. Tensor-network simulation

```python
import quante as qt
op = qt.generate.operas

L = 10
ham = op.heisenberg_operator(L)
H = ham.to_mpo(pauli=False)
eng, vec = H.dmrg(nsweep=10)
ham.gdenergy(pauli=False)
```

More usage examples can be found in `README_zh.md` and the `example/` folder, but they are written in Chinese.


## Citation

If you find this toolkit useful for your research, please cite or acknowledge the repository in your work:

```bibtex
@misc{zhu2025quante,
  author       = {Hao Zhu, Dingzu Wang},
  title        = {Quante: A Python toolkit for simulating quantum many-body spin systems},
  year         = {2025},
  howpublished = {\url{https://github.com/zhuhaodavid/quante}}
}
```


## License

This project is released under the MIT License.
You are free to use and modify it for academic or research purposes with proper attribution.

Parts of this package (quante/linalg/krylov/) are based on the Julia package KrylovKit.jl,
which is licensed under the MIT License.
The original license is included in quante/linalg/krylov/LICENSE.KrylovKit.

Parts of this package (quante/bridge/torch_utils/networks) are based on the Julia package ITensors.jl,
which is licensed under the Apache License 2.0.
The original license is included in quante/bridge/torch_utils/networks/LICENSE.ITensors.

## Contact

**Hao Zhu**
School of Physics, Beihang University
📧 [zhuhao6590@gmail.com](mailto:zhuhao6590@gmail.com)
🔗 [https://github.com/zhuhaodavid](https://github.com/zhuhaodavid)

**Dingzu Wang**
National University of Singapore
📧 [dingzu.wang@sutd.edu.sg](mailto:dingzu.wang@sutd.edu.sg)





