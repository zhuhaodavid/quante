import numpy as _np
from .symmetry.basis_class import SpinBasis

__all__ = ['spin_basis', 'show_spin_basis']

def spin_basis(L: int, S: str | int | float = ..., Nup: int | None = None, kblock: int | None = None, pblock: int | None = None, zblock: int | None = None, pzblock: int | None = None, jmblock: int | tuple[int, int] | None = None) -> SpinBasis: ...
def show_spin_basis(vector: _np.ndarray) -> tuple[list[_np.ndarray], list[str]]: ...
