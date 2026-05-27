# -*- coding: utf-8 -*-

import numpy as np


class ProcessTensor:
    """Minimal in-memory process tensor container."""

    def __init__(self, times=None, *, initial_tensor=None):
        """Create an empty process tensor container."""
        self.times = None if times is None else np.asarray(times)
        self.initial_tensor = initial_tensor
        self.mpo_tensors = {}
        self.cap_tensors = {}

    def get_mpo_tensor(self, step):
        """Return the stored process MPO tensor for ``step`` if present."""
        return self.mpo_tensors.get(step)

    def set_mpo_tensor(self, step, tensor):
        """Store a process MPO tensor for ``step``."""
        self.mpo_tensors[step] = tensor

    def get_cap_tensor(self, step):
        """Return the cap tensor for ``step`` if present."""
        return self.cap_tensors.get(step)

    def set_cap_tensor(self, step, tensor):
        """Store the cap tensor for ``step``."""
        self.cap_tensors[step] = tensor

    def bond_dims(self):
        """Return left and right bond dimensions of stored MPO tensors."""
        dims = []
        for step in sorted(self.mpo_tensors):
            tensor = self.mpo_tensors[step]
            if tensor is None or np.ndim(tensor) < 2:
                dims.append(None)
            else:
                dims.append((tensor.shape[0], tensor.shape[-1]))
        return dims


def pt_tempo_compute(system, bath, ts, params, **kwargs):
    """Placeholder entry point for future PT-TEMPO computation."""
    raise NotImplementedError("PT-TEMPO is not implemented yet")
