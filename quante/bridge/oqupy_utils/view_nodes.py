# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2026-05-29 12:59:20
# @Last Modified by:   hzhu
# @Last Modified time: 2026-05-29 12:59:24

"""Small helpers for inspecting OQuPy ``NodeArray`` objects.

OQuPy stores tensor-network structure in TensorNetwork edges, so a raw
``node.get_tensor()`` is often not in MPS/MPO axis order.  These helpers first
reorder each node into the logical NodeArray order, then convert the tensors to
quante's local ``MPS``/``MPO`` classes for dense checks during debugging.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...tensornetwork.networks import MPS, MPO

TensorTrainKind = Literal["auto", "mps", "mpo"]


def nodearray_tensors(node_array, *, copy: bool = True) -> list[np.ndarray]:
    """Return NodeArray tensors in logical left/array/right edge order.

    Parameters
    ----------
    node_array:
        An ``oqupy.backends.node_array.NodeArray`` instance.
    copy:
        If ``True``, copy the NodeArray before reordering edges so the original
        debug object is left untouched.
    """
    array = node_array.copy() if copy else node_array
    tensors = []
    for i, node in enumerate(array.nodes):
        edges = _logical_edges(array, i)
        node.reorder_edges(edges)
        tensors.append(np.asarray(node.get_tensor()))
    return tensors


def nodearray_to_mps(node_array, *, copy: bool = True) -> MPS:
    """Convert an OQuPy MPS-like NodeArray to quante ``MPS``."""
    return MPS(_with_open_mps_bonds(nodearray_tensors(node_array, copy=copy)))


def nodearray_to_mpo(node_array, *, copy: bool = True) -> MPO:
    """Convert an OQuPy MPO-like NodeArray to quante ``MPO``."""
    return MPO(_with_open_mpo_bonds(nodearray_tensors(node_array, copy=copy)))


def nodearray_to_tt(
    node_array,
    *,
    kind: TensorTrainKind = "auto",
    copy: bool = True,
) -> MPS | MPO:
    """Convert a NodeArray to ``MPS`` or ``MPO``.

    ``kind="auto"`` uses ``node_array.rank`` when available: rank 1 maps to
    MPS and rank 2 maps to MPO.
    """
    if kind == "auto":
        rank = getattr(node_array, "rank", None)
        if rank == 1:
            kind = "mps"
        elif rank == 2:
            kind = "mpo"
        else:
            raise ValueError(f"Cannot infer TensorTrain kind from rank {rank!r}")

    if kind == "mps":
        return nodearray_to_mps(node_array, copy=copy)
    if kind == "mpo":
        return nodearray_to_mpo(node_array, copy=copy)
    raise ValueError(f"Unknown kind: {kind!r}")


def dense_from_nodearray(
    node_array,
    *,
    kind: TensorTrainKind = "auto",
    copy: bool = True,
) -> np.ndarray:
    """Convert a NodeArray to local TT and return its dense vector/matrix."""
    return nodearray_to_tt(node_array, kind=kind, copy=copy).to_matrix()


def print_nodearray_summary(node_array, *, name: str = "NodeArray") -> None:
    """Print logical tensor shapes and boundary flags for quick debugging."""
    tensors = nodearray_tensors(node_array)
    print(
        f"{name}: len={len(tensors)}, rank={getattr(node_array, 'rank', None)}, "
        f"left={getattr(node_array, 'left', None)}, "
        f"right={getattr(node_array, 'right', None)}"
    )
    for i, tensor in enumerate(tensors):
        print(f"  [{i}] shape={tensor.shape}, dtype={tensor.dtype}")


def view_nodes(
    node_array,
    t: str = "",
    *,
    kind: TensorTrainKind = "auto",
    copy: bool = True,
    print_result: bool = True,
) -> np.ndarray:
    """Compatibility helper used in notebooks: print and return dense data."""
    dense = dense_from_nodearray(node_array, kind=kind, copy=copy)
    if print_result:
        print(t, dense)
    return dense


def view_adt(mps: MPS, t: str = "", *, print_result: bool = True) -> np.ndarray:
    """Return dense data for a local TEMPO ADT MPS without mutating it."""
    mps_copy = mps.copy()
    a, b, c = mps_copy.data[-1].shape
    mps_copy.data[-1] = mps_copy.data[-1].reshape(a, b * c, 1)
    dense = mps_copy.to_matrix()
    if print_result:
        print(t, dense)
    return dense


def compare_nodearray_to_mps(
    node_array,
    mps: MPS,
    *,
    kind: TensorTrainKind = "mps",
    copy: bool = True,
) -> float:
    """Return dense norm difference between an OQuPy NodeArray and local MPS."""
    node_dense = dense_from_nodearray(node_array, kind=kind, copy=copy)
    local_dense = mps.to_matrix()
    return float(np.linalg.norm(node_dense - local_dense))


def _logical_edges(node_array, i: int):
    edges = []
    if i == 0:
        if node_array.left:
            edges.append(node_array.left_edge)
    else:
        edges.append(node_array.bond_edges[i - 1])

    edges.extend(node_array.array_edges[i])

    if i == len(node_array) - 1:
        if node_array.right:
            edges.append(node_array.right_edge)
    else:
        edges.append(node_array.bond_edges[i])
    return edges


def _with_open_mps_bonds(tensors: list[np.ndarray]) -> list[np.ndarray]:
    tensors = [np.asarray(tensor) for tensor in tensors]
    if not tensors:
        raise ValueError("Cannot convert an empty NodeArray to MPS")

    out = []
    for i, tensor in enumerate(tensors):
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, tensor.shape[0], 1)
        elif tensor.ndim == 2:
            if i == 0:
                tensor = tensor.reshape(1, *tensor.shape)
            elif i == len(tensors) - 1:
                tensor = tensor.reshape(*tensor.shape, 1)
            else:
                raise ValueError(f"Interior MPS tensor {i} has no two bonds: {tensor.shape}")
        elif tensor.ndim != 3:
            raise ValueError(f"MPS tensor {i} should have ndim 1, 2, or 3, got {tensor.shape}")
        out.append(tensor)
    return out


def _with_open_mpo_bonds(tensors: list[np.ndarray]) -> list[np.ndarray]:
    tensors = [np.asarray(tensor) for tensor in tensors]
    if not tensors:
        raise ValueError("Cannot convert an empty NodeArray to MPO")

    out = []
    for i, tensor in enumerate(tensors):
        if tensor.ndim == 2:
            tensor = tensor.reshape(1, *tensor.shape, 1)
        elif tensor.ndim == 3:
            if i == 0:
                tensor = tensor.reshape(1, *tensor.shape)
            elif i == len(tensors) - 1:
                tensor = tensor.reshape(*tensor.shape, 1)
            else:
                raise ValueError(f"Interior MPO tensor {i} has no two bonds: {tensor.shape}")
        elif tensor.ndim != 4:
            raise ValueError(f"MPO tensor {i} should have ndim 2, 3, or 4, got {tensor.shape}")
        out.append(tensor)
    return out

