# Local TEMPO MPS Backend Plan

The goal is to implement `backend="mps"` using only
`quante.tensornetwork`, while keeping the style of this package: shallow,
explicit, and close to the existing `TempoEngine.run()` / `measure()` pattern.

We use OQuPy as a reference for the tensor-network logic, but we do not copy its
backend class hierarchy or naming.

## OQuPy Logic To Keep In Mind

OQuPy's useful flow is:

```text
Tempo.compute(end_time)
  _check_time(end_time)
  backend.initialize()
    initialize_mps_mpo()
      build basis transforms
      build north/west boundary vectors
      build influence tensors/MPO
      initialize MPS from rho0
  loop:
    backend.compute_step()
      prop_1, prop_2 = propagators(step)
      compute_system_step(step, prop_1, prop_2)
```

The core step does:

```text
1. choose the influence MPO for this time step
2. contract the west boundary into the MPO
3. apply the first half-step propagator to the current MPS
4. contract one north boundary if MPS/MPO lengths need alignment
5. apply the influence MPO to the MPS
6. SVD/canonicalization sweep
7. append the second half-step propagator as the newest memory site
8. contract north boundaries on a copy to read out rho
```

For quante, this should become private methods on `TempoEngine`, not a separate
`TempoBackend`.

## File Layout

Keep everything TEMPO-specific in:

```text
quante/tensornetwork/opensystem/tempo.py
```

Do not add `tempo_backend.py` for now.  Do not add generic `split` / `join` to
`TensorTrain` unless a second algorithm also needs them later.

Only move a helper to `core/tensor_operations.py` if it becomes a genuinely
generic MPS/MPO contraction.  In the first implementation, the north/west
boundary operations can stay as private helpers in `tempo.py`, because their
meaning is TEMPO-specific.

## Existing Tools To Reuse

```text
MPS
  from_vector, to_vector
  apply_gate_
  apply_mpo_
  canonicalize_, orthogonalize_

MPO
  from_matrix, from_eye

TensorTrain
  apply_mpo_
  apply_mpo_naive_
  canonicalize_

tensor_operations
  _full_contract_right_mps
  _full_contract_two
  _apply_on_mps_step
  canonicalize
```

The intended replacement for OQuPy's `zip_up` is mostly:

```python
mps.apply_mpo_(mpo, trunc_para=trunc_para)
```

plus small TEMPO-specific preparation steps to make the MPO length and boundary
legs match the current memory MPS.

## TempoEngine State

For `backend="mps"`, `TempoEngine` should own all state directly:

```python
self.adt             # MPS storing the augmented density tensor
self.influence_mpo   # reusable influence MPO tensors
self.super_u
self.super_u_dagg
self.trunc_para
```

Initialization should prepare reusable objects only.  It should not propagate
to the final time.

## Private Methods To Add In `tempo.py`

### MPS Initialization

```python
def _init_mps(self):
    """Prepare local MPS TEMPO objects without advancing time."""
```

Responsibilities:

```text
1. build coupling-basis transforms
2. set D = dim**2
3. build the influence MPO tensors
4. initialize adt as one-site MPS from rho0
```

### Propagators

```python
def _mps_propagators(self, step):
    """Return first and second half-step propagators for a TEMPO step."""
```

First version can use the same time-independent half propagator twice:

```python
prop_1 = self.system.half_propagator(dt)
prop_2 = self.system.half_propagator(dt)
```

Keep the method boundary because later time-dependent systems need it.

### Influence MPO

```python
def _build_influence_mpo(self):
    """Build reusable four-leg influence tensors in local MPO leg order."""

def _select_step_mpo(self, step):
    """Return the current-step influence MPO, shortened or shifted by memory."""
```

Use the existing `MPO` convention:

```text
(left_bond, physical_out, physical_in, right_bond)
```

For `dk=0`, include the coupling-basis transforms, following OQuPy's idea:

```text
infl_four_legs
  -> super_u_dagg on the input leg
  -> super_u.T on the output leg
```

For `dk>0`, use the plain four-leg influence tensor.

There is no separate `_create_delta()` helper in the first version.  Build the
four-leg influence tensor directly when constructing `self.influence_mpo`.

### Boundary And Old-Leg Contractions

OQuPy expresses some sums as contractions with boundary vectors named
`sum_north` and `sum_west`.  In quante we do not need to store those vectors as
state.  Use direct `np.sum`-style contractions over the appropriate tensor axis.

Keep these private in `tempo.py` first:

```python
def _sum_mpo_left_boundary(mpo):
    """Close the left/west boundary of a TEMPO MPO by summing that axis."""

def _sum_adt_oldest_leg(adt):
    """Trace/sum the oldest memory leg of the ADT MPS."""
```

These functions are TEMPO-leg-convention code, so they should not go into
`TensorTrain` yet.

### Local Join

Keep this private in `tempo.py` first:

```python
def _append_site_to_mps(mps, site):
    """Append one MPS site to the right side of the memory chain."""
```

This replaces OQuPy's `join(self._mps, prop_2_na)`.

### Applying The Network

```python
def _apply_first_half_step(self, prop_1):
    """Apply the first half propagator to the newest/current memory leg."""

def _apply_influence_mpo(self, mpo):
    """Apply the current influence MPO using MPS.apply_mpo_."""

def _append_second_half_step(self, prop_2):
    """Grow the memory chain by appending prop_2 as the newest/next site."""
```

The first implementation should try to express the zip-up steps using existing
MPS/MPO methods:

```python
self.memory.apply_mpo_(mpo, trunc_para=self.trunc_para)
self.memory.canonicalize_(trunc_para=self.trunc_para)
```

If a one-site propagator is simpler as a local gate, use `apply_gate_`.

### Readout

```python
def _readout_mps(self):
    """Sum old ADT legs and return rho."""
```

This should contract a copy of `self.memory` so the stored ADT is unchanged.

## `_run_mps` Shape

Target flow:

```python
def _run_mps(self):
    if self.cur_step >= len(self.ts):
        return self.rho

    if self.cur_step == 0:
        self.cur_step += 1
        return self.rho

    prop_1, prop_2 = self._mps_propagators(self.cur_step - 1)
    mpo = self._select_step_mpo(self.cur_step)
    mpo = self._sum_mpo_left_boundary(mpo)

    self._apply_first_half_step(prop_1)

    if len(self.adt) != len(mpo):
        self._sum_adt_oldest_leg(self.adt)

    self._apply_influence_mpo(mpo)
    self.adt.canonicalize_(trunc_para=self.trunc_para)
    self._append_second_half_step(prop_2)

    self.rho = self._readout_mps()
    self.cur_time = self.ts[self.cur_step]
    self.cur_step += 1
    return self.rho
```

The names do not need to match OQuPy; this structure is just a checklist for
the tensor-network operations.

## Tests

### TEMPO Private Helper Tests

Add tests in the existing opensystem tempo tests rather than making a backend
test file:

```text
tests/tensornetwork/test_opensystem_tempo.py
```

Tests:

1. `_init_mps()` creates a one-site `MPS`.
2. `_build_influence_mpo()` creates `memory_steps + 1` MPO tensors.
3. `_select_step_mpo(step)` returns the expected shortened length.
4. `_sum_mpo_left_boundary()` agrees with explicit `einsum`.
5. `_sum_adt_oldest_leg()` agrees with explicit `einsum`.
6. `_append_site_to_mps()` increases length and preserves simple contractions.

### Physics Tests

Keep benchmark tests small:

```text
tests/tensornetwork/test_opensystem_tempo_benchmark.py
```

Tests:

1. zero correlation: `backend="mps"` equals `backend="system"`.
2. no truncation: `backend="mps"` equals `backend="reference"` for tiny memory.
3. small spin-boson case agrees with OQuPy within reference tolerance.
4. trace and Hermiticity remain stable.

## Implementation Order

1. Restore `backend="mps"` initialization, but only prepare objects.
2. Implement direct four-leg influence MPO tensor construction.
3. Implement private boundary/old-leg sums in `tempo.py`.
4. Implement `_select_step_mpo`.
5. Implement one-site propagator application.
6. Implement influence MPO application via `MPS.apply_mpo_`.
7. Implement append/readout.
8. Turn `_run_mps()` on.
9. Restore MPS benchmark tests.
10. Update the quickstart notebook to use the real local MPS backend.

## Design Notes

- Keep `reference` backend as the correctness oracle.
- Keep OQuPy only in tests/benchmarks and notebooks, never in runtime code.
- Keep heavy computation out of `TempoEngine.__init__`.
- Avoid a separate `TempoBackend` layer unless the single `TempoEngine` becomes
  genuinely too large.
- Keep TEMPO-only helpers private in `tempo.py` until another algorithm needs
  them.
