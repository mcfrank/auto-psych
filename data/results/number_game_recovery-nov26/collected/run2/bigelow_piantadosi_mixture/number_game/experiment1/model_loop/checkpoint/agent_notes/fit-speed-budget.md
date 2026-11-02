---
name: fit-speed-budget
description: Admission fit is 7000 iters/chain warm-start NUTS; per-gradient cost must be ~<25ms; tricks that worked
metadata:
  node_type: memory
  type: project
  originSessionId: 1d82c78b-0bac-4618-97c8-da7bd32baa1b
  modified: 2026-10-30T21:06:41.047Z
---

Admission fits use production draws 4000 + tune 3000 per chain (warm_start_nuts, dense mass matrix), 30-min limit per run. Machine has 1 core. Incumbent `tenenbaum_digit_concepts` logp+grad ≈ 12 ms. A memory-subset mixture over its hypothesis space (≈700 states) cost 62 ms naive → too slow (~49 min).

**Why:** the check_candidate smoke fit (200 iters) passes even when the full fit would time out.

**How to apply:** benchmark logp+dlogp time and NUTS n_steps (probe `_warm_start_sample` with small draws) before finishing. Speedups that worked: replace gathering posterior rows per cell (slow AdvancedIncSubtensor1 gradient) with a `pytensor.sparse.CSR` + `structured_dot` (44→25 ms); declaring module-level `SAMPLER_SETTINGS = {"target_accept": 0.9}` halves steps (10.8→5.3). Don't name a scratch script `timeit.py` (shadows stdlib, breaks pymc import). Related: [[digit-concepts-key]].

Also worked (2026-10-30, `order_weighted_concept_learning`): when intervals must be grouped by (size, per-position hit pattern) instead of (size, k), keep only each set's non-empty groups (≤473 of 1600 on this data) padded per set, and gather size-dependent terms by index — 47 → 18 ms.
