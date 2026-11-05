---
name: incumbent-speedup-sparse-gather
description: How to make the exp3 incumbent-family PyMC graph ~2.3x faster per gradient with identical logp (sparse one-hot + softmax)
metadata:
  type: project
---
In the range_specific_evidence_strength-family models, ~1/3 of gradient time is AdvancedIncSubtensor1 from `posterior[cell_set]` (840 cells x ~1015 hyps), plus logsumexp's Isinf composite. Replacing with
`onehot = ps.CSR(pt.ones((n_cells,)), cell_set, pt.arange(n_cells+1), pt.stack([n_cells, n_sets]))` (import pytensor.sparse as ps) and `ps.structured_dot(onehot, pt.special.softmax(full, axis=1))` gives identical logp and 26 -> 11.5 ms per logp+grad. Padding cells per set instead (batched_dot) is NOT faster (max 35 targets/set). Smoke-check time barely drops (164 vs 175 s/chain) because short tuning dominates, but production fits should benefit.
**Why:** incumbent-based candidates keep getting dropped for >30-min admission fits.
**How to apply:** use this in any candidate built on the incumbent machinery. Scratch: experiment3/model_loop/iter_2/candidate_1/scratch (timing.py, prof.py, eq.py, map.py).
Related: [[experiment3-data-patterns]]
