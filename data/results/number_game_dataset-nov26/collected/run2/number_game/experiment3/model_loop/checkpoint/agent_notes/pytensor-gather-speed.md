---
name: pytensor-gather-speed
description: In these PyMC candidates, gathering a parameter-dependent tensor by integer index is slow; one-hot matmul is ~5x faster
metadata:
  type: project
---
Indexing a large parameter-dependent tensor like post[cell_set] (shape cells×rules×widths) made the gradient 37 ms per eval. Changing it to a one-hot (cells×sets) matrix times post reshaped to (sets, -1), and pooling the widths into 24 log-spaced bands, brought it down to 7 ms. The incumbent rule_within_range_conjunction takes about 3 ms. The check_candidate smoke fit (200 iterations) takes about 4 min at 37 ms and 47 s at 7 ms. Full admission allows 30 min per run.
**Why:** admission rejects slow fits; I found this on 2026-11-01 (iter_1 candidate_3).
**How to apply:** benchmark compile_dlogp before running the check. Compute per distinct set, then send results to cells with a one-hot matmul.
