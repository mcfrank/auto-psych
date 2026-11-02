---
name: experiment2-data-structure
description: experiment2 responses = 10 participants x 450 trials, 2 stimulus orders; PPC keeps row order; shown-example targets get only 62% yes
metadata:
  type: project
---
experiment2 (2026-10-30): responses.csv is 4500 rows = 10 participants × 450 (participant = cumsum(trial_index==0)), only 2 distinct stimulus sequences; 119 sets. The PPC harness (src/critique/ppc.py) copies the CSV and overwrites only `rating`, so row-order/participant/lag statistics are valid.
Raw observations: when the target is one of the shown examples (85 trials) the yes rate is only 0.62 — the lapse-free incumbent predicts ~1, so a lapse/guessing term looks likely to be needed. Session late-minus-early yes rate -0.06; participant yes-rate SD 0.087.
**Why:** iter_0 critique statistics were built on this structure.
**How to apply:** check the iter_0 ppc_results.json for which of these turned out significant; see [[round4-null-refinements]] for the experiment1 data layout.
