---
name: stray-closing-example
description: exp3 iter_3 c0: last-shown example may be a stray (mixture with prefix-only reading) on sequence_opens_bottom_ends_top; raw example yes first 0.91 / mid 0.84 / last 0.74
metadata:
  type: project
---
2026-10-22, exp3 iter_3 candidate_0 (exploratory slot, brief asked for order effects). In raw data, in sets of 2 or more, shown examples get yes 0.91 when shown first, 0.84 in the middle and 0.74 when shown last. Submitted `stray_closing_example`: the incumbent plus a stray_prob ~ Beta(1,4) mixture. Under the stray reading, the last example has likelihood 1/100 and the posterior is computed from the prefix (with edge marks read from the prefix). It is built as a second set-quantity reading stacked on axis 1, so the hypothesis count doubles.
**How to apply:** check its LOO and stray_prob. If it wins, try position-graded stray probabilities. Related: [[exp3-example-rejection-critique]], [[limited-capacity-example-memory]], [[sequence-opens-bottom-ends-top]]
Check OK: smoke LL -3093.1 (incumbent about -3093.3); about 170 s per sampling run after switching to the batched_dot BLOCK=8 layout. The dense posterior[cell_set] gather took 0.19 s per gradient and the first check timed out; with batched_dot it is 0.076 s. Pitfall again: `pkill -f "until grep"` killed its own shell.
