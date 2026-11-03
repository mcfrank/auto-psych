---
name: exp3-weber-ranges-and-collapse
description: exp3 iter1 cand0 repair — free Weber range exponent fits -0.04 (collapses); fixed Weber -22 MAP; p**g readout null; nested params all collapse on fives incumbent
metadata:
  type: project
---
2026-10-31, exp3 iter_1/candidate_0_repair_1 (`weber_scaled_range_widths`). The base is the fives incumbent, MAP loglik -3435.8.
- Erlang range scale × (mean example/40)^w, with w free: w = -0.04, loglik -3435.66. This would be rejected as a duplicate.
- With w fixed at 0.5 the loglik is -3443.8. With w fixed at 1 (the version submitted) it is -3457.6.
- A p_raw**g readout gives g = 1.03, which is null. The lapse fits about 0 and beta 0.85.
- candidate_0 (limited retrieval) was rejected as a duplicate (RMSE 0.00074), because its K fits near the nested limit.

**Why:** on exp3, any nested one-parameter add-on to the fives incumbent fits back to the incumbent and fails the novelty gate (RMSE < 0.002).
**How to apply:** in a repair slot, MAP-screen the new parameter first. If it collapses, submit a fixed-form, non-nested claim. Related: [[limited-retrieval-generation]], [[exp3-iter1-exploratory-screens]].
