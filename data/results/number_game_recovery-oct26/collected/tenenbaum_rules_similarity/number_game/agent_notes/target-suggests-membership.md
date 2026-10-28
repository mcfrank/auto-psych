---
name: target-suggests-membership
description: exp3 iter_4 c2 repair: query-as-hint (target added as fractional example) on incumbent; MAP hint 0.015, LL -3090.3, RMSE 0.0057; fallible-verification and fuzzy-magnitude probes null/bad
metadata:
  type: project
---
2026-10-22, exp3 iter_4 candidate_2_repair_1. The carryover model failed novelty (RMSE 0.0006) because the novelty pool has no trial_index, so every sequential model collapses to its base there. Never propose sequential or trial-order models.
Probes run on the incumbent sequence_opens_bottom_ends_top:
- Fallible arithmetic verification (computed-rule membership errs at rate err*t/100): MAP err 0.004, RMSE 0.0009. Null.
- Gaussian "numbers around N" concepts in place of ranges: MAP LL -3375, about 280 nats worse. Sharp edges are needed.
- Submitted `target_suggests_membership`: the target counts as an extra example with weight hint. It is cheap and exact: yes = H/(1+H-P), where H = sum post*m*exp(hint*log1p(900/size)). Conjunction size is approximated as |rule|*W/100. MAP LL -3090.3 (≈ incumbent), hint 0.015, RMSE 0.0057. Large gain factors make it novel even at a small hint weight.
Check OK: smoke LL -3093.5, 108 s per run. Replacing the two batched_dot calls with one pt.matmul on stacked [posterior, boosted] took 0.092 s per gradient down to 0.074.
**Why:** this mechanism (pruned on the old exp2 base) is not covered by any live model and it passes the RMSE proxy. **How to apply:** check this candidate's admission outcome. Related: [[previous-trial-concept-carryover]], [[pymc-model-speed-patterns]]
