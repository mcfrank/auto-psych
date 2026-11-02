---
name: exp2-iter2-sequence-direction
description: Exp2 iter-2 cand_2 — forgetting earlier examples rejected (forget→0); sequence-direction range credit +0.5 nats MAP, submitted; pytensor pow/cumprod NaN-grad-at-0 pitfall
metadata:
  type: project
---
Exp2 iter 2, candidate_2 (2026-10-30). The target was the recency residual from [[exp2-iter2-critique]]. Baseline digit_sum MAP NLL is 1925.26.
- Forgetting earlier examples (the last one always kept), with a constant or a position-graded forget rate: forget goes to 0. Forcing forget=0.05 costs 8 nats and 0.2 costs 87, so it is ruled out.
- Range credit exp(beta*a), where a = sign(last-first)*(extension above max - extension below min)/99:
  - unnormalized (an evidence factor): 1924.73 at beta 0.61, RMSE 0.0064 vs incumbent.
  - normalized per set as a prior: 1925.08.
  - "side of the last example" instead of travel direction: 1924.93.
- Submitted the unnormalized version as `sequence_direction_range_extrapolation`. Smoke loglik -1928.2. Grad 5 ms with 20 Taylor terms over moments per (set, width) and per observed cell.

**Pitfalls:** `beta ** pt.arange(K)` and `pt.cumprod` both give a NaN gradient at beta=0, so stack `[1, beta, beta**2, ...]` instead. To evaluate p_yes, compile `model.replace_rvs_by_values([p_yes])` with `inputs=model.value_vars`.

**How to apply:** this is another order effect worth only ≈0.5 nats ([[number-game-ceiling]]).
