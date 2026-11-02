---
name: exp2-iter2-heading-direction
description: Exp2 iter-2 cand_1 — order-graded forgetting null; last-step heading range prior +2.0 nats MAP submitted; beta**0 NaN-gradient pitfall
metadata:
  type: project
---
Exp2 iter 2, candidate_1 (2026-10-30), exploratory slot, on top of digit_sum_concepts. The numpy MAP NLL is 1925.26 at base.
Residuals: far out-of-range targets on the side of the LAST example get z=+2.4; on the first example's side, z=-1.5.

Screens:
- recency or primacy forgetting (a mixture over retained subsets): f→0, null.
- ranges centred on the last example, or on the first: ≤0.5 nats.
- trend direction sign(last - mean of the earlier examples) × (room above - room below)/100 as a per-set normalized range prior: +1.0.
- the same with the **last step** sign(x[-1]-x[-2]): **+2.0** (beta 0.83). Submitted as `heading_direction_range_extrapolation`.
- the same factor as a proper direction likelihood (2σ), or unnormalized: +0.5 / +0.4.

Novelty RMSE vs the incumbent is 0.0105. Smoke loglik -1927.1, 5 s per 200 iterations. Implementation: Taylor moments per (set, width, k) and per cell, then pt.dot.

Pitfall: `beta ** np.arange(K)` gives a NaN gradient at beta=0 (the k=0 term). Concatenate ones(1) with beta**arange(1,K).
Also, pytensor.function / compile_fn on this model returned constant p. Verify with a numpy re-implementation instead.

See [[exp2-iter2-critique]], [[exp2-iter1-range-location-prior]], [[fit-speed-budget]].
