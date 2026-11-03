---
name: exp3-iter1-two-sided-margin
description: Exp3 iter-1 cand_4 repair — upward-reach-replaces-heading was a dup (RMSE 0.0008) of upward graft; two-sided margin graft on Weber-heading +1.07 MAP, submitted
metadata:
  type: project
---
Exp3 iter 1, candidate_4_repair_1 (2026-10-31). The first attempt, upward room replacing heading, was rejected as a near-dup of heading_weber_with_upward_counting (RMSE 0.00079), because heading beta is about 0 on exp3.

The repair is `heading_weber_two_sided_margin`. It is the upward-counting file with coef_b = taylor_coef(delta - h*beta), so the exponent is delta*(above+below) plus the heading term.
- Numpy MAP: 3188.83 against 3189.90 for the base and 3189.30 for upward-only, with delta 0.28.
- Data-cell RMSE is 0.0076 against the base and 0.0034 against upward-only.
- Smoke run gave -3192.7 in 17 s.

**How to apply:** since heading is null on exp3, any variant that only replaces heading duplicates whatever else carries the same room term. See [[exp3-iter1-upward-graft]].
