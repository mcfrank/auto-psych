---
name: exp3-iter2-with-replacement-swap
description: Exp3 iter-2 cand_4 — on the upward-counting incumbent, proportional reach and ambiguity graft are dups; with-replacement sampling ties (MAP -0.06) with pool RMSE 0.0067, submitted
metadata:
  type: project
---
Exp3 iter 2, cand_4 (2026-10-31), refinement slot on heading_weber_with_upward_counting. MAP LL of the incumbent: -3190.24 (gamma -0.07, beta 0.06, delta 0.19, so every add-on is nearly dead).
- Proportional (log hi/max) upward reach in place of absolute: -3190.47, pool RMSE 0.0018, so a dup.
- kind-ambiguity kappa graft: kappa→0.009 at MAP, a dup.
- With-replacement sampling (remaining = SIZE, 0 params): -3190.30, pool RMSE 0.0067 vs the incumbent. Submitted as `heading_weber_upward_with_replacement`, smoke -3193.2.
**How to apply:** WOR vs WR is a cheap non-nested swap that keeps novelty; the scratch pool/MAP RMSE scripts are in iter_2/candidate_4/scratch (pool3.py). See [[exp3-iter1-example-blind-lapse]], [[number-game-ceiling]].
