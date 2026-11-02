---
name: digit-slip-stray-refinement
description: Stray examples as miswritten concept members (digit slips) in the number-game incumbent — about +0.4 to +1 MAP, distinct; the magnitude near-miss kernel gave nothing
metadata:
  type: project
---
iter_4/candidate_3 (2026-10-30), refining uncertain_reliability_noise_concepts (MAP log-likelihood -815.9). In this change a stray is uniform with probability 1-eps, and with probability eps it is a "slip": uniform over the non-members that a member could produce by an edit.
- Magnitude kernel r^|x-y| (r = 0.3, 0.6, 0.85): MAP -815.8 to -816.0, no gain.
- Damerau-1 digit edits (one digit changed, added or dropped, or adjacent digits swapped): MAP -815.44, eps ≈ 0.22, novelty-pool RMSE 0.017. Submitted as digit_slip_stray_concepts. Check OK, 16 s smoke fit, smoke log-likelihood -830.1.
- A buggy 2-digit zip version scored -814.9. It made 100 a slip of 11-19 and of 20, 30, ... 90, which hints that the 100 in set 31,19,49,100 (110 rows) drives some of the gain.
Speed trick: the per-example product is a polynomial in (genuine, slip, uniform), so prepare_observed precomputes the C[g,t] coefficients (5x5) per set, per rule and per cell. The graph only weights them, so it stays fast.
**Why:** this is another stray-shape change that still moves predictions; see [[complement-stray-refinement]].
**How to apply:** the scratch harness is in iter_4/candidate_3/scratch (scr.py + var.py, compared against base.py).
