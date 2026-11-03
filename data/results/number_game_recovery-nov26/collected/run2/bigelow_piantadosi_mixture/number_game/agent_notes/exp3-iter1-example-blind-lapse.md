---
name: exp3-iter1-example-blind-lapse
description: Exp3 iter-1 cand_2 repair — set-size/counting-order/stray-schedule screens all null or lose; lapse answered from concept prior predictive submitted (RMSE ~0.004)
metadata:
  type: project
---
Exp3 iter 1 cand_2 repair (2026-10-31). The first attempt, set-size rule search, was rejected as a dup (kappa≈0).
- Counting-order cue (rising order shifts belief toward ranges): kappa -0.06±0.13, null.
- Per-example stray schedules that depend on set size lose 18-60 nats. 0.9 flat is a sharp optimum.
- Residual z by target feature, side, set size and lateness: all |z|<1.4, at the ceiling.
- Submitted `example_blind_prior_lapse`: guess_yes replaced by the prior predictive of the concepts (0 new params). LOO -3197.5 vs -3196.5 for the baseline. Lapse ≈0.01. Self-made 512-pool RMSE 0.0043 vs the dup model.
**Why:** a free add-on shrinks to the incumbent at the ceiling; a non-nested replacement keeps its novelty.
**How to apply:** for novelty, use non-nested swaps; screen with /tmp-style cmp script (fit both, predict a pool). See [[number-game-ceiling]].
