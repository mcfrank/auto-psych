---
name: range-misperception-uncertainty-criterion
description: exp3 iter_0 c5 repair: range_size_misperception + incumbent entropy shift; rho≈0.99±0.01, shift −0.21, pool RMSE 0.004 vs incumbent
metadata:
  type: project
---
exp3 iter_0/candidate_5_repair_1 (2026-10-31), range_misperception_uncertainty_criterion. The first attempt was rejected with RMSE 0.000 because it re-implemented the already-live mixed_sampling_uncertainty_criterion (candidate_3 had used the same idea).
Quick 2×300 fit: rho 0.990±0.011, uncertainty_shift −0.21±0.15, mean ll −3653.5 vs incumbent −3653.1. Novelty-pool posterior-mean RMSE vs incumbent 0.0040, vs range_size_misperception 0.0061 (its rho 0.994). Smoke check OK (62 s of sampling).
**Why:** on exp3, rho≈1 and shift≈−0.2 are both near-null. The graft only just passes novelty and is unlikely to gain.
**How to apply:** before a refinement slot, check that the idea isn't already a live model from an earlier slot in the same round. Novelty script: /tmp/nov6/nov.py (copy of iter_0/candidate_4_repair_1/scratch/nov5/nov.py). See [[mixed-sampling-uncertainty-criterion]], [[range-size-misperception]].
