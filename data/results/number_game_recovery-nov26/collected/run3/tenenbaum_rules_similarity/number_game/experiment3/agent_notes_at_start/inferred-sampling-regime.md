---
name: inferred-sampling-regime
description: exp2 iter_4 c3: set-level strong/weak sampling mixture θ≈0.72, LOO −0.2 vs incumbent, RMSE 0.006; kind-specific θ (rules vs ranges) was a dup
metadata:
  type: project
---
exp2 iter_4/candidate_3 (2026-10-31), refining mixed_strong_weak_sampling_concepts. Submitted inferred_sampling_regime_concepts: P(X|h) = θ·Π strong + (1−θ)·Π weak, taken over the whole set with θ~Beta(2,2), so the learner infers the sampling regime per set. Quick 2x300 fit: θ 0.72±0.15, LOO −2273.68 vs incumbent −2273.49, p_yes RMSE 0.006 (max 0.022). Check OK, 55 s smoke.
Tried first: separate θ for rules and ranges (0.87 and 0.91), LOO −0.5, RMSE 0.003, so a novelty dup and not submitted.
Incumbent residuals were flat by set size, target-in-set, p-bin and distance (scratch/resid.py).
**Why:** sampling-assumption variants are all ≈null on exp2 ([[mixed-strong-weak-sampling]]).
**How to apply:** check whether this one passed novelty before trying more sampling variants.
