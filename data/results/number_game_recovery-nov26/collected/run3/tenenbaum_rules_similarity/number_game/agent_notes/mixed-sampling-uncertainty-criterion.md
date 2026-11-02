---
name: mixed-sampling-uncertainty-criterion
description: exp3 iter_0 c5 refinement: mixed_strong_weak learner + incumbent's entropy-shifted criterion; exp3 blocks differ by design not people
metadata:
  type: project
---
exp3 iter_0/candidate_5 (2026-10-31): mixed_sampling_uncertainty_criterion = mixed_strong_weak_sampling_concepts + uncertainty_shifted_criterion's entropy shift on the logit (built by text-patching the live source).
Exp3 facts: the three participant blocks (yes ≈0.40/0.58/0.49) see different set lists (75/71/109 sets), and within-block yes-rate SD ≈0.01, so per-person bias is not needed. individual_rule_vs_range_tendency hard-codes N_MAX_PEOPLE=12 < 15 participants, so its −323 is probably a bug (every row falls back to an empty population average).
Only near-signal in the iter_0 critique: multiple_rule_vs_span_preference, one-sided p=0.04 (people favour multiples rules over ranges more than the model does).
Smoke check OK: 88 s sampling, ll −3653.3 (in-sample, not LOO).
**How to apply:** check novelty RMSE against both parents; they may be too close ([[mixed-sampling-prior-leak]]).
