---
name: kind-disagreement-target-criterion
description: exp3 iter_2 c4: per-target kind-disagreement criterion (between-kind variance share), shift≈0.10±0.10, RMSE 0.0076 vs incumbent
metadata:
  type: project
---
exp3 iter_2/candidate_4 (2026-10-31), kind_disagreement_target_criterion: the incumbent kind_uncertainty_shifted_criterion, but its set-level kind entropy is replaced by a per-target share: between-kind variance of membership / p(1-p) (law of total variance; computed as sum_k num_k^2/P(k) − p^2 with a family one-hot matmul).
Smoke OK, ll −3653.6, 17 s sampling. 1×200 fit: shift +0.10±0.10 (null-ish); RMSE of p_yes vs the incumbent on data trials 0.0076 (not checked on the novelty pool). Quick-fit script: /tmp/c4fit.py cand.py responses.csv other.py.
**How to apply:** target-specific uncertainty readouts also land at ≈0 on exp3; see [[kind-uncertainty-shifted-criterion]], [[uncertainty-damped-decisiveness]].
