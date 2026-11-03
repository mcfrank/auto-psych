---
name: representative-coverage-sampling
description: exp3 iter_4 c2: evidence × (share of concept inside examples' span)^kappa on beta_share; kappa≈-0.19±0.14, smoke OK ll −3653.2
metadata:
  type: project
---
exp3 iter_4 candidate_2 (2026-10-31), representative_coverage_sampling, built on beta_share_width_exception_concepts. Rule coverage = |h∩[min,max]|/|h| (exact, per set); ranges use min(1, span/L) per size (exact for intervals containing all examples). One-example sets get coverage 1 (no spread).
2×300 NUTS: kappa −0.19±0.14 (slightly anti-representative, i.e. null-ish), converges; smoke 15 s. Novelty not checked — dup risk vs beta_share / range_size_misperception if kappa≈0.
The critique hit (late_vs_early_discrimination, see [[experiment3-iter4-critique-stats]]) needs trial_index, which is forbidden, so this exploratory slot ignored it.
