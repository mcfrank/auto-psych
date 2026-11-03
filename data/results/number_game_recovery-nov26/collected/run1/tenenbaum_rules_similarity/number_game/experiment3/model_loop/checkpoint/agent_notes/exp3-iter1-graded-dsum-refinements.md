---
name: exp3-iter1-graded-dsum-refinements
description: Exp3 iter_1 cand_3 MAP probes on graded_prior_digit_sum_concepts (LL -3513.54) — primes own family +1.3 (submitted); quadratic dsum prior, range-location slope, graded multiples, powers own family all dead
metadata:
  type: project
---
2026-11-01, exp3 iter_1 candidate_3, base graded_prior_digit_sum_concepts MAP LL -3513.54 (dsum_decay -0.25, scale 79, kappa 0.017; contains-digit and exception weights 0):
- primes own family: -3512.26 (+1.3), RMSE vs base 0.0095 → submitted as graded_digit_sums_primes_family.
- quadratic term in digit-sum prior: +0.19. Range-centre location slope on the range prior: +0.18. Multiples prior exp(-d*m): d≈0, +0.13. Powers own family: +0.07. All dead.
Probe harness: candidate_3/scratch/p/map.py (find_MAP + datalogp, ~20 s per model).
**How to apply:** concept space and priors on this base are saturated apart from primes; see [[exp3-concept-family-probes]], [[exp3-iter0-centre-weighted-probes]].
