---
name: parity-preserving-multiple-route
description: exp3 iter1 c2 — parity mismatch cuts the multiples route on whole_set_round_category, +22 MAP log-p; smoke -4083.8
metadata:
  type: project
---
Exp3 iter_1 candidate_2 (2026-11-02), `parity_preserving_multiple_route`. Residuals on the incumbent's single-example sets: even multiples of odd examples were over-predicted (17→34/68, 43→86, 31→62, 9→6/12), and multiples of even examples or odd multiples were under-predicted (22→44/66/88, 33→99, 14→70, 16→64). The candidate multiplies the multiples route by exp(-gain·c_par_route·pdiff). MAP log-p: incumbent -4122.7; gating only the multiples route -4100.9 (submitted); gating all three routes -4104.5; the same gate using c_par -4110; a hard 0/1 gate -4154; parity_homog=1 for single examples -4127 (worse). Set-mean residual SD on the incumbent is 0.13, about twice binomial. Smoke fit took 25 s.
**Why:** this addresses the polarization critique ([[exp3-iter1-critique-stats]]) without per-set random effects ([[common-divisor-membership-route]]).
**How to apply:** if it's admitted, look next at the leftover round residuals (95→10, 50→90 over-predicted).
