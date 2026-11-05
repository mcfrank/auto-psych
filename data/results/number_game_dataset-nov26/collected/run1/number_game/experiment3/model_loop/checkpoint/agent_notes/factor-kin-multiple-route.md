---
name: factor-kin-multiple-route
description: exp2 iter2 c3 — additive gcd>=3 non-multiple/non-divisor kin route on multiple_route_exemplar, MLE -8 nats; smoke -2687.5
metadata:
  type: project
---
Experiment 2, iter_2 candidate_3 (2026-11-02): `factor_kin_multiple_route` refines [[multiple-route-exemplar]].
The change is a second additive route: + exp(log_w_kin*gain)*[gcd(t,e)>=3, t is not a multiple of e, e is not a multiple of t], applied to every example. In-sample MLE NLL was 2682.4, against 2690.4 for the base model. The smoke fit was -2687.5.
Other variants screened:
- Kin for single examples only: 2683.8.
- Log-gcd graded: 2684.8.
- Route only from one-digit examples (weight × exp(-tau·[e>=10])): 2685.9, with tau → ∞.
- Times-table gate: 2686.9.
- log(e) scaling: 0 gain.
**Why:** this targets single_shared_factor_nonmultiple_contrast (q=0.04) from [[exp2-iter2-critique-stats]].
**How to apply:** the one-digit-only gate is an untried, roughly independent gain of about 4.5 nats (it addresses multiple_route_example_size_contrast).
