---
name: first-example-anchored-reliability
description: Order model for exp2: first example has its own trust a1, later ones Beta-marginalised — (m, first_in, k, s) table
metadata:
  type: project
---
experiment2 iter_0/candidate_2 (2026-11-01): first_example_anchored_reliability = family_pooled_uncertain_reliability, but example 1 gets an independent reliability a1 and only examples 2..n enter the Beta(alpha,kappa) table. Table log L[m,first,k,s] = first_log[first,s] + T[m,k,s] for m = n-1 = 0..3; interval counts are (sets,2,4,100). Exp2 data contain stray examples in varying positions (e.g. 92,82,27,42).
Repair (candidate_2_repair_1): original (46 rule_dev + tau funnel) timed out at 30 min. Fix: drop rule_dev/tau, rules equal within family, family_level ZeroSumNormal -> 11 params, smoke 72 s/200 iters under load 22, check OK, smoke loglik -2035.8.
**Why:** the brief asked for an order effect; WSLS variants are already live, and serial-position weighting was a near-dup in exp1.
**How to apply:** check iter_0 admission/ELPD for the outcome. See [[wsls-uncertain-reliability-graft]], [[family-pooled-reliability-graft]].
