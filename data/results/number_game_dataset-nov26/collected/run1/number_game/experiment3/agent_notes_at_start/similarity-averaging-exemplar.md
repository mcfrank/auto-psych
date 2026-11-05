---
name: similarity-averaging-exemplar
description: exp2 iter1 c4 repair — partial averaging of similarity over examples (−kappa·log n) on common_divisor_sharpened incumbent, +10.6 nats MAP
metadata:
  type: project
---
Experiment 2, iter_1 candidate_4_repair_1 (2026-11-02), `similarity_averaging_exemplar`. The first attempt, a factor-kinship variant, was rejected as an exact duplicate of divisor_family_exemplar. Candidates c2–c5 all converged on that same gcd-kinship idea, so avoid it.
The change: log summed similarity minus kappa·log(n_examples), with kappa ~ N(0.5, 1).
MAP log-lik, against the incumbent's -2714.4:
- Free kappa: -2703.8 (kappa ≈ 0.75).
- Pure mean (kappa = 1): -2709.6.
- No gain: log-magnitude distance (-2716), Weber exponent (-2714.5), gradient-shape exponent (-2713.9), free breadth for single examples (-2712.9), spread×(n-1) (-2734).
- Setting singles' placeholder spread to 0 instead of 0.5 costs 21 nats, so single examples need broad generalization.
Smoke fit: -2708.4.
**Why:** the background_set_rate_sd critique. Residuals show over-predicted background for sparse incoherent pairs ("25,82", "50,2", "3,33,99") and under-predicted background for singles and for "4,66,78,96".
**How to apply:** this offset could be grafted onto [[multiple-route-exemplar]]. See [[exp2-iter1-critique-stats]].
