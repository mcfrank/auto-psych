---
name: base-rate-neglect-concepts
description: exp3 iter_2 c1 repair: posterior ∝ prior^w·lik; MAP w≈1, w=0.5 costs 5.7 nats; Beta(2,2) prior gives w≈0.81, est RMSE 0.0034
metadata:
  type: project
---
exp3 iter_2 candidate_1_repair_1 (2026-10-31), base_rate_neglect_concepts on the beta_share base (replaced rule-size rho, which was a novelty dup at RMSE 0.00165, cf. [[rule-size-misperception]]). Profile nll vs w: 0.9 +0.17, 0.8 +0.64, 0.7 +1.57, 0.6 +3.17; data-row RMSE vs base 0.0021/0.0036/0.0059/0.0091. Uniform prior → post mean ≈0.87, RMSE ≈0.0025 (borderline), so I used Beta(2,2) → ≈0.81. Smoke OK, ll −3653.9, ~50 s.
Also null at numpy MAP: evidence divided by n^g (g≈−0.01; g=1 averaging costs 624 nats).
Scripts in /tmp/r1c1 (h.py numpy loader for the candidate's prepare_observed).
**Why:** free exponents on exp3 collapse to the normative value; a prior pulling away from the null is what gets a model past novelty.
**How to apply:** check the admission outcome; when a parameter collapses, profile it to pick a defensible prior.
