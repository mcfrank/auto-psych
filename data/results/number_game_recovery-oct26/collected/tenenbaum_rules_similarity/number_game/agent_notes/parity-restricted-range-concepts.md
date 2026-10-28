---
name: parity-restricted-range-concepts
description: exp3 iter_1 c4: parity critique is an over-prediction of opposite-parity outside-span yes (n=2 sets); parity tweaks all <0.4 nats at MAP
metadata:
  type: project
---
2026-10-22, exp3 iter_1 candidate_4 (refinement of the incumbent): submitted `parity_restricted_range_concepts`. Even/odd-within-range conjunctions get their own Dirichlet share, not diluted by rule weight. MAP LL -3090.13 vs the incumbent's -3090.50; RMSE 0.0036; check OK with smoke LL -3094.2 and 310 s per sampling run.
The parity_match_outside_span_gap discrepancy comes from the model over-predicting yes to opposite-parity targets outside the span, mostly for 2-example sets (obs 0.39 vs model 0.48). Same-parity predictions are well calibrated.
MAP probes that did not move the stat (0.183→0.188, obs 0.248): parity as its own kind (+0.16), no parity conjunctions (0), conjunctions restricted to parity only (+0.25), free Erlang shape (fit ≈2.1, -0.3).
**Why:** the incumbent family is saturated, see [[incumbent-near-ceiling-exp2]]. **How to apply:** don't retry parity-prior tweaks. Scripts are in iter_1/candidate_4/scratch (cmp.py, brk2.py).
