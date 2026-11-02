---
name: exp3-iter2-llo-confidence-distortion
description: Exp3 iter-2 cand_1 — target-encoding noise, size exponent, non-uniform/corrupted strays, kind temperature, likelihood power all sharp at normative; LLO swap for lapse submitted
metadata:
  type: project
---
Exp3 iter 2, cand_1 (2026-10-31). These were MAP screens on heading_weber_with_upward_counting, whose ref NLL is 3189.30:
- Target-encoding noise (±1 confusion, Weber Gaussian, or constant Gaussian): the noise goes to 0. Null.
- Size-principle exponent (all, finite, or interval concepts): sharp at 1. Values of 0.9 or 1.1 cost 25-45 nats.
- Stray distribution ∝ x^-e: uniform is best. e=±0.5 costs about 25 nats.
- Strays as single-digit corruptions of members: worse at every weight w (w=0.2 costs 3.7, w=1 costs 78).
- Rule-vs-range kind temperature: τ≈0.98, +0.08 nats, RMSE 0.0019. A dup.
- Likelihood power ρ: sharp at 1.
- Linear-in-log-odds response map replacing lapse+guess: slope≈0.98, bias≈0.06, -1.46 nats, row RMSE 0.0053. Submitted as `log_odds_confidence_distortion`, smoke -3193.8 in 8 s.

Scripts are in iter_2/candidate_1/scratch. base.py is a numpy port of the incumbent; pc_full gives the full sets×100 matrix.
**How to apply:** exp3 is at the ideal observer for every parameterised distortion. To get novelty, use non-nested swaps, as in [[exp3-iter1-example-blind-lapse]].
