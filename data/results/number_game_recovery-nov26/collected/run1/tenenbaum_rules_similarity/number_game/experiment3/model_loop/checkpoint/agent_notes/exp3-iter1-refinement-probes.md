---
name: exp3-iter1-refinement-probes
description: Exp3 iter_1 candidate_5 MAP probes on menu models — location prior and digit vocabulary dead; round edges +1 on cw digit sums (submitted); thresholds hurt the incumbent structure
metadata:
  type: project
---
iter_1 candidate_5 (2026-11-01, refinement slot), exp3 data. MAP LL from find_MAP (~15-35 s each); incumbent graded_prior_digit_sum_concepts -3513.54 (decay -0.25, scale 79, kappa 0.017).
- Base centre_weighted_ranges_digit_sums -3515.03.
  - + range location prior (linear in range centre): -3514.92, slope 0.13. Dead.
  - + digit-vocabulary family (all 1023 digit sets): weight 0, dead.
  - + round-number edge bonus: -3514.03, bonus -0.55 (RMSE vs incumbent 0.0098). Submitted as round_edge_ranges_digit_sums.
- centre_weighted_round_edge_ranges + graded digit sums: -3514.6, so the threshold family costs ~0.6 nats.
- Incumbent's worst sets: 27,37,67 (-12.9 nats; 70s targets 0.11 yes vs 0.03 predicted, maybe "contains 7", but the contains-digit family weight is 0), 99,59,31,67, 64,4, 81,9.
- No critique statistic was significant in iter_1 (largest: all-even sets with odd targets, z -1.7).
Probe script: /tmp/c5i1/map.py <model.py> <out.npy> (may not persist).
**How to apply:** exp3 is saturated, and round edges stay a stable ~1 nat effect. See [[exp3-iter0-centre-weighted-round-edges]], [[exp3-concept-family-probes]].
