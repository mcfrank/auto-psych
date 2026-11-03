---
name: exp3-iter3-round-edge-graded-final-digit
description: Exp3 iter_3 cand_5 (free refinement slot) — round edges on fast graded_final_digit base, MAP -3513.42 (+0.6 vs parent), posterior novelty RMSE 0.0027, 16 s/600-iter chain; submitted round_edge_graded_final_digit_sums
metadata:
  type: project
---
2026-11-01, exp3 iter_3 candidate_5. I took the fast width-grouped graded_final_digit_digit_sum_concepts and grouped the ranges by (width, number of round edges), giving 300 columns. It adds round_bonus ~ N(0,1).
- MAP LL: -3513.42. Parent -3514.01, round_edge_ranges_digit_sums -3514.03, incumbent -3514.87.
- Posterior (300/300, 1 chain, 16 s): round_bonus -0.57 ± 0.47, no shrinkage. Posterior-mean pool RMSE vs the parent is 0.0027, above the 0.002 gate. MAP RMSE was 0.0025.
- MAP RMSE: incumbent 0.0077, round_edge_ranges_digit_sums 0.011.
- Probe script for posterior novelty: iter_3/candidate_5/scratch/post.py (uses sample_posterior_predictive on the pool).
**How to apply:** prefer post.py over MAP nov.py for a gate check on bias parameters. See [[exp3-iter2-novelty-gate-map-vs-posterior]].
