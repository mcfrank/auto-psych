---
name: exp3-iter3-round-edge-equal-prior
description: Exp3 iter_3 cand_3 — round-edge bonus on equal_prior_familiar_rules_digit_sums +1.15 nats MAP (bonus -0.49), MAP RMSE 0.0033 vs the incumbent; group-by-(width, round edges) precompute gives 46 ms/grad; submitted as equal_prior_rules_round_edge_ranges
metadata:
  type: project
---
2026-11-01, exp3 iter_3 candidate_3 (incumbent refinement slot).
- On the incumbent (MAP LL -3514.87): dropping the centre weight gives -3514.51, and adding the round-edge bonus on top gives -3513.72 (round_bonus -0.49).
- MAP pool RMSE: 0.0033 vs the incumbent, 0.011 vs round_edge_ranges_digit_sums, 0.0067 vs graded_final_digit. Posterior shrinkage may still push it under the 0.002 gate; see [[exp3-iter2-novelty-gate-map-vs-posterior]].
- Speed: the naive (sets x 5050 ranges) version took 181 s for 200 smoke-test iterations. The earlier graded+round model timed out at admission.
  - Fix: group ranges by 3*(W-1)+n_round_edges (300 groups) and precompute, per set, the group log total and the in-target share (154 sets x 300 x 100). Gradient cost is 46 ms.
  - Gathering per row (6750 x 300) is slower (209 ms). Compute per set, then gather.
- Scripts are in iter_3/candidate_3/scratch/p (tg.py times the gradient). Use iter_2/candidate_5_repair_1/scratch/nov.py for MAP and RMSE.
