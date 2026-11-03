---
name: exp3-iter2-novelty-gate-map-vs-posterior
description: Novelty gate shrinkage — MAP-based pool RMSE overstates posterior-mean RMSE by ~3x for weak nested params; how to probe it (exp3 iter_2 cand_5 repair)
metadata:
  type: feedback
---
half_decade_round_edge_ranges had MAP pool RMSE 0.0048 vs round_edge_ranges_digit_sums, but the posterior-mean RMSE was only 0.00153, so it was rejected (gate 0.002). A weakly identified extra bias shrinks toward 0 under the posterior.
For the repair I submitted round_edge_digit_sums_primes_family (primes own family on the round-edge base). It has MAP RMSE 0.0035 vs its parent, and MAP LL is -3512.87 against the parent's -3514.03. The same split on the graded base also has MAP RMSE 0.00345, and that model passed the gate.
**Why:** a second rejection in a slot is final.
**How to apply:** use iter_2/candidate_5_repair_1/scratch/nov.py (MAP fit on the responses, predictions on novelty_pool.json, RMSE). Aim for a MAP RMSE of at least 0.0035 for structural changes, and more for bias parameters. See [[exp3-iter2-menu-refinement-probes]].
