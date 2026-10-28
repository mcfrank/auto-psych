---
name: multi-example-top-marking
description: exp3 iter_0 c3 repair: near-top softening was novelty-rejected (RMSE 0.00065); replaced with no top-marking credit for singletons; raw singleton data symmetric
metadata:
  type: project
---
2026-10-22, exp3 iter_0 candidate_3_repair_1 (`multi_example_sequence_top_marking`). [[near-top-closing-example]] was rejected as a near-duplicate of the incumbent: its rho went to about 0, so its p_yes RMSE was 0.00065. In the repair, the one change to the incumbent is top_share = 1 only when len(set) >= 2 and the last example is the max. Raw exp3 data: after a singleton, yes within 1-5 below = 0.36 and above = 0.38, so it is symmetric. Max-last sets with n = 3 or 4 have dn 0.92/0.76 vs up 0.51/0.55. For max-last pairs, dn is 0.08 (n = 25) and up is 0.48.
Check OK, smoke LL -3095.0. The gather version took 295 s per smoke run, so it was ported to batched_dot blocks of 8.
**Why:** The incumbent claims a strong downward pull after a single example, which the raw data do not show. The change also moves 108 singleton stimuli in the novelty pool.
**How to apply:** Check the LOO outcome. If it wins, the singleton marking claim is dead. Related: [[last-example-top-marking]]
