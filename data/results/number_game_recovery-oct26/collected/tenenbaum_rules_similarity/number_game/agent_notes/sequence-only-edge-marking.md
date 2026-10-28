---
name: sequence-only-edge-marking
description: exp3 iter_4 c3 repair: Weber blur novelty-rejected (posterior RMSE 0.0003 though MAP RMSE 0.004); replaced by incumbent + no top credit for singletons
metadata:
  type: project
---
2026-10-22, exp3 iter_4 candidate_3_repair_1 (`sequence_only_edge_marking`). [[weber-blurred-range-edges]] was rejected: posterior p_yes RMSE vs the incumbent was 0.00029 even though the MAP proxy gave 0.004, so the MAP proxy over-states novelty when a parameter collapses to its null. The repair is sequence_opens_bottom_ends_top with top_share requiring len(set) >= 2, the change [[multi-example-top-marking]] made on the older model. It is structural (it cannot collapse), and each pairwise difference matches a pair that was already admitted.
**Why:** Continuous tweaks that the data push to zero always fail novelty. **How to apply:** Prefer structural (parameter-free) changes for repair slots. Check OK: smoke LL -3093.7, 365 s per smoke run. Check outcome. Related: [[novelty-gate-tiny-tweaks]], [[sequence-opens-bottom-ends-top]]
