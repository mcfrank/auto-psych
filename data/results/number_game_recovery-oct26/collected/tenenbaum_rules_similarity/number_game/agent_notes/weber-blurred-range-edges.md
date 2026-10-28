---
name: weber-blurred-range-edges
description: exp3 iter_4 c3: Weber-blurred plain-range membership on incumbent; data reject blur (weber→0.003 at MAP, LL drops 9 nats at 0.02); MAP RMSE 0.004
metadata:
  type: project
---
2026-10-22, exp3 iter_4 candidate_3 (refinement of sequence_opens_bottom_ends_top), named `weber_blurred_sequence_range_edges`. Perceived target ~ N(t, 0.25+weber*t), truncated to 1-100, applied only to the plain-range part of p_yes. A per-set (S,4G,100) range-membership table built in prepare_observed, batched_dot with the posterior, then the kernel read at cell targets.
Profile LL at the MAP of the other parameters: weber 0.001 → -3089.8, 0.01 → -3093.0, 0.02 → -3098.7, 0.08 → -3138. People's range edges are sharp, so magnitude blur is ruled out. MAP RMSE vs the incumbent is 0.004.
Final check OK, smoke LL -3093.7; 360-379 s per smoke run on a host with load 18 (speed risk). Speed: the first version (HalfNormal prior, full 100x100 kernel dot) took 311-319 s per smoke run, too slow. The fix was a LogNormal(log .01, 1) prior plus reading the kernel only at cell targets, which brought the gradient to 0.073 s vs the incumbent's 0.070 s.
Per-cell residual scan of the incumbent: the top cells are all n=5 noise (e.g. 3,7→67 yes), with no systematic gap.
**Why:** so no one re-proposes blurred or Weber range edges. **How to apply:** check outcome; don't blur edges. Related: [[sequence-opens-bottom-ends-top]], [[pymc-model-speed-patterns]]
