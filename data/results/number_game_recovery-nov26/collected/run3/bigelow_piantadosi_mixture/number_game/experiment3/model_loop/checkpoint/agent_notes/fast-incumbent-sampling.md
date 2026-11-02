---
name: fast-incumbent-sampling
description: exp2 iter_0 (2026-11-01): every family_pooled_uncertain_reliability refinement timed out at 0.99; the fixes were a fixed spread, sparse dots, and finite k>n cells
metadata:
  type: feedback
---
All six exp2 iter_0 candidates (refinements of family_pooled_uncertain_reliability) were rejected as too slow. Admission runs 3000 tune + 4000 draws at target_accept 0.99 with warm_start_nuts, under a 30-minute limit. The incumbent's graph takes about 11 ms/grad and has a tau*rule_dev funnel, which took more than 1 s per NUTS iteration.
The repair in candidate_3_repair_1 (strong_weak_sampling_mixture):
- Centred rule_logw ~ N(family_level, 1.75), with the spread fixed instead of an inferred tau. This removes the funnel: 31 steps per iteration, tree depth 5, no divergences.
- Family levels as a ZeroSumNormal, since the softmax ignores a common shift.
- The (n,k,s) table flattened. Interval evidence is a dense dot product for sets; cells use pytensor.sparse CSR structured_dot, with the CSR parts passed as pm.Data. The rule_post[cell_set] gather became a sparse one-hot dot (the AdvancedIncSubtensor gradient was the top cost). Result: 2.3 ms/grad.
- Gotcha: table cells with k>n set to -1e4 in every j give NaN gradients once a dot product touches them. Keep them finite and mask only j>k.
**How to apply:** start any refinement of this incumbent from that candidate.py, not from the model file. See [[warm-start-saddle-priors]] and [[exp2-incumbent-refinement-scan]].
