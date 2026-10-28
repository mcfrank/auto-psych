---
name: iter4-exploratory-screens
description: iter_4 cand_2 MAP screens on uneven_rule_familiarity_ends — Gaussian magnitude clusters lose 115 nats; graded range edges null; best-explanation power submitted
metadata:
  type: project
---
2026-10-21/22, experiment3 iter_4 candidate_2 (exploratory slot). Incumbent uneven_rule_familiarity_ends MAP loglik -3186.35.
- Replacing intervals with Gaussian magnitude clusters (centre 1..100 x log-spread grid, graded typicality membership): -3301 to -3354 for every width variant. Crisp intervals clearly win. Do not retry.
- Graded range edges (target d steps past an edge belongs by r^d, via right/left edge-mass batched_dot + kernel): r=0.22, loglik -3186.5 (null), RMSE vs incumbent 0.00196 (near-duplicate), and gradients 3.4x slower. Dead.
- Best-explanation / soft-max-over-concepts judgment: belief = (sum_{h∋t} w^β / sum w^β)^(1/β). After reparameterising, this is the incumbent posterior with familiarity scaled by β and belief^(1/β). MAP β≈1.3, loglik -3187.7, RMSE 0.0044, same speed as the incumbent. Fixing β=5 costs 14 nats and β=0.5 costs 8. Submitted as best_explanation_target_judgment.
- Without β on the familiarity the RMSE was only 0.0020 (a novelty risk). Sampling raw β·w trades off strongly with the evidence weights, so sample effective weights instead.
**How to apply:** scripts are in iter_4/candidate_2/scratch (mapfit.py, rmse.py, edge.py). See [[map-screening-refinements]] and [[incumbent-saturation-screens]].
