---
name: mixed-sampling-rule-family-doubt
description: exp3 iter_4 c4 repair: mixed strong/weak sampling grafted onto rule_family_doubt incumbent; MAP theta≈0.93, pool RMSE ≥0.003 vs all live models; smoke OK ll −3653.5
metadata:
  type: project
---
exp3 iter_4/candidate_4_repair_1 (2026-10-31), mixed_sampling_rule_family_doubt. Built from mixed_sampling_kind_uncertainty.py by swapping in the incumbent's P(rule)·H(family|rule) doubt. The first attempt was the centroid-pull graft. It was rejected with RMSE 0.00000 against the same-round prototype_pull_rule_family_doubt, because candidates 3, 4 and 5 all submitted the same idea.
MAP novelty-pool RMSE: incumbent 0.0031, kind_uncertainty 0.0038, mixed_sampling_kind_uncertainty 0.0039, first_impression 0.0032, others ≥0.0044. MAP logp is −3673.8, vs −3671.9 for the incumbent, so expect no gain.
Tool: /tmp/c4r1/mapfit.py cand.py others... runs an L-BFGS MAP fit for each model and prints the pool RMSE of the first model against each of the rest (~1 min per model).
**How to apply:** check same-round siblings before a graft. See [[mixed-sampling-kind-uncertainty]] and [[prototype-pull-rule-family-doubt]].
