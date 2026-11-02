---
name: mixed-sampling-kind-uncertainty
description: exp3 iter_1 c4 repair: mixed strong/weak sampling grafted onto kind-entropy incumbent; smoke OK ll -3653.4 (~5 min)
metadata:
  type: project
---
exp3 iter_1/candidate_4_repair_1 (2026-10-31), mixed_sampling_kind_uncertainty: mixed_sampling_uncertainty_criterion source with its concept-entropy line swapped for the incumbent's kind entropy (built by text-patching). First attempt (binary rule-vs-range entropy) was rejected with RMSE 0.00000 vs rule_vs_range_doubt_criterion — the same idea had already been submitted as c3 this round.
Smoke OK, ll −3653.4, ~90 s sampling. 1×300 fit: theta 0.95, shift +0.23; pool RMSE 0.0024 vs incumbent (thin margin over 0.002), 0.0084 vs mixed_sampling_uncertainty_criterion. Novelty script: /tmp/nov41/nov.py responses pool out.npy model.py.
**How to apply:** before a repair, check same-round live models (existing_hypotheses "no comparison row") for the same idea. See [[rule-vs-range-doubt-criterion]], [[mixed-sampling-uncertainty-criterion]].
