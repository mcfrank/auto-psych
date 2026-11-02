---
name: two-cluster-rule-family-doubt
description: exp3 iter_3 c5 repair: two_cluster + incumbent rule-family doubt criterion; pool RMSE 0.0036 vs two_cluster_kind_doubt, 0.0044 vs incumbent; smoke OK
metadata:
  type: project
---
exp3 iter_3/candidate_5_repair_1 (2026-10-31), two_cluster_rule_family_doubt: two_cluster_range_concepts with the criterion taken from rule_family_doubt_criterion, P(rule)·H(family|rule). The first attempt (satisficing + rule-family doubt, [[satisficing-search-rule-kind-doubt]]) was rejected as a dup of the incumbent (RMSE 0.0012), so the satisficing effort collapses once the criterion is present.
Results from a 2×300 fit: shift 0.18, cluster share 0.040, plug-in ll −3648.4 (incumbent −3648.1, two_cluster_kind_doubt −3648.9).
Pool RMSE: 0.0036 vs two_cluster_kind_doubt, 0.0044 vs the incumbent, 0.0056 vs two_cluster. Smoke check OK at −3653.3.
Script: /tmp/c5r1/nov.py <model.py> <out.npy> fits a model and saves pool posterior-mean p_yes.
**How to apply:** grafts onto the incumbent criterion pass novelty only when the other component changes the concept space (two clusters), not when it reweights existing concepts. See [[two-cluster-kind-doubt-criterion]].
