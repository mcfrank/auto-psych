---
name: two-cluster-kind-doubt-criterion
description: exp3 iter_1 c5 repair: two_cluster + incumbent kind-entropy shift; shift 0.24±0.22, cluster share 0.035, ll −3653.7, pool RMSE ≥0.0035 vs checked models
metadata:
  type: project
---
exp3 iter_1/candidate_5_repair_1 (2026-10-31), two_cluster_kind_doubt_criterion: two_cluster_range_concepts plus the incumbent's normalised kind-entropy logit shift. The kinds are the rule families, single range and two clusters (log(N_FAMILIES+2) normaliser).
2×300 fit: ll −3653.7 (two_cluster alone −3654.2). Pool RMSE: 0.0035 vs incumbent, 0.0053 vs two_cluster, 0.006 vs beta_share. Smoke check OK in 33 s.
mixed_sampling_uncertainty, range_misperception_uncertainty and rule_vs_range_doubt were not checked for novelty.
**How to apply:** this is the fallback after [[person-breadth-and-momentary-odds-nulls]]. It is likely a statistical tie. See [[two-cluster-range-concepts]].
