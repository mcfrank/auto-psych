---
name: representative-coverage-family-doubt
description: exp3 iter_4 c5 repair: coverage^kappa graft on rule_family_doubt incumbent; MAP pool RMSE ≥0.005 vs all nearest live models, smoke OK ll −3652.7
metadata:
  type: project
---
exp3 iter_4/candidate_5_repair_1 (2026-10-31), representative_coverage_family_doubt: the incumbent rule_family_doubt_criterion plus the coverage**kappa evidence term from [[representative-coverage-sampling]]. The first try, the prototype-pull graft ([[prototype-pull-rule-family-doubt]]), was rejected as a dup (RMSE 0.00026) of the same-round c4.
MAP: kappa −0.16, shift 0.18. MAP log-posterior −3672.0 vs the incumbent's −3671.9, so no fit gain. MAP pool RMSE vs: incumbent 0.0050, rep_coverage 0.0052, prototype_pull_rfd 0.0056, kind_uncertainty 0.0059, two_cluster_rfd 0.0061, beta_share 0.0076. Smoke 42 s.
Tool: /tmp/c5r1/mapfit.py cand.py others... does an L-BFGS MAP fit and prints the pool RMSE vs the first model (takes ~1 min/model).
**How to apply:** before grafting, check the same-round live models for dups, not just older ones.
