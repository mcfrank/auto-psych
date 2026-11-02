---
name: times-table-familiarity
description: exp2 iter_2 c3 repair — per-table Dirichlet familiarity weights over multiples rules on beta-share incumbent; smoke ll −2270.9, ~33 s smoke
metadata:
  type: project
---
exp2 iter_2 candidate_3_repair_1 (2026-10-30), times_table_familiarity_concepts. The first attempt (size-exponent within-kind prior) was rejected as a near-duplicate of specificity_biased_rule_prior (RMSE 0.00015).
Repair: the multiples family gets a free Dirichlet(3) over its 9 tables, not tied to size. Smoke check OK, ll −2270.9; adds 8 params.
**Why:** rule-size-exponent priors collapse to the incumbent ([[size-weighted-within-kind-rules]], [[specificity-biased-rule-prior]]), so only non-size-based reweighting has a chance of passing novelty.
**How to apply:** check this model's admission outcome before retrying per-rule weights.
