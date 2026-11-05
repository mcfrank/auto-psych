---
name: exp3-iter4-critique-stats
description: exp3 iter4 critique of coincidence_rule_violation_route — 8 PPC stats: polarization carry-overs, trial drift, digit-violation proximity, violator arithmetic rescue, parity-vs-digit violation gap, partial digit majority, violation set-size
metadata:
  type: project
---
Wrote 8 stats in experiment3/model_loop/iter_4/critique/test_stats (2026-11-02). Observed values:
- Carried over unchanged: single_example_cell_polarization 0.248, incoherent_multi_cell_polarization 0.187 (both significant in iter3, and the incumbent doesn't address them), within_item_trial_drift 0.025 (significant in iter0, and the model is still stationary).
- New: digit_rule_violation_proximity_contrast 0.001 (people reject digit-rule breakers flatly: 0.136 near vs 0.126 far; parity-rule breakers are graded, 0.29 vs 0.13). coherent_violator_arithmetic_contrast 0.095. parity_vs_digit_rule_violation_gap 0.049 (the incumbent's c_viol is shared across rule types). partial_digit_majority_contrast 0.32 (13 sets where all examples but one share a digit). violation_rate_set_size_contrast -0.056 (n4 vs n2).
Each runs in under 0.04 s.
**Why:** follows [[exp3-iter3-critique-stats]]; incumbent is [[coincidence-rule-violation-route]].
**How to apply:** check iter_4 ppc_results.json before proposing these ideas again.
