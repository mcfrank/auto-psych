---
name: exp3-example-rejection-critique
description: exp3 iter_0 critique: people reject some shown examples (outliers/last-shown), e.g. 100 after 31,19,49,100 gets 7% yes; 8 stats submitted
metadata:
  type: project
---
Exp3 (2026-10-22) has 15 sessions x 450 and 194 sets. Raw data: shown examples get only 0.82 yes in sets of 2 or more. Outlier examples that break the others' rule get much less: 100 in 31,19,49,100 (0.07), 72 in 18,8,72,98 (0.2), 12 in 4,16,12 (0.2), 27 in 92,82,27,42 (0.4). The last-shown example gets 0.14 less yes than earlier examples. iter_0 critique stats: example_target_yes_rate, last_vs_earlier_example_yes, rule_breaking_example_yes_gap, session_yes_rate_sd, above/below edge asymmetry for max-last and min-last sets, singleton near generalization, interior proximity corr.
**Why:** The incumbent cannot reject an example beyond eps = 0.1 noise.
**How to apply:** Check iter_0/critique/ppc_results.json. If the example-rejection stats are significant, try an outlier or "example-is-a-distractor" mechanism. Related: [[last-example-top-marking]], [[session-heterogeneity]]
