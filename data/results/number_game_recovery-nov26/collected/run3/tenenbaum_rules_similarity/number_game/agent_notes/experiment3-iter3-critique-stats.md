---
name: experiment3-iter3-critique-stats
description: exp3 iter_3 critique of rule_family_doubt_criterion — 8 new PPC stats + observed values (iters 0-2 got 0/24 hits)
metadata:
  type: project
---
iter_3 critique (2026-10-31), incumbent rule_family_doubt_criterion. Iters 0-2: 0/24 significant (closest: multiple_rule_vs_span 0.08, decade_shared 0.08, target_100 0.10).
New stats, observed: digit_reversal_target_contrast -0.022 (295 reversal rows, raw yes 0.31); double_half_target_contrast +0.028 (non-pow2 sets); below_vs_above_span_asymmetry -0.029; recency_endpoint_proximity -0.037; within_set_target_profile_sd 0.245; target_one_yes 0.39; interior_yes_setsize_contrast (n4-n2, span>=20) 0.059; single_example_unshown_yes 0.442.
Big exp3 sets look designed around odd concepts (31,3,1,15 = 2^k-1; 83,38,58,68 = digit 8/reversals) but only ~5 answers per (set,target) cell, so per-set stats lack power.
No null simulation run; /tmp/crit3 refit scripts are gone.
**Why:** avoid re-probing; see [[experiment3-iter2-critique-stats]].
**How to apply:** read iter_3/critique/ppc_results.json before choosing iter_4 stats.
