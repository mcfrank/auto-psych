---
name: experiment2-iter1-critique-stats
description: exp2 iter_1 critique of beta_share_width_exception_concepts — the 8 PPC stats chosen and observed values, so later rounds avoid repeats
metadata:
  type: project
---
iter_1 critique (2026-10-30) stats, with observed values: wide_spread_inrange_yes_rate 0.53 (>=3 ex, span>=60); within_cell_disagreement 0.34; set_size_inrange_contrast (2-ex minus 4-ex inside range) -0.11; sparse_rule_numeric_neighbor_yes 0.00 (50 trials, all no); sparse_rule_member_yes 0.61; last_digit_vs_parity_contrast 0.76; below_minus_above_outside_yes -0.06; interior_proximity_gradient -0.03.
Note: there is no participant_id column; the data has unquoted single-example sets (e.g. `8,1,129,0`), which pandas reads fine.
**Why:** iter_0's 8 stats were all null ([[prior-leaking-judgment]]); this records what iter_1 probed.
**How to apply:** check iter_1/critique/ppc_results.json for which came out significant before choosing new stats; see [[experiment2-data-structure]].
