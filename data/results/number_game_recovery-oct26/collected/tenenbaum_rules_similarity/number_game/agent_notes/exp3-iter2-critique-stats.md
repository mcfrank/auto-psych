---
name: exp3-iter2-critique-stats
description: exp3 iter_2 critique of compressed_rule_extension: 8 stats on parity interp/extrap, unanimity, set size, drift, singleton far, non-parity rules, min-first
metadata:
  type: project
---
2026-10-22, iter_2 critique (incumbent compressed_rule_extension) submitted: opp_parity_just_outside_yes (obs 0.444), parity_gap_inside_minus_outside (-0.052), cell_unanimity_rate (0.370; cells have about 5 responses each), set_size_near_outside_gap, trial_index_yes_corr, singleton_far_yes, nonparity_rule_far_gap (vectorised 42-rule membership via M[:, Xi].all(axis=2)), below_asym_min_first. All run in about 0.03 s.
**Why:** so later rounds don't re-propose these. **How to apply:** read iter_2/critique/ppc_results.json first. Related: [[exp3-iter1-critique-stats]], [[compressed-rule-extension]]
