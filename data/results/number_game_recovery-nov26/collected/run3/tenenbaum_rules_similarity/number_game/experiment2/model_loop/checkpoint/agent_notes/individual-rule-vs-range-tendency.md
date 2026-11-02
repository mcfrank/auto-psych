---
name: individual-rule-vs-range-tendency
description: exp2 iter_3 c5: per-participant rule-vs-range prior odds on beta-share incumbent — sd≈0.08, LOO −0.5 (null); per-person range width too slow
metadata:
  type: project
---
exp2 iter_3/candidate_5 (2026-10-30), submitted individual_rule_vs_range_tendency: participant = cumsum(trial_index==0), hierarchical non-centred logit rule share; rows without trial_index get the average over the fitted people.
Quick 2x1000 NUTS: LOO −2274.0 vs incumbent −2273.6 (2x300); rule_logit_sd 0.08±0.07, so people do not differ. Fit 105 s.
Incumbent calibration is near perfect: per-participant yes rates match (the 0.40 vs 0.58 groups come from the stimuli), and so do p-bins and session position.
Per-person mean_share (a people×cells×sizes tensor) took >10 min for 600 draws, so it is too slow.
**Why:** individual differences look absent. **How to apply:** don't retry person-level parameters; see [[experiment2-data-structure]].
