---
name: uncertain-sampling-compressed-rules
description: exp3 iter_2 c3: range size power null (MAP 0.98); strong/weak mixture on incumbent MAP strong 0.83, LL -1.2 vs inc, RMSE 0.004
metadata:
  type: project
---
2026-10-22, exp3 iter_2 candidate_3 (refinement of the incumbent compressed_rule_extension). A first try gave ranges their own size power. At MAP the power was 0.98, LL +0.03 nats, so it was a duplicate of the incumbent; kept as scratch/v1_range_power.py.
Submitted `uncertain_sampling_compressed_rules`: the incumbent with a strong/weak sampling mixture. Every hypothesis column is duplicated per mode; range and conjunction evidence are precomputed per mode, because range evidence does not depend on any parameter. At MAP strong_prob was 0.83 and obsLL -3092.35, against -3091.17 for the incumbent at its own MAP. MAP p_yes RMSE 0.0040, a novelty risk. The gradient takes 0.089 s.
Pitfall: `pkill -f <pattern>` killed my own bash shell (exit 144), because the pattern matched the shell's own command line.
**Why:** The set-size critique is not closed by range compression or by weak sampling; the data want true-size ranges. **How to apply:** Check this candidate's admission and novelty outcome. Related: [[uncertain-sampling-sequence-top]], [[compressed-rule-extension]], [[novelty-gate-tiny-tweaks]]
