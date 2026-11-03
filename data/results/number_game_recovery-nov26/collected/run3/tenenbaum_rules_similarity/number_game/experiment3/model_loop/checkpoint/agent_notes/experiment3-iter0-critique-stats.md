---
name: experiment3-iter0-critique-stats
description: exp3 iter_0 critique of uncertainty_shifted_criterion — data structure + 8 PPC stats with observed values
metadata:
  type: project
---
Exp3 data (2026-10-31): 15 participants × 450 trials (split on trial_index==0), 6750 rows, ~190 sets, adaptive-looking design (5–270 trials per set). Participant yes rates fall in three blocks of 5 (≈0.40, 0.58, 0.49). Very noisy: shown-example targets only 66% yes, and many sets sit near 0.5. No MCMC cache existed at critique time.
iter_0 stats, observed: participant_yes_rate_sd 0.071; participant_parity_discrimination_sd 0.16; parity_span_interaction -0.065; lag1_response_repeat 0.539; single_example_distance_gradient (≤10 vs ≥40) 0.02; setsize_outspan_sharpening (n4−n2) 0.012; shown_example_target_yes 0.658; multiple_rule_vs_span_preference (k 3..12) 0.27.
**Why:** so later rounds don't probe the same things again; exp2 versions of lag1/participant SD/target-in-set were null ([[experiment2-iter4-critique-stats]]).
**How to apply:** read iter_0/critique/ppc_results.json for hits before choosing iter_1 stats.
