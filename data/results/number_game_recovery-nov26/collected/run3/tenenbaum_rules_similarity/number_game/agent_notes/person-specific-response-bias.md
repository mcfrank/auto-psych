---
name: person-specific-response-bias
description: exp3 iter_0 c0: per-participant logit yes-bias on beta_share learner; individual_rule_vs_range lost 323 because N_MAX_PEOPLE=12 < 15
metadata:
  type: project
---
Exp3 (2026-10-31) yes-rates by participant come in 3 blocks of 5: 0.40, 0.58, 0.49 (very homogeneous within block). iter_0 candidate_0 = `person_specific_response_bias`: beta_share_width_exception_concepts + hierarchical per-person bias added to logit(membership), N_MAX_PEOPLE=40, unknown participant → population mean.
The live `individual_rule_vs_range_tendency` had N_MAX_PEOPLE=12, so on exp3's 15 participants it silently fell back to "unknown" for every row — its −323 nats is a capacity bug, not evidence against individual differences.
**Why:** any per-participant model must size its capacity for the current experiment's participant count.
**How to apply:** check participant count (trial_index==0 resets) before reusing exp2 per-person code. See [[experiment3-iter0-critique-stats]].
