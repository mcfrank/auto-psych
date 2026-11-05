---
name: single-example-leniency
description: extra "yes" bias on single-example sets gave +27 MAP loglik on the exp2 incumbent; recency drops 1.4 -> 0.3
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_1/candidate_4, single_example_leniency_overlap). The incumbent seed_overshadowing_last_digit_recency has MAP loglik -3152.7. Adding an intercept shift for 1-example sets gives -3125.5 (b about 0.83). A slope multiplier for single sets gives -3126.6. Shrinking the overlap toward 1 gives -3127.3. A linear term in the number of examples gives only -3143.9. The smoke fit gave -3132.2. With the shift in place, the last-digit recency weight falls from 1.4 to 0.3, so recency was partly standing in for the single-set baseline.
**Why:** this addresses the critique's single_example_yes_gap (q=0.008). The early_session_yes_gap still needs trial_index, which only prepare_observed can see.
**How to apply:** treat single-example sets as their own regime. Check whether recency is still needed. See [[last-example-digit-recency]].
