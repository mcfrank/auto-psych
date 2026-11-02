---
name: stimulus-only-default-lapse
description: trial_index models are rejected at admission; stationary fitted default-yes lapse on incumbent smoke loglik -2211.6
metadata:
  type: project
---
2026-10-30, exp2 iter_0 candidate_3_repair_1 (`default_no_lapse_costly_squares`): admission rejects any model that binds trial_index (the novelty pool and held-out evaluation hold stimuli only), so the session-drift critique can't be targeted directly. The repair swaps the incumbent's 0.5 lapse coin for a fitted Beta(2,2) default_yes. Smoke log-lik was -2211.6, the same as the drifting version in [[session-drift-default-finding]]: the drift itself adds ~nothing. Each smoke run takes ~3 min.

**Why:** see [[session-drift-null-finding]].
**How to apply:** don't propose trial_index mechanisms; compute everything from number_set/target.
