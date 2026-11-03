---
name: trial-index-forbidden
description: models may not bind trial_index (admission rejects); stimulus-only proxies for session drift
metadata:
  type: project
---
Admission rejects any model binding trial_index: it must be evaluable on bare (number_set, target) stimulus pools.
In exp2, trial_index correlates -0.21 with set size and ~0 with target; yes rate drops with target magnitude (0.39 for 1-60 vs 0.32 for 61-100).
**Why:** session-drift candidates (see [[session-drift-finding]]) were all rejected for this.
**How to apply:** address drift critiques only through stimulus features, e.g. the small_number_expectation_concepts target-magnitude slope (smoke loglik -2212.2).
