---
name: session-drift-default-finding
description: lapse default answer drifting with trial_index (exp2) — smoke-fit loglik -2211.6 vs incumbent ELPD -2347
metadata:
  type: project
---
2026-10-30, experiment2 iter_0 candidate_4 (`session_drifting_default_concepts`): incumbent costly_squares model with the lapse's 0.5 coin flip replaced by sigmoid(a + b * trial_index/450). It needs prepare_observed, because compute_features doesn't get trial_index. A 100-draw smoke fit gave log-lik -2211.6, against the incumbent's ELPD -2347 (not a like-for-like comparison). Each smoke run takes about 2 min.

**Why:** the critique flagged the yes-rate drift over the session (q=0.048); see [[experiment2-data-quirks]].
**How to apply:** check its final ELPD; if it wins, later refinements should keep the drifting default.
