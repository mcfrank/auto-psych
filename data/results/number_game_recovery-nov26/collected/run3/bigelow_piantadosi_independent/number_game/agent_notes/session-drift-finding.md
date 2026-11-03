---
name: session-drift-finding
description: experiment2 trial_index drift on the incumbent: logit yes-bias drift +5.6 log-post; beta/sigma/lambda drifts null
metadata:
  type: project
---
Tried 2026-10-30 (experiment2 iter_0 candidate_1, `session_drifting_yes_bias`), on costly_squares_final_digit_concepts, MAP:
- Logit shift -d*(t/449-0.5) on the concept belief: d≈0.39, log-post -2226.0 vs -2231.6 for the base (one param).
- An extra intercept (c0≈0.28) adds nothing. A power transform p^exp(k) did worse (-2232). Default-no lapse rising over the session and a drifting lapse guess rate both gave 0.
- Drift in commitment beta, interval sigma or rule share lambda: 0 gain. Sets cluster in session time: only 361 (set, 5-phase) cells.
- Base MAP here: beta≈0.77 (NOT near all-or-none on exp2), lapse→0.
- Trial orders: each trial_index holds 2 distinct stimuli across 10 rows, so there are 2 stimulus orders.

**Why:** the critique's trial_index_rating_correlation (q=0.048) is mostly explained by stimulus order, and the leftover drift is small.
**How to apply:** don't expect big gains from session-drift mechanisms. Related: [[experiment2-data-quirks]].
