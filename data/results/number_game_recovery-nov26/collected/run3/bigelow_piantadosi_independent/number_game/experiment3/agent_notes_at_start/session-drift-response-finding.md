---
name: session-drift-response-finding
description: "experiment2 yes-rate drift over session: default-no fatigue +1.2 log-post MAP, logit criterion shift worse"
metadata:
  node_type: memory
  type: project
  originSessionId: 814163db-e185-427c-8227-e4e428de7ddd
  modified: 2026-10-30T22:18:21.620Z
---

Tried 2026-10-30 (experiment2 iter_0 candidate_3, `session_fatigue_default_no_concepts`) on the costly-squares incumbent:
- Rising default-"no" rate p_no = sigmoid(a + b*(t/450-0.5)): MAP a=-3.6, b=4.1 (≈3% early to ≈20% late), log-post -2230.5 vs -2231.6 for the incumbent. Small gain.
- Logit criterion shift by session position: -2234.0 (worse). A logit shift barely moves confident p near 0/1.
- Lapse stays at 0.

**Why:** most of the yes-rate drift is already explained by stimulus order. The residual (critique z=-2.5) is small.
**How to apply:** don't expect big ELPD gains from session-drift mechanisms. Related: [[experiment2-data-quirks]].
