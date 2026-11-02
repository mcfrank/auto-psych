---
name: exp3-decision-rule-probes
description: Exp3 iter_0 decision-stage probes on without_replacement_size_principle — lapse+guess bias pins at 0; logit gain+criterion gives +1.7 nats (submitted)
metadata:
  type: project
---
experiment3 iter_0 candidate_0 (2026-11-01), base without_replacement_size_principle, MAP LL -3523.11 on exp3 data (6750 rows; interval_scale 86):
- Lapse with a learned guess bias, (1-e)*belief + e*g: e = 9e-8 at MAP, no gain at all. Dead (it would be a near-duplicate).
- Logit gain + criterion, sigmoid(gain*(logit belief - c)): gain 0.956, c -0.07, LL -3521.38 (+1.7), data RMSE vs base 0.010. Submitted as graded_belief_criterion_decision; smoke check OK in about 2 min.
Probe scripts /tmp/map_c0.py and /tmp/rmse_c0.py (they may not persist). Run check_candidate from repo/, not from the _runs dir.
**Why:** belief is almost perfectly calibrated, and only a slight hedging of answers helps.
**How to apply:** don't spend a slot on lapse or bias decision rules again. See [[exp2-memory-and-prior-anchoring-probes]].
