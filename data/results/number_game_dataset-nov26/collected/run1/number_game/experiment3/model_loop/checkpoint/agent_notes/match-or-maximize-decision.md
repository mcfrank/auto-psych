---
name: match-or-maximize-decision
description: exp3 iter3 c2 — match/maximize mixture decision rule on parity_preserving_decisive_route, +1.6 nats MAP; dropping coincidence slope costs 73
metadata:
  type: project
---
Decision-rule candidate (2026-11-02): p = (1-p_max)*sigmoid(L) + p_max*sigmoid(8L), with L the incumbent's coincidence-scaled belief logit. MAP: p_max ≈ 0.13, loglik -4064.4 vs incumbent -4066.0. Prediction RMSE vs the incumbent on training stimuli is 0.017, so it should pass the novelty check. Smoke fit: -4072.0. Removing the coincidence slope, so that match/maximize alone does the work, gives MAP -4139.4, 73 nats worse.
**Why:** the brief asked for a decision-level mechanism; the critique reported polarization after single-example and incoherent sets.
**How to apply:** as with [[biased-lapse-exemplar]], decision-rule tweaks gain about 1-2 nats here. The coincidence-scaled slope is essential.
