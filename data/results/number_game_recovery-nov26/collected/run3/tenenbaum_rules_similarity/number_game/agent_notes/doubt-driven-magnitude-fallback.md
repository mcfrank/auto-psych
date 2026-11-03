---
name: doubt-driven-magnitude-fallback
description: exp3 iter_3 c3 repair: rule-kind doubt mixes p_in toward range-only posterior instead of a criterion shift; fallback≈0.04, RMSE 0.006 vs incumbent, 0.0027 vs beta_share
metadata:
  type: project
---
exp3 iter_3/candidate_3_repair_1 (2026-10-31), doubt_driven_magnitude_fallback: [[rule-family-doubt-criterion]]'s doubt (P(rule)·H(family|rule)) no longer shifts the logit. Instead p_in = (1-w)·full + w·range-only posterior, with w = fallback·doubt and fallback ~ Beta(1,3). Smoke OK (45 s sampling, ll −3653.5).
Quick 2×300 fits: fallback 0.044±0.037; the incumbent's uncertainty_shift is 0.18±0.12. Pool RMSE: 0.0063 vs incumbent, 0.0027 vs beta_share (thin).
The first attempt ([[exception-as-variant-doubt]]) was rejected: RMSE 0.0004 vs the incumbent.
Script: /tmp/nov33/nov.py <model.py> <out.npy> saves pool p_yes means; one model takes about 4 min.
**How to apply:** re-pooling the doubt measure does not change predictions. Changing what the doubt *does* (replacing the readout) does.
