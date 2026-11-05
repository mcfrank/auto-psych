---
name: feature-distribution-prototype
description: iter3 cand2 exploratory: naive-Bayes category estimation (normal magnitude + smoothed digit/parity freqs) smoke log-lik -1598.5
metadata:
  type: project
---
iter_3 candidate_2 (exploratory slot, 2026-11-02): `feature_distribution_prototype` — target's log LR under estimated category (normal on magnitude w/ var+tau², digit freq smoothed toward parity freq smoothed toward chance) vs uniform; 5 params. Smoke log-lik -1598.5 (incumbent ELPD -1584.9). Aimed at set_level_yes_rate_sd critique. Fast (~2s smoke).
**Why:** non-exemplar, non-rule process family competitive with exemplars.
**How to apply:** if it survives, refine (e.g. t-predictive magnitude) rather than re-propose. See [[iter3-critique-stats]].
