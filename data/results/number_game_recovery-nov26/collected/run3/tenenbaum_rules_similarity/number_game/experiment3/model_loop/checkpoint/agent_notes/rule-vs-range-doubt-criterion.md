---
name: rule-vs-range-doubt-criterion
description: exp3 iter_1 c3: incumbent's kind entropy replaced by binary rule-vs-range entropy; shift −0.08±0.10 (null), RMSE 0.0089 vs incumbent, 0.0052 vs beta_share
metadata:
  type: project
---
exp3 iter_1/candidate_3 (2026-10-31), rule_vs_range_doubt_criterion: kind_uncertainty_shifted_criterion with criterion shift driven by binary entropy of P(rule) vs P(range) instead of the 9-kind entropy.
Quick 1×300 fit: shift −0.08±0.10 (incumbent +0.24±0.20), so the incumbent's signal comes from doubt among rule families, not rule-vs-range. Smoke OK, ll −3653.5.
Novelty script: /tmp/nov31/nov.py responses.csv a.py b.py (posterior-mean p_yes RMSE, ~3 min per pair).
**How to apply:** rule-vs-range doubt is ruled out as the criterion driver; see [[kind-uncertainty-shifted-criterion]].
