---
name: specificity-tilted-rule-prior
description: exp2 iter_2 cand_4: within-family rule prior ∝ size^-specificity on beta-share incumbent — tilt≈0.09±0.11, LOO +0.25, RMSE 0.0055
metadata:
  type: project
---
exp2 iter_2 candidate_4 (2026-10-30), aimed at critique stat nested_multiple_specificity (obs 0.49 > null 0.39, q=0.28).
Quick 2x300 NUTS: specificity 0.09±0.11 (null), elpd_loo -2273.3 vs incumbent -2273.6; RMSE of mean p_yes vs incumbent 0.0055 (passes novelty gate). Fit is fast (~11 s smoke).
**Why:** within-family specificity preference adds nothing on exp2; another null like [[experiment2-decision-rule-nulls]].
**How to apply:** don't retry rule-size tilts on exp2; the nested-multiples gap needs a different mechanism or more targeted stimuli.
