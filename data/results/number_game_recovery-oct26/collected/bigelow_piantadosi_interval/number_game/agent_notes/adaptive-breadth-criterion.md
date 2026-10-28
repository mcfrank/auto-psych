---
name: adaptive-breadth-criterion
description: Decision-rule idea (iter_3 cand 2) — criterion shifts with log posterior expected concept size; MAP result
metadata:
  type: project
---
2026-10-21, experiment3 iter_3 candidate_2: adaptive_breadth_criterion = inflated_size_familiar_rule_prior inference (per-cell batched_dot layout from serial_position file, recency removed) + criterion = bias + slope * log(E_post[size]/100 / 0.2). One param, cheap (posterior·sizes per set).
MAP: loglik -3187.9 vs base -3190.1 (+2.2), slope +0.37 (broad concept → more liberal), p_yes RMSE vs base 0.0117 (novel).
**Why:** decision-rule slot; lapses/gain/bias/threshold/vote/batch rules were all already tried.
**How to apply:** set-level decision modulators are cheap and pass novelty; see [[map-screening-refinements]].
