---
name: drifting-criterion-exemplar
description: exp3 iter0 c0 — incumbent round_number_averaging + yes-criterion drifting linearly with trial_index; smoke -4113.3
metadata:
  type: project
---
Decision-rule candidate aimed at the within_item_trial_drift critique ([[exp3-iter0-critique-stats]]). Inference = [[round-number-averaging-exemplar]] unchanged; alpha + drift*(trial_index/449 - 0.5). Needs prepare_observed (compute_features gets no trial_index); can't declare both hooks, so features are a private helper. Bare rows default to the session midpoint and rating 0.
**Why:** risk: novelty gate on bare stimuli sees ≈ incumbent predictions (RMSE < 0.002 → near-duplicate).
**How to apply:** if it's rejected as a duplicate, the drift idea needs a mechanism that also shows up on bare stimuli.
