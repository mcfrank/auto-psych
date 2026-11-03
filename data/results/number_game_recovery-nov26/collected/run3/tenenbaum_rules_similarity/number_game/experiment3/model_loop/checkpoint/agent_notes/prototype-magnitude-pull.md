---
name: prototype-magnitude-pull
description: exp3 iter_3 c1 repair: logit pull toward the examples' mean magnitude, strength≈0.15, RMSE 0.0048 vs beta_share; the nearest-example, digit, parity and ratio pulls all collapse
metadata:
  type: project
---
exp3 iter_3 candidate_1_repair_1 (2026-10-31), prototype_magnitude_pull. The first attempt, [[nearest-example-proximity-pull]], was rejected as a duplicate of beta_share (RMSE 0.0012; MAP strength −0.05).
Pulls of the form strength·exp(-d/tau) on the beta_share base, at MAP (pool RMSE vs beta_share): nearest example 0.0019; distance outside the examples' span 0.0013; shared-digit 0.0004; parity mismatch 0.001; log-ratio 0.0012. Distance to the examples' mean gave strength 0.154, ll +0.75 and RMSE 0.0048, so I submitted it. SD-scaled centroid distance gave about the same. RMSE is 0.0059 vs rule_family_doubt and 0.0057 vs mixed_strong_weak. Smoke OK, ll −3653.6, 20 s.
Tool: /tmp/c1r1/mapfit.py <model.py>... fits the MAP with L-BFGS on value vars and reports the novelty-pool RMSE vs the first model. Use pytensor.function(model.value_vars, model.replace_rvs_by_values([p_yes])), not compile_fn: compile_fn's inputs were mis-ordered.
**How to apply:** in exp3, the only similarity cue left beyond ranges is centrality, not proximity to an example.
