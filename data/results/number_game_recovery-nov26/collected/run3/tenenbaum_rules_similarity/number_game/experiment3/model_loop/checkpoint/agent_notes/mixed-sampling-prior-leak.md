---
name: mixed-sampling-prior-leak
description: exp2 iter_4 c4 repair: prior-leak graft on mixed_strong_weak incumbent — leak 0.06±0.07, pool RMSE 0.004 vs incumbent; set-level version had been a dup of beta_share (0.0019)
metadata:
  type: project
---
exp2 iter_4/candidate_4_repair_1 (2026-10-31), submitted as mixed_sampling_prior_leaking_judgment: incumbent mixed_strong_weak_sampling_concepts + the logit prior-membership leak from beta_share_width_prior_leaking_judgment. Smoke OK (4.5 min).
Quick 2x300 fits, novelty-pool posterior-mean RMSE: cand vs incumbent 0.0041, vs beta_share leak 0.0046, vs beta_share 0.0075. theta 0.91, leak 0.06±0.07 (so null-ish).
The first attempt (set-level strong/weak, [[set-level-strong-weak-sampling]]) was rejected: RMSE 0.0019 vs beta_share.
Script: /tmp/nov4/nov.py fits models and compares p_yes on model_loop/novelty_pool.json (set_data + sample_posterior_predictive on p_yes works).
**How to apply:** before submitting a repair, run this pool-RMSE check against the nearest live models.
