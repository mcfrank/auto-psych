---
name: prior-leak-on-beta-share
description: exp2 iter_1 cand4: prior-leaking judgment grafted onto beta_share_width_exception_concepts — contrast collapses to -0.08±0.06, LOO unchanged
metadata:
  type: project
---
experiment2 iter_1/candidate_4 (2026-10-30), prior_leaking_beta_share_concepts. Quick 2x300 NUTS LOO: incumbent beta-share -2273.57, graft -2273.76 (no gain). contrast -0.078 ± 0.064 (was -0.15 on the gamma base), mean_share 0.47, concentration 5.4. p_yes RMSE vs incumbent 0.0086. Fit ~30 s.
**Why:** the beta-share width prior already absorbs what prior leakage captured on the gamma base.
**How to apply:** don't graft prior leakage onto beta-share again; see [[prior-leaking-judgment]], [[experiment2-incumbent-null-refinements]].
