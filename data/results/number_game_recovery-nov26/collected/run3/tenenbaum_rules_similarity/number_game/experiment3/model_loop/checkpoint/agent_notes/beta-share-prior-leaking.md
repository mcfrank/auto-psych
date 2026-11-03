---
name: beta-share-prior-leaking
description: exp2 iter_1 cand 5: prior_leaking judgment + beta-share width prior; smoke fit OK, ~17 s
metadata:
  type: project
---
experiment2 iter_1/candidate_5 (2026-10-30), beta_share_prior_leaking_judgment: prior_leaking_membership_judgment with its gamma width swapped for the incumbent's beta-share prior. It is built by text-replacing the source. The check passed: smoke log-lik -2268.7, 1 chain of 100+100 draws in 17 s. Full LOO not yet known.
**Why:** both pieces gave a small gain on their own (+1.3 and +5 LOO); this tests whether the gains add up. **How to apply:** check the manifest for its standing before repeating it. See [[prior-leaking-judgment]].
Outcome: rejected at admission as a near-duplicate of beta_share_width_prior_leaking_judgment (RMSE 0.00034); see [[beta-share-parity-range]].
