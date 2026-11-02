---
name: beta-share-prior-leak-graft
description: exp2 iter_1/candidate_3 — prior-leak judgment grafted onto beta_share incumbent: leak≈0.08±0.06, LOO −0.4; beta share already absorbs it
metadata:
  type: project
---
experiment2 iter_1/candidate_3 (2026-10-30), submitted as beta_share_width_prior_leaking_judgment. It is beta_share_width_exception_concepts plus the target prior log-odds shift from [[prior-leaking-judgment]]. Quick 2x300 NUTS: LOO -2273.9 vs incumbent -2273.6; lppd +0.7; leak 0.075 ± 0.063 (it was −0.15 on the gamma base, i.e. the beta share prior already captures the mid-range effect). p_yes RMSE vs the incumbent is 0.008 (passes novelty). Fit takes 31 s.
Incumbent residuals by set and cell look like binomial noise on n=5 cells (worst: 60,54→62; 69,9,39,21→19; 4,66,78,96→70/74).
**Why:** the gamma→beta gain and the prior-leak gain are the same effect; they do not stack.
**How to apply:** don't retry readout shifts on the beta-share base; see [[experiment2-incumbent-null-refinements]].
