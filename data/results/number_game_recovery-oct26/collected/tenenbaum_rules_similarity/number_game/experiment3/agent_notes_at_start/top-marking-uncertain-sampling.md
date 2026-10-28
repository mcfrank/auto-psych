---
name: top-marking-uncertain-sampling
description: exp2 iter_4 c4 grafted strong/weak sampling mixture onto incumbent top_marking_range_sampling; batched_dot speed trick; check outcome
metadata:
  type: project
---
2026-10-19, exp2 iter_4 candidate_4 (refinement slot, incumbent `top_marking_range_sampling`): `top_marking_uncertain_sampling`. All hypotheses (rules, intervals, conjunctions) are duplicated into strong and weak modes with Beta strong_prob. The weak-mode top credit uses a 1/100 draw probability. It reproduces the incumbent exactly when strong_prob is near 1.
Speed: doubling the hypotheses made the posterior[cell_set]*M gather 2x slower (0.10 s per gradient). Padding a broadcast multiply per set was worse (0.3 s). The fix: blocks of 8 targets per row, `pt.batched_dot(M[P,8,H], post[P,H,1])`, about 0.05 s per gradient.
Check OK: smoke LL -2054.6, 214-233 s per run under load ~40 (SAMPLER_SETTINGS target_accept 0.9 added after).
iter_4 critique: all 8 stats null, including the session-SD stats ([[session-heterogeneity]] is reproduced by the model).
Pitfall: compile_fn on model["p_yes"] draws from the priors unless you call replace_rvs_by_values first.
**How to apply:** reuse batched_dot for any large hypothesis-space model. Check this candidate's LOO before trying more sampling-assumption grafts. Related: [[uncertain-sampling-candidate]], [[pymc-model-speed-patterns]]
