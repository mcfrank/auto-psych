---
name: exp2-iter2-incumbent-saturation
description: exp2 iter_2 c3: one-change variants of familiar_strays_family_level_prior all tie at MAP (-2031); submitted power-law familiarity strays
metadata:
  type: project
---
exp2 iter_2/candidate_3 (2026-11-01). The incumbent familiar_strays_family_level_prior has MAP ll -2030.98. Each variant below is one change, with its MAP ll and its p_yes RMSE against the incumbent:
- q ∝ familiarity^beta (power form, no extra param): -2030.86, RMSE .003, beta≈+0.24 (not negative). **Submitted** as graded_familiarity_strays. The check passed, with smoke fit 16 s.
- Size-dependent rule prior (log|r| slope): -2030.98, RMSE .0004.
- Stray generator weighted by family rule prior: -2030.97, RMSE .0009.
- First-example anchor: -2031.16, RMSE .014. Sibling candidate_4 took it.
- Tempered size principle s^-gamma (pass raw run counts, dot with size_pow in-graph): -2030.82, RMSE .0046.
- Readout "member OR familiar stray": reach→0.005, a dup.
Siblings c1 and c2 took "random-looking strays" (the negative-familiarity direction from the critique).
Harness: iter_2/candidate_3/scratch/cmp.py. To get p_yes, use per-obs logp; compile_fn on the p_yes Deterministic gives garbage.
**How to apply:** the incumbent family is saturated. Gains need a new kind of structure, not a reparameterisation. See [[exp2-incumbent-refinement-scan]], [[familiar-strays-family-level]].
