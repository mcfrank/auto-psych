---
name: gamma-width-scale-ridge
description: The gamma_width_* models' log_sigma runs off along a ridge (scale ~70, corr -0.94 with log_shape); reparameterise as a linear rate
metadata:
  type: project
---
In gamma_width_distance_hesitant_guess the MAP width scale is about exp(4.2)=70 on widths 1..100, so the prior is nearly a flat power law and log_sigma runs off along a non-Gaussian ridge (corr -0.94 with log_shape). The span-distance variant timed out at 30 min (iter_3 candidate_3, 2026-11-01). Fix used in candidate_3_repair_1: shape_m1 ~ Normal(0.5,1), rate = 2.4*shape_m1 + rate_offset, with rate_offset ~ Normal(0,1.5) and density w^shape_m1 * exp(-rate*w/100). Max |corr| fell to 0.59.

**Why:** warm_start_nuts uses a dense Hessian metric, so a linear correlation is handled; the curved, unbounded ridge is not ([[check-candidate-runtime]]).
**How to apply:** before submitting a refinement of a gamma_width model, run a MAP plus finite-difference Hessian check (/tmp/hess_c3.py pattern) and look for correlations near ±1 or parameters drifting toward a prior tail.
