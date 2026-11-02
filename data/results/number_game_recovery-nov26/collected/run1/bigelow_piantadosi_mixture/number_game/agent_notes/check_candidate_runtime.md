---
name: check-candidate-runtime
description: Runtime of check_candidate for sticky-guess concept-learning models (7-15 min)
metadata:
  type: reference
---
check_candidate on the gamma_width_* sticky-guess model family takes ~210 s per 100+100 smoke fit, run twice (~7 min, sometimes >10 min; 17 min wall / 10 min CPU on 2026-11-01); run it with run_in_background or a 600000 ms timeout. Copying the best model file and editing one component via a python string-replace script is a fast, reliable way to build refinements.

Admission uses warm_start_nuts: an L-BFGS mode, then a fixed dense mass matrix from the Hessian, with 3000 tune + 4000 draws x 4 chains. A parameter whose posterior piles against a bounded edge (e.g. a Beta "fade" factor near 1 = no effect) gives a skewed posterior that the Gaussian metric fits badly, and the fit exceeds 30 min. Prefer unconstrained parameters whose null value sits in the interior (e.g. Normal(0,1) decay).
