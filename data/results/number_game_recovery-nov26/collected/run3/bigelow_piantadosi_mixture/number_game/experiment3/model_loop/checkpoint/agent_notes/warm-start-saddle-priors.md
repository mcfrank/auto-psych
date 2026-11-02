---
name: warm-start-saddle-priors
description: warm_start_nuts "Hessian not positive definite" can mean L-BFGS-B stopped at a saddle/local mode from the prior initial point; fix by centring priors near the true mode
metadata:
  type: feedback
---
warm_start_nuts (src/models/pymc_inference.py `_warm_start_sample`) runs L-BFGS-B from model.initial_point (the prior medians) and fails if the Hessian there is not PD. In iter_2/candidate_1 (shared_property_contrast_similarity, 2026-10-30) the identified real mode (logp -950) was elsewhere; from the initial point it stopped at a saddle (-1058). Nelder-Mead from several starts found the mode; LogNormal priors centred near it fixed the check (6 s smoke fit, loglik -946).
**Why:** a "not PD" error looks like non-identifiability, but here it was really a bad starting point for the optimizer.
**How to apply:** before dropping parameters, compute the Hessian eigenvalues at a properly optimised mode; if they are all positive, move the prior medians toward that mode. See also [[interval-tempering-speed-trick]].
