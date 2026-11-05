---
name: specificity-weighted-overlap
description: size-principle tilt inside overlap (weight x exp(gamma*n*log(100/size)/log100)) on exp2 incumbent; smoke loglik -3144.1
metadata:
  type: project
---
exp2 iter_3 candidate_0 (2026-11-02): `specificity_weighted_overlap` = progression_graded_magnitude_band with each shared property's weight
multiplied by exp(gamma * n_examples * specificity). Passed check; smoke log-lik -3144.1 (compare band+primacy -3137.6, [[band-plus-primacy]]).
Per-trial T x J matrices via prepare_observed; ~84 s per 200 NUTS draws (slower than compute_features models).
**Why:** overlap models ignore property extension size; this tests a coincidence-weighting variant distinct from [[coincidence-gated-pattern-belief]].
**How to apply:** check its final ELPD before building further on size-principle weighting.
