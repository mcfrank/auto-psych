---
name: tempered-size-principle-overlap
description: weighting each shared property by (size/100)^-alpha on the progression_graded_magnitude_band incumbent; smoke loglik -3144.5
metadata:
  type: project
---
exp2 iter_3 candidate_2 (specificity_weighted_overlap): the incumbent overlap with each property's weight multiplied by (size/100)^(-alpha), alpha ~ HalfNormal(0.5). This is a tempered size principle. It targets the single_set_breadth_dispersion critique. The smoke fit passed with loglik -3144.5. Compare [[band-plus-primacy]] at -3137.6, so the gain is not clearly above the incumbent. Sampling ran slowly: 100+100 draws took about 70 s using T×J (5040×47) matrices.

**Why:** this records whether a size principle inside the overlap models helps.
**How to apply:** check its final ELPD before trying further specificity-weighting variants.
