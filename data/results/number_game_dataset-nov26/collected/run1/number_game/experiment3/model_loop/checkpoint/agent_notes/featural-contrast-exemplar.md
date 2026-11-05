---
name: featural-contrast-exemplar
description: exp3 iter1 c0 — Tversky contrast model over arithmetic/notational features, MAP -4474 vs incumbent -4098 (~375 behind); band (uniform-criterion) link gives 0 gain
metadata:
  type: project
---
Exploratory candidate featural_contrast_exemplar (2026-11-02). Each number is a set of features (parity, mult3-12, square, cube, pow2, prime, end digit, decade), with weights fixed at log(100/size). Pairwise sim = theta*common - a*target_only - b*example_only - c*dist, combined by logsumexp - kappa*log n. MAP loglik -4474; c_mag goes to ~0 because the decade feature takes over; smoke -4476.8. A HalfNormal on c_mag failed the warm-start Hessian check (mode at 0), and log-normal weights fixed it.
Also screened a decision-level change: a uniform-band criterion link on whole_set_round_category, MAP -4098.45 vs -4098.30, no gain.
Residuals: polarized cells the incumbent grades at 0.3-0.5 are mostly lone-example multiples (yes for 7→14, 16→64, 33→99, 22→44; no for 17→34, 43→86, 49→98, 13→78) and round or far route targets in mixed sets (75,95→round; 5,77,89→round; 3,33,99→mult of 3).
**Why:** cell polarization (q=0.018) is item-level and is not fixed by the link shape.
**How to apply:** don't retry link-shape tweaks for polarization; lone-example multiples split by example parity/familiarity is worth a look. See [[exp3-iter1-critique-stats]], [[biased-lapse-exemplar]].
