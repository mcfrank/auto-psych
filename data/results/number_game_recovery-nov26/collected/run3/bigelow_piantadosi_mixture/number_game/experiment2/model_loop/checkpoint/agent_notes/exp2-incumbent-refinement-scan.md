---
name: exp2-incumbent-refinement-scan
description: experiment2 iter_0 MAP scan of one-change refinements to family_pooled_uncertain_reliability — all tie (±0.5 nats); the model is calibrated at ceiling
metadata:
  type: project
---
experiment2/iter_0/candidate_3 (2026-11-01). Incumbent MAP loglik is -2012.27 (alpha .92, kappa 20, lambda_0 .22). Each variant changes one thing:
- Digit-slip strays: -2012.76, eps .08.
- Without-replacement sampling: -2013.8.
- Interval width^gamma prior: -2012.25, gamma ≈ 0, so it's a near-dup (RMSE .0012).
- Log-odds slope: -2011.76, slope 1.05.
- Family-only weights (tau = 0): -2032.
- Strong/weak sampling mixture (rho .81): -2012.7, pool RMSE .0062. Submitted as strong_weak_sampling_mixture.
Item residuals show no overdispersion: the fraction of items answered 0/5 or 5/5 is .406 observed vs .405 expected. They show no feature bias either (parity, digits, hull, magnitude, examples all have |z| < 1.6). The data have 10 blocks of 450 trials (two item sequences, 5 participants each), and yes rates don't differ between participants.
Siblings in iter_0: c0 uses concept-type-specific reliability, c1 weak sampling, c2 a first-example anchor, c5 WSLS+reliability+family.
**How to apply:** don't rescan these; gains will need parsimony, not new cues. Scratch cmp.py/resid.py are in that candidate's scratch/. See [[number-game-refinement-findings]].
