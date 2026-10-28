---
name: prototype-matching-poor-fit
description: exp3 iter_0 c2: information-weighted prototype (mean/spread + property proportions) smoke LL -4106, ~1000 nats worse than incumbent
metadata:
  type: project
---
On 2026-10-22, exp3 iter_0 candidate_2 (a prototype brief) submitted `information_weighted_prototype_matching`. The prototype is the examples' mean and variance plus the proportion of examples with each property. Mismatch is weighted by -log of the base rate. Smoke LL: the Gaussian magnitude version got -4210; the exponential spread-scaled distance version got -4106. The incumbent is about -3099. Sampling takes about 1 s because all features are computed in compute_features.
**Why:** this account is a lot worse than the exemplar accounts (about -2770 to -2900). Comparing to the centroid ignores the examples' span.
**How to apply:** for prototype briefs, expect a non-competitive fit. A centroid prototype fits worse than nearest-exemplar similarity. See [[exemplar-softmin-poor-fit]], [[attention-exemplar-similarity]].
