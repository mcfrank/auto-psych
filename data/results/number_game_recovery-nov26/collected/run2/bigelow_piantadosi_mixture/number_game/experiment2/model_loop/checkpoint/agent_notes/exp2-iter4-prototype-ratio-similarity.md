---
name: exp2-iter4-prototype-ratio-similarity
description: Exp2 iter-4 similarity slot — Tversky-ratio prototype over concept features ~-2154 LL, best similarity model yet but ~230 nats behind
metadata:
  type: project
---
Exp2 iter-4 cand_2 (2026-10-30): `prototype_feature_ratio_similarity`. Features = 63 rules (incl. digit sums) + 5050 stretches; prototype weight share^kappa, salience size^-gamma, rule class weight; sim = shared/total; lapse+guess. Smoke LL -2154, kappa≈13 (≈ only all-shared features), gamma≈3.6. Grouping features by (size, rule?, share) gives 1200 columns → ~80 ms/grad, 2 chains×1000 in 164 s, R-hat 1.0, no divergences.
MAP screens: pure magnitude (hull/nn distance) -2840..-2980; Tversky ratio with target-distinctive term -2430..-2460.

**Why:** fixed salience (no n-dependent size principle) and no strays cap similarity models far below Bayesian ones (~-1925).
**How to apply:** for future similarity slots, reuse this grouping template; don't expect it to be competitive. Related: [[exp2-set-feature-contrast]], [[concept-exemplar-summed-similarity]].
