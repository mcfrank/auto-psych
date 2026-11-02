---
name: exp3-iter4-nearest-exemplar
description: Exp3 iter-4 exemplar slot — max-over-examples Shepard concept-mediated resemblance, MAP ~-4264, smoke LL -4248; 1.3 ms/grad
metadata:
  type: project
---
Exp3 iter-4 candidate 1 (2026-10-31): `nearest_exemplar_concept_resemblance`. For each (example, target) pair, resemblance = P(a concept containing x also contains t), computed over 63 rules (share lam) plus all intervals, with weights size^-gamma. Response = floor + (ceil-floor)·max over the examples. The rules × pairs matrices (2900 unique pairs) are built in prepare_observed, so a gradient takes 1.3 ms. MAP put lam≈0.64, gamma≈0.65, ceil≈0.89 and floor≈0.09; the smoke LL was -4248, about the same as the GCM and ~1050 nats behind. It was motivated by the prefix-broadening critique, and I submitted it untuned.
**How to apply:** max/mean/geometric-mean pairwise exemplar rules are now all tried; don't re-propose. Related: [[concept-exemplar-summed-similarity]], [[exp3-iter0-set-adaptive-attention-exemplar]].
