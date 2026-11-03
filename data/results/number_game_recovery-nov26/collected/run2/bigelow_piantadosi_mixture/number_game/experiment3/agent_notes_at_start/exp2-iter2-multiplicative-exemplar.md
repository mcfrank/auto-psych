---
name: exp2-iter2-multiplicative-exemplar
description: Exp2 iter-2 conjunctive (geometric-mean) concept-mediated exemplar model ~-2813 LL, ~880 nats behind; plain product worse (-2970) and degenerate
metadata:
  type: project
---
Exp2 iter 2, candidate 0 (2026-10-30): `multiplicative_exemplar_resemblance`. Pairwise similarity is P(y shares a concept | x alone) over the 45 rules plus intervals, with lambda free. The model takes the geometric mean over examples, raised to a power, with a lapse. MAP LL is -2813 (4 params), and NUTS on 2 chains gives R-hat 1.0 in seconds. The plain product (not normalized by the number of examples) scores -2970, with lambda→1 and lapse 0.52, which is degenerate.

**Why:** even a conjunctive exemplar rule cannot reproduce the set-level size principle. It does worse than the additive version on exp1 ([[concept-exemplar-summed-similarity]]) and the feature contrast on exp2 ([[exp2-set-feature-contrast]]).
**How to apply:** in an exemplar or prototype slot, submit a valid contrast model fast and don't tune it.
