---
name: exp3-iter3-tightest-shared-description
description: Exp3 iter-3 similarity slot — specificity-loss metric log(|tightest concept w/ target|/|w/o|), smoke LL -4012; pure magnitude metrics ~-4310..-4424
metadata:
  type: project
---
Exp3 iter-3 candidate 1 (2026-10-31): `tightest_shared_description_similarity`. Each set has a smallest rule (from 63) or interval covering its examples, and broadening is the log of how much that smallest concept grows once the target is added. sim = exp(-lam·d^shape), and p_yes runs from floor to ceiling. It has 4 params, and the 4 s smoke fit gave LL -4012 (better than the GCM's -4240 but ~800 behind the Bayesian models). The MAP fit put shape≈0.37 and ceiling≈0.64.
MAP screens of pure magnitude metrics (distance to the hull or to the nearest example, linear or log, width scaled by the set's spread) gave -4310 to -4424.
**How to apply:** if a similarity slot comes up again, this metric is the cheapest decent one. Related: [[exp3-iter0-set-adaptive-attention-exemplar]], [[exp2-iter4-prototype-ratio-similarity]].
