---
name: recency-gated-arithmetic-route
description: exp2 iter4 c1 (order slot): arithmetic routes on round_number_kinship scaled by exp(-decay*lag); smoke -2664.8
metadata:
  type: project
---
Exp2 iter_4 candidate_1 (exploratory, order-effects brief): `recency_gated_arithmetic_route` = round_number_kinship_exemplar with multiples/factor-kin/round routes multiplied by exp(-decay * #examples shown after that example); decay ~ Normal(0,1) so primacy also allowed. Smoke log-lik -2664.8 (check OK).
Raw data hint: far multiples of small (2-12) examples endorsed 0.63 when the example is last vs 0.37 when first (confounded by set identity, n small).

**Why:** serial-position weighting of whole similarity was pruned in exp1 ([[serial-position-weighted-exemplar]]); gating only the arithmetic routes is the untested order variant.
**How to apply:** if it ties/wins, order effects live in arithmetic retrieval, not similarity; related [[exp2-iter4-critique-stats]].
