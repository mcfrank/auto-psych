---
name: concept-space-additions-null
description: iter_4 MAP profiles — off-by-one rules, multiples of 13-50, digit-sum concepts all rejected on uneven_rule_familiarity_ends base
metadata:
  type: project
---
2026-10-21, experiment3 iter_4 candidate_0 (concept-space brief). Base uneven_rule_familiarity_ends (MAP loglik -3186.35). Added concept family with prior share sigmoid(offset_logit), split within family by size_exponent; profiled at fixed logit via pm.do + find_MAP:
- Off-by-one rules (powers/squares/cubes ±1, >=3 members): logit -5 costs 90 nats, -2 costs 305. Data: 3,7 -> 31 only 20% yes. Free MAP logit -8.7 (prior N(-3,1)), loglik -3191.7, RMSE vs base 0.021. Submitted as off_by_one_rule_concepts (expected to lose).
- Including singleton/2-member shifted sets is even worse via the size principle.
- Multiples of 13..50: best at share->0 (logit -5: +0.1 nats; -3: -1.9).
- Digit sums (3+ members): -5: -2.7; -3: -41.
**Why:** the incumbent's 42-rule space looks saturated; extra rule families only steal belief.
**How to apply:** for concept-space briefs, don't re-screen these; script pattern iter_4/candidate_0/scratch/prof.py. See [[map-screening-refinements]], [[incumbent-saturation-screens]].
