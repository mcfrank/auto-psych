---
name: concept-family-quick-test
description: How to quickly screen a new concept family before submitting, and families already screened (digit-order rise/fall gets ~0 share)
metadata:
  type: project
---
Experiment 3, round 4 (2026-10-31): the base learner (graded_ratio_ceiling_criterion) is already well calibrated. Binned predicted vs observed yes-rates match, and the floor is about 0.005. A short NUTS run (2 chains, 300 tune / 200 draws, about 5 min on the 1-core box) gives a summed log-likelihood at the posterior-mean p_yes of about -1971.5. `pm.find_MAP` is badly off for this model (-2525), so don't use it for residuals.
Families screened: digit-order concepts (two-digit numbers whose digits rise or fall) got a share of about 0.004 and no gain in log-likelihood. Digit-sum concepts in this data just mirror multiples of 9. Unions of final digits are contradicted by 9,33,73,99 and 23,35,95,83, where people answer "odd".
**Why:** small concept families either dominate through the size principle or shrink to zero share. Zero share risks rejection as a near-duplicate at admission.
**How to apply:** screen a family with a short fit before you build a candidate on it. Prefer mechanisms that change predictions across many sets.
