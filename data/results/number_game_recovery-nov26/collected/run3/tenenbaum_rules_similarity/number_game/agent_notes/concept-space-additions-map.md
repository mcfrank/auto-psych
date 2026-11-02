---
name: concept-space-additions-map
description: Adding new rule families to arbitrary_exception_rule_prior barely moves MAP ll (~-1067); which additions were tried
metadata:
  type: project
---
Round 3 (2026-10-30), adding one new rule family (own Dirichlet weight) to the incumbent arbitrary_exception_rule_prior (MAP ll -1067.0):
- offset residue classes r mod k, k=3..6: -1066.7 (best; submitted as offset_progression_concepts, iter_3/candidate_0)
- same with k up to 12 or 25: -1067.2, weight -> 0
- multiples of 13..50: -1067.8; digit-sum = s: -1067.2 (weight 0.08); odd multiples/2-adic: -1067.2; decade blocks: -1067.3
After 69,9,39,21 the model still predicts 0.61 "yes" for 19 (observed 0/5), even with offset progressions in the space.
**Why:** the concept space is not the bottleneck. Remaining residuals are scattered single cells with n=5.
**How to apply:** don't spend another round on concept-space additions. See [[powers-of-2-37-misfit]].

Round 4 (2026-10-30), iter_4/candidate_1: a rule-vs-range prior log-odds that grows linearly with the number of examples (rule_noticing_grows_with_examples), added to parity_within_range_concepts. MAP ll -1063.98 for both models; the slope ≈ 0.03. The set size does not change how readily people think of a rule, so don't retry this.
