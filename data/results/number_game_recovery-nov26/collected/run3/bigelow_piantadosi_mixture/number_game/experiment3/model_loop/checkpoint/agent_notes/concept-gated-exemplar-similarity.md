---
name: concept-gated-exemplar-similarity
description: exp3 iter_0 c2 exemplar slot: p = P(in concept) * exp(-d_nearest_example/w) on rule_count_familiar_strays; w→∞ (MAP w≈8 prior-held), −1.9 MAP ll
metadata:
  type: project
---
exp3 iter_0/candidate_2 (2026-11-01): concept_gated_exemplar_similarity is the incumbent rule_count_familiar_strays with the readout multiplied by exp(-d/w). Here d is the target's distance to the nearest example, divided by 100, and log w ~ N(log .5, 1).
The data want no decay. At the MAP w = 8.2, held there only by the prior, and the MAP ll is −3323.7 against the incumbent's −3321.8. p_yes RMSE against the incumbent is .007 (max .05), so it may pass novelty.
Applying the decay to rule members only (to target the clean_rule_member_inside_minus_outside_hull critique, z=2.2, q=.29) gives the same ll and w≈7.4. Proximity-to-examples readouts are ruled out. Submitted anyway, with a 14 s smoke fit.
See [[range-restricted-rules]], [[concept-additions-collapse]].
