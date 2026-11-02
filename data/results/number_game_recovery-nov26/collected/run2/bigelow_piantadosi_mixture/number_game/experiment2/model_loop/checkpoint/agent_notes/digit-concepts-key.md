---
name: digit-concepts-key
description: In the number-game data, "contains digit d" concepts drive much of the fit; Tenenbaum+digit rules hits ~-914 in-sample loglik with 2 params
metadata:
  type: project
---
Raw responses show digit-containment generalization (set 83,38,58,68 → yes to 84-89; 92,82,27,42 → yes to 20s), and distance-to-nearest-example barely matters. Adding 11 digit concepts (contains 0-9, ends in 0) to tenenbaum_rules_similarity (round 0, candidate_5, `tenenbaum_digit_concepts`) gave smoke-fit loglik -914.4, vs incumbent bigelow_piantadosi_independent ELPD-LOO -926.6 (45 free rule weights).

**Why:** the digit rules were the gap between Tenenbaum (-1249) and the incumbent, not the per-rule weights.
**How to apply:** future refinements should keep the digit concepts; free per-rule weights may be unnecessary.

Round 1 cand_1: `conservative_evidence_digit_concepts` (likelihood tempered by power beta, LogNormal(0,0.5)) smoke-fit loglik -913.8 — only ~0.6 better than the untempered model, so evidence tempering probably isn't where the gap is.

Round 1 (candidate_3) MAP tests on the incumbent: a free prior share for digit concepts fits omega≈0.255 (= the equal-weight 11/44, so no gain, novelty risk); freeing NOISE gives noise≈0.136, LL -910.8; a guess lapse gives lapse≈0.028, LL -909.0. Submitted the free-noise version (`digit_concepts_inferred_outlier_rate`).
