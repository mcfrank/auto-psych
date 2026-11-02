---
name: relative-concept-size-criterion
description: exp3 iter_3 c2: logit criterion shifted by logit of posterior expected concept share (b + c·z) on beta_share, smoke OK ll −3653.7
metadata:
  type: project
---
experiment3 iter_3 candidate_2 (2026-10-31), decision-rule brief: `relative_concept_size_criterion` = beta_share_width_exception_concepts learner; logit p_yes = logit m − bias − slope·(logit E[|h|]/100 − logit 0.25). E[|h|] is cheap from the posterior (rule sizes, interval sizes). Smoke ll −3653.7 (same as other exp3 smoke fits), 17 s/100+100 draws. Slope posterior not inspected — novelty risk vs beta_share if slope≈0 and bias≈0.
**Why:** decision-rule transforms on these data have been null repeatedly ([[experiment2-decision-rule-nulls]]).
**How to apply:** if this is rejected as a dup, decision-rule slots on exp3 are exhausted too; see [[uncertainty-damped-decisiveness]].
