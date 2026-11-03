---
name: number-game-refinement-findings
description: Which refinements of the number-game Bayesian concept incumbent help or are dead ends (iter_2, 2026-10-30)
metadata:
  type: project
---
Tested in iter_2/candidate_4 (2026-10-30) against inferred_example_noise_concepts (MAP loglik -817.3, my per-row LOO -842.3..-842.9, p_loo ~22):
- Offset arithmetic progressions (c mod k) as an extra rule group: +0.4 MAP loglik only, so a near-duplicate.
- Bounded rules (rule ∩ interval): the share went to 0, so a duplicate.
- Family-only rule weights (7 families, equal within): MAP 16 nats worse.
- Hierarchical (partially pooled, non-centred) rule weights around family levels: LOO -841.8 (+0.6..1.2), p_loo 14, distinct predictions (RMSE 0.022). Submitted as family_pooled_rule_prior.
Data facts: 5 participants, each did the same 450-trial sequence (blocks of 450 rows), with no individual differences in yes-rate. Calibration by set size is already good and the residuals look like noise.
**Why:** the incumbent is near ceiling, so gains come from cutting effective parameters, not from new concepts.
**How to apply:** future refinements should target prior structure or parsimony; skip hypothesis-space additions. See [[interval-tempering-speed-trick]].
