---
name: variable-lambda-incumbent-saturation
description: exp3 iter_3 c4: one-change scan of variable_rule_range_expectation — nearly all collapse; one-sided graft +0.16 submitted; set-level lambda RE +72 MAP but is an item effect
metadata:
  type: project
---
exp3 iter_3/candidate_4 (2026-11-02). Incumbent variable_rule_range_expectation, MAP ll −3319.24. Scripts are in iter_3/candidate_4/scratch/w (diag.py, mapeval.py with fixed paths).
These MAP screens collapse or tie:
- lambda_sd depending on n: slope .04.
- Per-judgement variable rho: sd .007.
- Variable evidence weight (tempering per kind): w_sd → 0; alone −0.8.
- Beta in place of logit-normal lambda: ties, conc 10 (unimodal), pool RMSE .0015, a dup.
- Target-in-span cue on lambda: cue → 0.
- Restricted rules as uniform member-runs instead of range cuts: −0.5.
Two that fail other ways:
- "At most one glitch per set" reliability: −1059. Sets need ≥2 strays.
- Gist memory + variable lambda: −0.28, a dup of gist (RMSE .0011). The two substitute: lambda_sd .31.
- Set-level random effect on logit lambda: +72 MAP, set_sd 2.5, judgement sd → 0. But the pools contain all 255 sets (unseen ones too), so it is an item effect. No set feature correlates with it (|r| < .1). Not submitted.
Submitted: one_sided_variable_rule_expectation, i.e. the one-sided threshold share tau on the incumbent. tau .031, +0.16 MAP, pool RMSE .0061 vs the incumbent and .009 vs one_sided. Smoke fit 64 s.
The critique outside_rule_member_setsize_contrast is diffuse: n=2 out-of-span overpredicted by .03.
**How to apply:** the incumbent is saturated. Don't retry these. See [[variable-rule-range-expectation]], [[one-sided-threshold-ranges]], [[check-sibling-candidates-first]].
