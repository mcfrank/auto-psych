---
name: approximate-near-miss-rules
description: exp3 iter_4 c0: approximate rules (sparse rule ±1 neighbours) on variable_min_max incumbent, eta≈.02 stated prior, +0.84 MAP, pool RMSE .033 (distinct); hull boost drops 6.8→2.8
metadata:
  type: project
---
exp3 iter_4/candidate_0 (2026-11-02) is approximate_near_miss_rules. It takes the incumbent variable_min_max_bounded_range and adds, for each rule with ≤20 members, the set {members ±1}. After deduplication there are about 30 of these. They get share eta of the rule prior, with logit eta ~ N(-3.5, .3), split equally among them.
- Implemented by concatenating them onto the rule-membership table used for rule_sub and cell_rule_member. Restricted rules and familiarity still use only the 45 base rules.
- MAP ll: −3317.05 vs −3317.89 for the incumbent. eta .021, boost 6.8→2.8. The approximate rules absorb part of the hull effect.
- Pool RMSE .033, so it is clearly distinct. This contrasts with APs and digit sums, which collapsed ([[concept-additions-collapse]], [[bounded-arithmetic-progressions]]).
- Motivation: the 31,3,1,15 → 2,4,16 hint ([[two-source-variable-rule-expectation]]). Including dense rules (±1 of multiples of 3 covers 99 numbers) is pointless.
