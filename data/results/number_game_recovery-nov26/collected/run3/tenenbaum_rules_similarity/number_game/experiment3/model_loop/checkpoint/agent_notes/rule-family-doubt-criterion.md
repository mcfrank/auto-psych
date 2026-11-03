---
name: rule-family-doubt-criterion
description: exp3 iter_2 c3: incumbent's kind entropy cut to P(rule)·H(family|rule) only; smoke OK, ll −3652.8
metadata:
  type: project
---
exp3 iter_2/candidate_3 (2026-10-31), rule_family_doubt_criterion: kind_uncertainty_shifted_criterion where the criterion shift uses only P(rule) × normalised entropy over the 8 rule families given a rule. The incumbent's kind entropy splits into H(rule vs range) + P(rule)·H(family|rule), so this is the complement of [[rule-vs-range-doubt-criterion]]. Smoke check OK (18 s sampling), ll −3652.8 (incumbent about −3653.3). Shift posterior and novelty were not checked.
**How to apply:** if it ties or wins, the incumbent's signal is in the within-rule doubt. If it is rejected as a duplicate, the incumbent is effectively already this. See [[kind-uncertainty-shifted-criterion]].
