---
name: prototype-pull-rule-family-doubt
description: exp3 iter_4 c4: centroid pull (prototype_magnitude_pull) grafted onto rule_family_doubt incumbent, smoke OK ll −3653.4; critique's session-time stat not addressable
metadata:
  type: project
---
exp3 iter_4/candidate_4 (2026-10-31), prototype_pull_rule_family_doubt: incumbent logit + strength·exp(-|t-mean|/tau). Smoke OK in 18 s, ll −3653.4. Posterior and novelty were not checked.
The only FDR-significant critique stat was late_vs_early_discrimination, which needs trial_index. It was not used, see [[sample-size-shifted-criterion]].
**How to apply:** if it is rejected as a dup, the centroid pull does not survive on top of the doubt shift. See [[prototype-magnitude-pull]] and [[rule-family-doubt-criterion]].
