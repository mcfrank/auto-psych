---
name: round-target-familiarity-bias
description: exp2 iter4 c0: example-independent logit bias for x10 targets replacing round kinship, MAP -6 nats vs incumbent; smoke -2682.1
metadata:
  type: project
---
exp2 iter_4 candidate_0 (exploratory): replaced round_number_kinship_exemplar's round route (x10 target ~ 0/5-ending examples) with a fixed logit bias for any multiple-of-10 target. Same param count; MAP log-lik -2676.1 vs incumbent -2670.2 (6 nats worse). Raw data: x10 far targets after sets with no 0/5 example still ~0.16-0.39 yes vs ~0.14 baseline, so some example-independent round appeal exists. Smoke ELPD -2682.1.
Cross-position digit overlap (38↔83) showed no raw effect (0.126 vs 0.107).

**Why:** tests whether round appeal is a target property or example-relative kinship ([[round-number-kinship-exemplar]]).
**How to apply:** if it loses, the round effect is example-relative; don't retry a pure target bias.
