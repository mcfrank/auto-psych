---
name: rule-breaker-distance-indifference
description: exp3 iter4 c5: violators' magnitude gradient flattened by coincidence strength on coincidence_violation_exclusion; smoke -4037.9
metadata:
  type: project
---
Refines [[coincidence-violation-exclusion]]: the magnitude term for rule-breaking targets is multiplied by exp(-rule_indiff*coincidence). One extra parameter. Smoke log-lik -4037.9, about the same as the incumbent's -4036 ([[coincidence-rule-violation-route]]).
Raw data: digit-rule violators are flat in distance (near 0.136, far 0.136); parity-rule violators are graded (near 0.29, far 0.115). That is why the flattening scales with coincidence strength.
The single-example polarization critique looks like cell-level overdispersion with n=9 per cell (e.g. 7→14 at 1.0 but 49→98 at 0.11). It was not addressed.
**How to apply:** if this ties the incumbent, gating violators by rule type is close to used up.
