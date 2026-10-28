---
name: paired-final-digit-rules
description: exp3 iter_3 c4: "ends in d1 or d2" union rules as 6th kind on incumbent (from ones-digit critique, q=0.18); smoke -3093.3, 245s
metadata:
  type: project
---
2026-10-22, exp3 iter_3 candidate_4 (`paired_final_digit_rules`): copied sequence_opens_bottom_ends_top, added 45 two-digit final-digit union rules as kind 5 (Dirichlet a=1), excluded from conjunction slots (N_BASE_RULES=42 keeps MAX_SLOTS=12 intact). Check OK, smoke LL -3093.3, 245 s sampling.
**Why:** motivated by shared_ones_digit_outside_gap critique (not FDR-significant).
**How to apply:** check admission/ELPD outcome before adding more concept families; see [[concept-space-saturated]], [[digit-sum-concepts]].
