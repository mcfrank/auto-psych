---
name: exp3-r4-leading-digit-refinements
description: Exp3 r4 MAP screen of single additions to leading_digit_string_similarity (MAP ll -4630.2)
metadata:
  type: project
---
Exp3 round 4 (2026-10-18), incumbent leading_digit_string_similarity MAP ll -4630.2. Gains from single additions:
- Even-number rule, taken from even_number_rule_abstraction: -4623.7 (+6.5). Submitted as leading_digit_even_rule_similarity (iter_4/candidate_3).
- Set-size criterion shift: +3.1.
- "Starts with d" leading-digit rule route: +0.8.
- Log-compressed out-of-span distance: 14 nats WORSE, even though the critique flags a log_magnitude gap.
**Why:** the even rule's gain shrank from +8.6 on the previous incumbent, so it partly overlaps the leading-digit dimension.
**How to apply:** don't retry log distance. Scripts are in iter_4/candidate_3/scratch (mapvals.py, plus an add() patch pattern). See [[exp3-r3-residual-scan]].
