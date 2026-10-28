---
name: loosely-held-digit-rules
description: exp3 iter_3 c5: compressed_rule_extension + own inferred outlier rate for final/contained-digit rules (ones-digit critique); check outcome
metadata:
  type: project
---
2026-10-22, exp3 iter_3 candidate_5 (`loosely_held_digit_pattern_rules`) refines compressed_rule_extension. Digit rules (kinds 2 and 3) get their own noise rate digit_noise ~ Beta(2,6) in place of EPSILON=0.1, set with pt.switch on IS_DIGIT_RULE. Conjunctions keep EPSILON. It targets shared_ones_digit_outside_gap (q=0.18, not FDR-significant).
Check OK, smoke LL -3094.7, the check took about 9 min.
**Why:** It is the only critique hit this round. **How to apply:** Check the MAP digit_noise and the novelty result. If digit_noise stays near 0.1, it is a duplicate. Related: [[exp3-iter3-critique-stats]], [[compressed-rule-extension]]
