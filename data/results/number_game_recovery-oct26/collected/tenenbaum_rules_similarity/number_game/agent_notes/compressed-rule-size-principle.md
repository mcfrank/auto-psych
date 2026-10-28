---
name: compressed-rule-size-principle
description: exp3 iter_1 c1: plain rules judged as size**gamma under size principle (targets parity-outside-span critique); smoke LL -3094.2, 227s; check novelty outcome
metadata:
  type: project
---
2026-10-22, exp3 iter_1 candidate_1 `compressed_rule_size_principle`: copied the incumbent last_example_marks_range_top. rule_log_lik is now computed in-model from rule_hits (sets x rules) and n_examples, with perceived size |h|**gamma and gamma ~ Beta(4, 1.5). Ranges and conjunctions are unchanged. It is motivated by the parity_match_outside_span_gap critique (q=0.096). Check OK: smoke LL -3094.2, sampling 227 s.
**Why:** this is the opposite of range_discounted_size_principle, which made ranges look wider. **How to apply:** check whether it passed novelty and how it ranks before reusing the idea. Related: [[exp3-iter1-critique-stats]], [[novelty-gate-tiny-tweaks]]
