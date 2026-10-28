---
name: parity-kind-last-shown-top
description: exp3 iter_1 c3: even/odd as own rule kind on the incumbent, aimed at the parity-outside-span critique; smoke -3094.6; check outcome
metadata:
  type: project
---
On 2026-10-22, exp3 iter_1 candidate_3 (`parity_kind_last_shown_top_marking`) copied last_example_marks_range_top and changed only the rule kinds: even/odd became kind 5, so N_KINDS=6. It targets the parity_match_outside_span_gap critique (obs 0.248 vs null 0.184, q=0.096). The check passed: smoke LL -3094.6 (the incumbent's ELPD is about -3099), sampling took 237 s.
**Why:** parity_kind_rule_prior lost on the old exp2 base, so this tests the same idea on the current base.
**How to apply:** Check the novelty and ELPD outcome before retrying parity edits. Related: [[exp3-iter1-critique-stats]], [[write-candidate-first]]
