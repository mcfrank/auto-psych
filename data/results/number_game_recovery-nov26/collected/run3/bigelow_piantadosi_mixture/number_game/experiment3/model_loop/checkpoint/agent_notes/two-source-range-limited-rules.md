---
name: two-source-range-limited-rules
description: exp3 iter_1 c5: range-limited rules as 3rd main class in two_source_familiar_glitches, +1.1 MAP over parent, RMSE .0075 vs incumbent; narrow-range width priors collapse again
metadata:
  type: project
---
exp3 iter_1/candidate_5 (2026-11-01), two_source_range_limited_rules. It is two_source_familiar_glitches with a third main-concept class: blocks of rule members cut by ranges, weighted gaps[a]*gaps[b+1]/(45*5050) as in the incumbent. Class weights are [lambda_0(1-rho), 1-lambda_0, lambda_0*rho].
- MAP ll -3321.13 vs parent -3322.25. MAP values: rho .10, mix .04, g .29.
- Data-cell RMSE: .0075 vs the incumbent, .009 vs range_restricted. Smoke check OK in 12 s.
Critique narrow_rule_inhull_nonmember_span_contrast (q=.08): the gap is on the narrow side (obs .52 vs pred .41). All live models give .41-.43, and the gap is driven by 31,3,1,15→2,4,16 and other 5-trial cells. An exponential interval-width prior on range_restricted goes to sigma→270 (uniform). Erlang loses 3.7 nats. Don't retry width priors (see [[normative-distortions-collapse]]).
Harness: scratch/mapfit.py (MAP + p_yes via replace_rvs_by_values) and scratch/cmp.py (span contrast). See [[two-source-familiar-glitches]], [[range-limited-rule-familiar-strays]].
