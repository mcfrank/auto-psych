---
name: opening-example-marks-edge
description: exp3 iter_2 c4 (retry): redundancy (n**-r evidence) strongly rejected; set-size critique = 2-example sets over-generalize just outside; submitted opening-example edge marking (null at MAP)
metadata:
  type: project
---
2026-10-22, exp3 iter_2 candidate_4 (refinement slot, incumbent compressed_rule_extension).
1. Sublinear evidence: size-principle log-lik times n**(-redundancy). Exact via pooled (group, ll) classes, scratch/redundant_example_evidence.py, grad 0.058 s. Conditional LL is sharply peaked at 0 (±0.15 costs about 40 nats); MAP r=0.026. Don't retry tempering by set size.
2. set_size_near_outside_gap diagnosis (MAP p_yes, scratch/bys.py, ord2.py): for 2-example sets, yes 1-10 outside the span is 0.417 observed vs 0.477 predicted, on both sides and for mixed-parity sets too. n=3-4 sets are fine. The first-shown extreme seems to mark its edge (min-first below, max-first above are over-predicted), but cells are small.
3. Submitted `opening_example_marks_range_edge`: the incumbent plus generative credit, with one opening_odds, for ranges ending at a first-shown min (bottom) or max (top) when n>=2. Plain ranges are split into 4 parts. MAP LL -3091.5 vs incumbent -3091.2 (mapgen); RMSE 0.0067. Profile is flat below log-odds -4.4 and costs 9 nats at -3. Grad 0.051 s. Check OK: smoke LL -3094.8; with SAMPLER_SETTINGS target_accept 0.9 a smoke sampling run takes 140 s (267 s without).
**How to apply:** order-based edge credits are null at MAP; check this one's novelty outcome. Related: [[compressed-rule-extension]], [[first-shown-bottom-marking]], [[exp3-iter2-critique-stats]]
