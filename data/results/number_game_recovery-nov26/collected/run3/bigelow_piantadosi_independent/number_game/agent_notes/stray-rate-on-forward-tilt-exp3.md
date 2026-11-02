---
name: stray-rate-on-forward-tilt-exp3
description: exp3 iter4 cand4 — free stray rate on forward_continuation_response_bias +4.1 MAP loglik; spacing +3.1; tilt reshapes (decay/order/singleton/linear) ≤1.1
metadata:
  type: project
---
2026-10-31, exp3 iter_4/candidate_4 (`trusted_example_forward_continuation`). The iter_4 critique had 0 significant stats.
- Incumbent MAP loglik is -3411.6 (tilt 0.14, veto 0.19).
- Gains from the one-change screens:
  - free stray rate with prior Beta(2,18): +4.1 (rate 0.068). This was submitted. Check OK, smoke -3416.4, 140 s sampling.
  - mean-gap spacing shift: +3.1
  - order-dependent tilt: +1.1 (asc coef -0.11)
  - linear distance tilt, singleton-extra tilt, symmetric outside shift: ≤0.2
  - decay tilt: -0.6
**Why:** the tilt's form is exhausted, and only the known grafts (stray rate, spacing) still add a few nats.
**How to apply:** the variant scripts are in iter_4/candidate_4/scratch (v_*.py, map.py). Related: [[fitted-stray-rate-on-veto-exp3]], [[forward-continuation-response-bias]], [[exp3-iter4-critique-stats]].
