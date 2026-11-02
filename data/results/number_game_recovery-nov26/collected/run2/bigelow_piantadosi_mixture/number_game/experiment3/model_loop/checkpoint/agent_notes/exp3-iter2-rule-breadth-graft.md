---
name: exp3-iter2-rule-breadth-graft
description: Exp3 iter-2 cand_3 — ~10 single-change screens on the upward-counting incumbent all tie (±0.3); rule-breadth graft +0.31 MAP, RMSE 0.0043, submitted; a loop variable `h` shadowed the new param
metadata:
  type: project
---
Exp3 iter 2, candidate_3 (2026-10-31), refinement slot on heading_weber_with_upward_counting. The base MAP LL is -3190.24.

MAP screens (ΔLL vs base):
- doubling chains in RULES: -27
- multiples of 13..50: -106. Adding rules into the equal-prior pool dilutes the prior.
- conservative evidence tau: tau→0.99, 0
- notation-rule weight: +0.07
- no-repeat discount: 0
- relative room above: -0.18
- relative room above and below: -0.03
- step-size heading: -0.06
- first→last trend heading: +0.18 (beta<0, RMSE 0.0049)
- rule breadth size^h: +0.31 (h=0.043, RMSE 0.0043)

Submitted `rule_breadth_heading_upward_counting`. Smoke -3193.5, about 60 s.

The per-cell deviance is about 650 over 1307 cells (5 observations each), so the model sits at the binomial ceiling.

Fast screening harness: /tmp/c3s/temper.py. It does a direct softmax over all 5113 hypotheses per set with no Taylor expansion, and it reproduces the incumbent's MAP exactly.

PITFALL: the incumbent builds its heading coefficients in a loop `for h in (-1, 0, 1)`. A new parameter named `h` gets overwritten and receives zero gradient. Use a different name.
See [[exp3-iter2-critique]], [[iter2-screens]].
