---
name: wsls-variable-rule-range
description: exp3 iter_3 c0: WSLS running guess on variable_rule_range_expectation — stay .076, ties MAP (-0.5), pool RMSE .0029; serial-weighted kind belief pinned at w=1
metadata:
  type: project
---
exp3 iter_3/candidate_0 (2026-11-02) is win_stay_rule_range_guess. It is an exact WSLS recursion on the variable-lambda incumbent. All concepts are aggregated by (subset of shown positions, j) size^-j tables. It matches brute force to 3e-16; MAP run 93 s.
- MAP: stay .076, ll -3319.76 vs incumbent -3319.24; lambda_mu shifts -1.31 → -0.60. Pool RMSE .0029, so it is distinct.
- Tried first: serial-position weights w^(t-1) on the rule-vs-range log-Bayes-factor increments. Free w → 1.00 (near-dup, pool RMSE .0002). Stated w=.7 or 1.5 cost 65 nats; the normalized version at w=.9 cost 7 nats. Order weighting of the kind belief is ruled out; equal weighting is strongly identified, though mostly via set size, not order.
**How to apply:** for order-effect briefs, WSLS on the current incumbent is the order mechanism that stays distinct. See [[wsls-exact-recursion]], [[variable-rule-range-expectation]].
