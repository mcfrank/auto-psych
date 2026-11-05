---
name: concept-space-additions
description: Extra concepts added to seed_overshadowing_last_digit_recency (exp2): upward arithmetic progression +8.6 loglik; repdigit +2; reversal/divisor/digit-sum/decade ~0
metadata:
  type: project
---
Tested 2026-11-02 (experiment2 iter_1/candidate_2). Each concept was added as one extra property with its own Dirichlet salience, overshadowed by other_w. Scores are posterior-mean-p log-lik, 2 chains × 300 draws. The incumbent scores -3154.3.
- Progression counted up from the smallest example by the gap (t >= min and (t-min) % gap == 0): -3145.7 (+8.6). Submitted as gap_step_progression_overlap.
- The same progression in both directions: +6.5. Only with 3+ examples: 0 (no exp2 sets have 3+ evenly spaced examples, so the whole gain comes from 2-example sets).
- Repdigits: +2.2. Digit reversal, divisors of the largest example, same digit sum, same decade: ~0.
- The single-example yes-gap critique (observed 0.59 vs 0.54 predicted) was not closed by any concept. It looks like a level/intercept effect, not a missing concept.
**Why:** this contradicts the "spacing ~0" result in [[seed-multiple-property]] (exp1). The exp2 sets include 8,16 / 2,8 / 52,24.
**How to apply:** for the single-example gap, try a set-size-dependent mechanism rather than a new concept. Note: pm.find_MAP on these Dirichlet models gave a bogus fit (-4820); use short NUTS instead.
