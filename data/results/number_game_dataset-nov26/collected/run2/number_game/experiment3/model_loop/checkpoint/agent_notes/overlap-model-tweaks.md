---
name: overlap-model-tweaks
description: Which tweaks to the salience feature-overlap models gained or did nothing (iter_3 tests, 2-chain arviz LOO)
metadata:
  type: project
---
Tested 2026-11-02 on top of decimal_multiple_salience_overlap (8 families). Plain arviz LOO was -1477.4 here; the pipeline's grouped ELPD differs.
- Splitting parity into separate odd and even salience: about +6 nats (even 0.11 vs odd 0.04). Submitted as iter_3/candidate_5 parity_direction_salience_overlap.
- ~0 gain (iter_4/candidate_2, on the one_sided incumbent): a shared final digit scaling down arithmetic-family salience ("search termination"); stop param stayed near its prior, LOO -1464.95 vs -1465.19.
- ~0 gain: size-principle property weights, slope or intercept scaled by set size, span salience scaled by set size, each family counted once, a power on overlap, a "no pattern" constant in the denominator, a separate ends-in-0/5 family, position weighting with the decimal family added.
**Why:** this model space is nearly saturated, so new family splits are where gains came from.
**How to apply:** try salience-family splits before reshaping the combination rule. See [[pytensor-gather-speed]].

Iter_4 (2026-11-02), tested on one_sided_magnitude_bound_overlap (plain arviz LOO -1465.2, 2 chains):
- Extra logit boost when the target ends in a final digit shared by 2+ examples: -1462.0 (+3.2; fd_boost about 0.85). This addresses the critique that shared final digits are under-endorsed. Submitted as iter_4/candidate_4 shared_final_digit_rule_overlap.
- Odd/even parity split grafted on: -1461.5 (+3.7). Not submitted.

iter_4 tests (2026-11-02, plain arviz LOO, 2 chains; incumbent one_sided_magnitude_bound_overlap = -1465.2):
- Parity split plus one-sided bounds: -1461.2 (+4). Submitted as iter_4/candidate_5 parity_split_one_sided_bound_overlap.
- Tversky contrast logit (a + b*matched - c*missed) in place of the ratio: -1472.4 (-7). Worse.
- Shared-final-digit critique: the final-digit salience is already ~0.48. The under-endorsement comes from the ~0.69 ceiling for full matches (only the 28,98 set has these targets; its shared multiple of 7 dilutes the overlap). This is still unsolved.

Update 2026-11-02 (iter_4/candidate_3), on top of one_sided_magnitude_bound_overlap (plain arviz LOO -1465.2):
- Gain of about +4.5 nats (-1460.6): make the slope grow with the shared profile's total salience. The slope becomes slope + gain*10*den, which is the same as adding the absolute matched salience to the logit. Submitted as salience_scaled_decisiveness_overlap.
- ~0 gain: Tversky-style distinctive features of the target in the denominator (beta went to 0), a separate family for multiples of 7/11/12 (+4 at MAP but degenerate), and the slope times den^gamma.
- Why it works: every target that fully matches its set had a predicted yes-rate capped near 0.7. In the data, matches to sets that share only parity get about 0.6, and final-digit, 5/10 or power matches get 0.8-1.0.
