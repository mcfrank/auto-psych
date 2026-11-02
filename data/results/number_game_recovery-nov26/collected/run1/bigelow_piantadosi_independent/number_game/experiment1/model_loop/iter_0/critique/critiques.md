# Critique of `bigelow_piantadosi_mixture`

6 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **near_parity_violating_yes_rate** — Yes rate for non-example targets within 5 of the nearest example whose parity contradicts a single-parity set of >=2 (odd target for all-even sets, even for all-odd); observed > model means people generalise by numerical proximity more than the crisp rules + size principle allow. (observed 0.0741 vs null mean 0.258, z=-9.43, p=0.002, q=0.004) [survives FDR]
- **even_set_far_even_target_yes_rate** — Yes rate for even targets more than 10 from every example in all-even sets of >=2 examples; observed < model means people extend even/multiple-type rules to distant numbers less than the model (it over-weights rules relative to proximity). (observed 0.453 vs null mean 0.221, z=9.10, p=0.002, q=0.004) [survives FDR]
- **in_range_setsize_slope** — Least-squares slope of yes on number of examples among non-example targets inside the set's [min-5, max+5] window; observed less negative (higher) than model means people tighten generalisation with more examples less than the size principle predicts. (observed -0.0313 vs null mean 0.024, z=-4.53, p=0.002, q=0.004) [survives FDR]
- **shared_last_digit_yes_rate** — Yes rate for non-example targets whose last digit equals the common last digit of a set of >=2 examples all ending in the same digit; observed < model means people endorse the ends-in-d rule less strongly than the model. (observed 0.978 vs null mean 0.82, z=4.04, p=0.002, q=0.004) [survives FDR]
- **late_minus_early_yes_rate** — Yes rate on trials with trial_index >= 225 minus yes rate on trials < 225; an observed value outside the model's null (the model has no time dependence) means response bias drifts over the session (positive: more yes later). (observed -0.113 vs null mean -0.145, z=2.75, p=0.012, q=0.0192) [survives FDR]
- **far_parity_violating_yes_rate** — Yes rate for targets more than 15 from every example whose parity contradicts a single-parity set of >=2 examples; observed > model means people say yes to clearly off-concept numbers more often (a lapse/noise floor) than the model predicts. (observed 0.0338 vs null mean 0.0583, z=-2.11, p=0.028, q=0.0373) [survives FDR]
