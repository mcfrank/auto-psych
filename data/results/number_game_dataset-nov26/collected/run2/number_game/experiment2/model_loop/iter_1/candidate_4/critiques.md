# Critique of `seed_overshadowing_last_digit_recency`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_example_yes_gap** — Yes-rate on single-example sets minus yes-rate on multi-example sets (non-example targets only); observed above null means the model under-generalizes from one example (too narrow), below means it over-generalizes. (observed 0.147 vs null mean 0.0735, z=3.88, p=0.002, q=0.00799) [survives FDR]
- **early_session_yes_gap** — Yes-rate in the first 90 trials of a session (trial_index < 90) minus yes-rate in later trials; the model has no time dependence, so observed below null means people start more conservative and become more liberal over the session (and vice versa). (observed -0.041 vs null mean 0.0137, z=-3.25, p=0.002, q=0.00799) [survives FDR]
- **first_example_digit_lift** — Among multi-example sets without a shared final digit, yes-rate on non-example targets sharing the FIRST example's final digit (but not the last's) minus yes-rate on targets sharing no example's final digit; the model gives only the last example a lingering digit, so observed above null means a primacy digit effect the model misses. (observed 0.143 vs null mean 0.0813, z=2.17, p=0.03, q=0.0799)
