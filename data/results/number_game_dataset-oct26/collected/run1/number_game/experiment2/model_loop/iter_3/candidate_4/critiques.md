# Critique of `final_digit_rule_suppresses_parity`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **trial_index_yes_slope** — OLS slope of rating on trial_index, in change of yes-probability per 100 trials; t_observed above null_mean (~0) means people drift toward saying yes more over the session, which the stationary model cannot produce (below = drift toward no). (observed 0.0186 vs null mean 0.00261, z=3.15, p=0.004, q=0.032) [survives FDR]
- **single_example_same_decade_other_digit_yes** — In single-example sets, yes rate for targets in the same decade (t//10 == x//10) as the example but with a different final digit; t_observed above null_mean means people generalize within a decade more than the model (digit similarity + band) predicts; below means the model over-produces same-decade yeses. (observed 0.426 vs null mean 0.277, z=2.39, p=0.028, q=0.0799)
- **set_yes_rate_dispersion** — SD across number_sets of the set-level yes rate (targets that are not examples); t_observed above null_mean means generalization breadth differs more between sets (some nearly all-no, others nearly all-yes) than the model's features produce; below means the model over-differentiates sets. (observed 0.195 vs null mean 0.173, z=2.15, p=0.03, q=0.0799)
