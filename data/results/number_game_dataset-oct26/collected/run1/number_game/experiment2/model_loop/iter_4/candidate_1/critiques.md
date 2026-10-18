# Critique of `rule_gated_decade_familiarity`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **trial_index_yes_slope** — OLS slope of rating on trial_index, in change of yes-probability per 100 trials; t_observed above null_mean (~0) means people drift toward saying yes over the session, which the stationary model cannot produce (below = drift toward no). (observed 0.0186 vs null mean 0.00378, z=2.93, p=0.00599, q=0.04) [survives FDR]
- **ruleless_multi_far_unmatched_yes** — In sets of >=2 examples with no explicit rule (mixed final digits, common factor <3), yes rate for non-example targets sharing no final digit with any example and lying >=2 decades outside the examples' band; t_observed above null_mean means people give rule-less sets a broad "anything goes" acceptance the model lacks; below means the model over-generalizes far from rule-less sets. (observed 0.519 vs null mean 0.382, z=2.46, p=0.00999, q=0.04) [survives FDR]
- **set_yes_rate_dispersion** — SD across number_sets of the set-level yes rate on targets that are not examples; t_observed above null_mean means generalization breadth still differs more between sets (some near all-no, others near all-yes) than the rule-gated model produces; below means the model over-differentiates sets. (observed 0.195 vs null mean 0.172, z=2.35, p=0.016, q=0.0426) [survives FDR]
