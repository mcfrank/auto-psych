# Critique of `decade_band_digit_common_factor`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **uniform_digit_set_same_parity_yes** — In sets of >=2 examples that all share one final digit, yes rate for targets of the same parity but a different final digit (not examples); t_observed below null_mean means people apply the shared-final-digit rule more sharply than the model's partial parity credit (model over-produces yes for parity-only matches). (observed 0.222 vs null mean 0.486, z=-3.81, p=0.002, q=0.016) [survives FDR]
- **trial_index_yes_slope** — OLS slope of rating on trial_index, in change of yes-probability per 100 trials; t_observed above null_mean (~0) means people drift toward saying yes more over the session, which the stationary model cannot produce (below = drift toward no). (observed 0.0186 vs null mean 0.0033, z=3.03, p=0.004, q=0.016) [survives FDR]
- **set_yes_rate_dispersion** — SD across number_sets of the set-level yes rate (targets that are not examples); t_observed above null_mean means people's overall generalization breadth differs more between sets (some sets almost all-no, others almost all-yes) than the model's features produce. (observed 0.195 vs null mean 0.171, z=2.48, p=0.016, q=0.0426) [survives FDR]
