# Critique of `outlier_tolerant_rule_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **trial_drift_yes_rate** — Yes-rate on trials in the second half of a session (trial_index >= median) minus the first half; observed > null means people become more liberal over the session, a drift the stationary model does not produce. (observed 0.0259 vs null mean 0.00771, z=1.94, p=0.05, q=0.352)
