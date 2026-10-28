# Critique of `outlier_tolerant_rule_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **late_minus_early_yes** — Yes-rate on trials in the top third of trial_index minus the bottom third; nonzero observed vs a ~0 null means the static model misses a drift in willingness to say yes over the session. (observed 0.0522 vs null mean 0.00843, z=2.96, p=0.002, q=0.016) [survives FDR]
