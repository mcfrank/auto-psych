# Critique of `capacity_limited_evidence_integration`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **two_example_dip** — Yes-rate on 2-example sets minus the mean yes-rate on 3- and 4-example sets; observed < null means people are more conservative on two-example sets (relative to larger sets) than the model's monotone capacity limit predicts. (observed -0.0678 vs null mean -0.0209, z=-2.85, p=0.002, q=0.016) [survives FDR]
