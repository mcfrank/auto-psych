# Critique of `bigelow_piantadosi_independent`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **far_outside_range_yes_rate** — Yes-rate for targets more than 20 below the smallest or above the largest example; observed above null means the model generalizes too narrowly beyond the example range (under-produces distant yeses), below means too broadly. (observed 0.215 vs null mean 0.193, z=1.98, p=0.05, q=0.4)
