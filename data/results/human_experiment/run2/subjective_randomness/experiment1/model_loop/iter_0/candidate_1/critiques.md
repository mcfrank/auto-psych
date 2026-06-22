# Critique of `prototype_similarity`

1 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **corr_periodicity_diff** — The Pearson correlation between the difference in periodicity (periodicity_a - periodicity_b) and the choice of A. (observed 0.238 vs null mean 0.156, z=3.04, p=0.00995, q=0.0796)
