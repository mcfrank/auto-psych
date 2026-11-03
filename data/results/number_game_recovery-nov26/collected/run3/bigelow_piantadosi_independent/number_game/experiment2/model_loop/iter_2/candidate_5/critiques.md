# Critique of `hundred_chart_row_ranges`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **prime_set_prime_target_lift** — Among sets of 2+ examples that are all prime, yes rate for non-example prime targets minus non-example odd composite targets; observed above null means people generalise along primality more sharply than the model, below means less. (observed 0.221 vs null mean -0.187, z=4.32, p=0.002, q=0.016) [survives FDR]
