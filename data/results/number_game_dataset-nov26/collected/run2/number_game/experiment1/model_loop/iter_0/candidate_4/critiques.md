# Critique of `bigelow_piantadosi_independent`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **outside_range_yes_rate** — Yes-rate for targets lying more than 10 beyond the [min, max] range of the examples; observed > null means the model under-generalizes far outside the example range (too interval-bound), observed < null means it over-generalizes there. (observed 0.431 vs null mean 0.497, z=-2.50, p=0.018, q=0.144)
