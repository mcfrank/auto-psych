# Critique of `extended_multiples_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **below_min_vs_above_max_extension** — Among sets of 2+ examples, yes rate for targets 1-15 below the smallest example minus yes rate for targets 1-15 above the largest example; observed above the null means people extend downward more readily than the model's symmetric intervals/rules predict (below = they extend upward more). (observed 0.139 vs null mean 0.0792, z=2.50, p=0.00799, q=0.0639)
