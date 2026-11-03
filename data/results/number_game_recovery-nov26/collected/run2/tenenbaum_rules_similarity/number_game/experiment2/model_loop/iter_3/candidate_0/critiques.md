# Critique of `extended_multiples_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **same_decade_near_contrast** — For non-example targets 1-5 away from the nearest example, yes rate when the target shares that example's tens decade (floor((n-1)/10)) minus when it crosses a decade boundary; observed above the null means people treat decades as category boundaries that the model's boundary-free intervals ignore (below = no such boundary effect, less than model). (observed 0.00542 vs null mean 0.0614, z=-2.29, p=0.02, q=0.16)
