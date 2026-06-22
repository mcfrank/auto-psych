# Critique of `inner_loop_model`

1 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **length_difference_saturation** — Difference in preference for the longer sequence between trials with a large length difference (>=4) and trials with a small length difference (<=2). (observed -0.0075 vs null mean 0.207, z=-3.07, p=0.0199, q=0.159)
