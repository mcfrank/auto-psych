# Critique of `prototype_similarity`

1 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **beta_max_run_norm** — Linear regression slope of chose_left on the difference in normalized max run length (max_run_norm_b - max_run_norm_a). (observed 0.0164 vs null mean 0.146, z=-1.96, p=0.0498, q=0.398)
