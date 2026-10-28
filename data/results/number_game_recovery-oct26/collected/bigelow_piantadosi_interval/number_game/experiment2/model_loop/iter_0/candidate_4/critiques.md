# Critique of `split_rule_range_tempering`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **lag1_response_autocorrelation** — Correlation between a participant's rating on consecutive trials (pairs within the same session only); observed > null means sequential response perseveration/assimilation the trial-independent model does not produce (observed < null means alternation). (observed 0.159 vs null mean 0.126, z=2.12, p=0.036, q=0.288)
