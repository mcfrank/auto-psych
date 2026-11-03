# Critique of `narrow_range_specificity_bias`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **moderate_cell_excess_disagreement** — Over (set, target) cells whose proxy P(yes) lies in (0.1,0.3) or (0.7,0.9), mean unbiased within-cell variance of the yes responses k(m-k)/(m(m-1)) minus proxy p(1-p); observed above null means respondents disagree more on moderately confident items than a single shared posterior implies (individual differences / response heterogeneity the model lacks). (observed 0.0147 vs null mean 0.00174, z=2.00, p=0.048, q=0.368)
