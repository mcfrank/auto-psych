# Critique of `anchored_neighbourhood_side_channel`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **set_level_heterogeneity** — Standard deviation, across example sets with at least 20 non-example trials, of each set's non-example yes-rate; observed above the null means sets differ more in overall generalization breadth (some diffuse, some tight) than the model reproduces, below means the model exaggerates between-set differences. (observed 0.19 vs null mean 0.202, z=-2.11, p=0.032, q=0.256)
