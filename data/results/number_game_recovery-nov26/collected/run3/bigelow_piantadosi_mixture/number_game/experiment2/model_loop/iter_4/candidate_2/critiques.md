# Critique of `familiar_strays_equal_rule_prior`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **familiar_outlier_excuse_contrast** — In >=3-example sets where all but one example share a final digit or a parity, yes rate for non-example targets obeying that majority rule when the lone outlier is a familiar number (1, 100, power of 2, square, 27, multiple of 10) minus when it is unremarkable; observed below null means people do not excuse familiar outliers more than plain ones as much as the familiar-stray model predicts, above means they excuse them even more. (observed -0.105 vs null mean -0.00556, z=-2.38, p=0.024, q=0.192)
