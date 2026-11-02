# Critique of `variable_rule_range_expectation`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **outside_rule_member_setsize_contrast** — Yes-rate for non-example targets outside the span that belong to the smallest incumbent rule (<=25 members) covering the examples, sets of 3-4 examples minus sets of 2; observed < null means extra examples strengthen rule extrapolation less than the model's size principle predicts (observed > null: more). (observed 0.334 vs null mean 0.249, z=1.89, p=0.05, q=0.368)
