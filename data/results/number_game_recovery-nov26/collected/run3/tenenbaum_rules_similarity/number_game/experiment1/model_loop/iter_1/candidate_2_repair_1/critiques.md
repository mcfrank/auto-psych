# Critique of `rule_family_prior_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **four_minus_two_example_yes_rate** — Yes-rate on 4-example sets minus yes-rate on 2-example sets; observed > model means people narrow their generalisation with more examples less than the size principle predicts (observed < model: they narrow more). (observed 0.114 vs null mean 0.167, z=-2.00, p=0.038, q=0.304)
