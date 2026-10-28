# Critique of `uneven_rule_familiarity_ends`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **singleton_rule_far_generalization** — For single-example sets, among targets more than 10 away from the example, yes rate on targets sharing a non-parity rule (<=34 members) with the example minus targets sharing none; observed above null_mean means people generalize a lone example by rule more than the model predicts, below means the model over-extends rules from one example. (observed -0.0561 vs null mean 0.0838, z=-2.10, p=0.038, q=0.304)
